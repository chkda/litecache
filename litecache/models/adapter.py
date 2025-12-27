from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

import torch
from transformers import PreTrainedModel

from litecache.cache.paged_attention import PagedAttentionCache
from litecache.config import CacheConfig, ModelConfig


class LiteCacheAdapterBase(ABC):

    def __init__(self, model: PreTrainedModel, cache_config: Optional[CacheConfig] = None):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        hf_config = model.config
        self.model_config = ModelConfig.from_pretrained_config(hf_config)

        if cache_config is None:
            cache_config = self.model_config.to_cache_config(
                num_blocks=256,
                block_size=16,
                device=str(self.device),
                dtype=self.dtype,
            )

        self.cache_config = cache_config

        self.cache = PagedAttentionCache(cache_config)

        self.captured_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        self._setup_hooks()

        print(f"LiteCache Adapter initialized:")
        print(f"  Model: {hf_config.model_type}")
        print(f"  Layers: {self.model_config.num_hidden_layers}")
        print(f"  Heads: {self.model_config.num_attention_heads} query, "
              f"{self.model_config.num_key_value_heads} KV")
        print(f"  Cache: {cache_config.num_blocks} blocks × {cache_config.block_size} tokens")
        print(f"  Memory: {cache_config.total_cache_memory_gb:.2f} GB")

    @abstractmethod
    def _setup_hooks(self):
        pass

    @abstractmethod
    def _get_layers(self):
        pass

    def _prefill(self, input_ids: torch.Tensor, seq_id: int):
        seq_len = input_ids.shape[1]

        self.cache.allocate_sequence(seq_id, num_tokens=seq_len)
        self.captured_kv.clear()

        with torch.no_grad():
            # Get embeddings
            hidden_states = self.model.transformer.wte(input_ids)
            position_ids = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
            position_embeds = self.model.transformer.wpe(position_ids)
            hidden_states = hidden_states + position_embeds

            # Process through each layer and capture KV
            for layer_idx, layer in enumerate(self.model.transformer.h):
                # Layer norm
                normed = layer.ln_1(hidden_states)

                # Compute Q, K, V
                qkv = layer.attn.c_attn(normed)
                hidden_size = self.model.config.n_embd
                q, k, v = qkv.split(hidden_size, dim=2)

                # Reshape to multi-head format
                num_heads = self.model.config.n_head
                head_dim = hidden_size // num_heads

                k = k.view(1, seq_len, num_heads, head_dim)
                v = v.view(1, seq_len, num_heads, head_dim)

                # Store in format [seq_len, num_heads, head_dim]
                k_cache = k.squeeze(0)
                v_cache = v.squeeze(0)

                self.captured_kv[layer_idx] = (k_cache, v_cache)

                # Complete the forward pass for this layer
                # (You need to compute attention and MLP to get correct hidden_states)
                # For now, just run the full layer
                hidden_states = layer(hidden_states)[0]

        # Store all captured KV in cache
        for layer_idx in range(self.model_config.num_hidden_layers):
            if layer_idx not in self.captured_kv:
                raise RuntimeError(f"Layer {layer_idx} not in cache")

            key, value = self.captured_kv[layer_idx]
            self.cache.set_kv_cache(
                layer_idx,
                seq_id,
                key,
                value,
                0
            )

    def generate(
            self,
            input_ids: torch.Tensor,
            max_length: int = 50,
            temperature: float = 1.0,
            do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Generate text using paged attention cache.

        Args:
            input_ids: Input token IDs [1, seq_len]
            max_length: Maximum total length
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy

        Returns:
            Generated token IDs [1, total_length]
        """
        if input_ids.shape[0] != 1:
            raise ValueError("Only batch size of 1 is supported")

        input_ids = input_ids.to(self.device)
        seq_id = 1

        # Prefill: Process input prompt
        prompt_len = input_ids.shape[1]
        self._prefill(input_ids, seq_id)
        print(f"✓ Prefill complete: {prompt_len} tokens cached")

        # Start with input tokens
        generated_ids = input_ids.clone()
        last_token_id = input_ids[0, -1].item()

        # Decode: Generate tokens one by one
        max_new_tokens = max_length - prompt_len
        for i in range(max_new_tokens):
            # Generate next token using our cache
            next_token_id = self._decode_one_token(
                seq_id=seq_id,
                last_token_id=last_token_id,
                temperature=temperature,
                do_sample=do_sample
            )

            # Append to sequence
            generated_ids = torch.cat([
                generated_ids,
                next_token_id.unsqueeze(0).unsqueeze(0)
            ], dim=1)

            # Update last token
            last_token_id = next_token_id.item()

            # Stop if EOS token
            if hasattr(self.model.config, 'eos_token_id'):
                if next_token_id.item() == self.model.config.eos_token_id:
                    break

        # Cleanup
        self.cache.free_sequence(seq_id)
        print(f"✓ Generation complete: {generated_ids.shape[1]} total tokens")

        return generated_ids

    @abstractmethod
    def _decode_one_token(self, seq_id: int, temperature: float, last_token_id: int, do_sample: bool) -> torch.Tensor:
        raise NotImplementedError("Work In Progress")


class GPT2Adapter(LiteCacheAdapterBase):

    def _setup_hooks(self):
        def make_hook(layer_idx: int):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    present = output[1]
                    if isinstance(present, tuple) and len(present) == 2:
                        key, value = present
                        self.captured_kv[layer_idx] = (key, value)

            return hook

        layers = self._get_layers()
        print(f"Registering hooks on {len(layers)} layers")
        for layer_idx, layer in enumerate(layers):
            layer.attn.register_forward_hook(make_hook(layer_idx))
            print(f"Registered hook on layer {layer_idx}")

    def _get_layers(self):
        return self.model.transformer.h

    def _decode_one_token(
            self,
            seq_id: int,
            last_token_id: int,
            temperature: float,
            do_sample: bool,
    ) -> torch.Tensor:
        """
        Generate one token using paged attention cache.

        Args:
            seq_id: Sequence identifier
            last_token_id: The last generated token ID
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy

        Returns:
            Next token ID (scalar tensor)
        """
        # Get current position in sequence
        current_len = self.cache.block_manager.sequence_manager.get_num_tokens(seq_id)

        # Allocate space for new token
        self.cache.append_slots(seq_id, num_tokens=1)

        # Prepare input: just the last token
        input_ids = torch.tensor([[last_token_id]], device=self.device)

        # Get embeddings
        hidden_states = self.model.transformer.wte(input_ids)  # [1, 1, hidden_size]
        position_ids = torch.tensor([[current_len]], device=self.device)
        position_embeds = self.model.transformer.wpe(position_ids)
        hidden_states = hidden_states + position_embeds

        # Process through each layer
        for layer_idx, layer in enumerate(self.model.transformer.h):
            # Layer norm
            normed = layer.ln_1(hidden_states)

            # Compute Q, K, V for the new token
            # GPT-2's c_attn computes Q, K, V together
            qkv = layer.attn.c_attn(normed)  # [1, 1, 3*hidden_size]

            # Split into Q, K, V
            hidden_size = self.model.config.n_embd
            q, k, v = qkv.split(hidden_size, dim=2)

            # Reshape to multi-head format
            # [1, 1, hidden_size] -> [1, 1, num_heads, head_dim]
            num_heads = self.model.config.n_head
            head_dim = hidden_size // num_heads

            q = q.view(1, 1, num_heads, head_dim)
            k = k.view(1, 1, num_heads, head_dim)
            v = v.view(1, 1, num_heads, head_dim)

            # Reshape for our cache: [seq_len, num_heads, head_dim]
            k_cache = k.squeeze(0)  # [1, num_heads, head_dim]
            v_cache = v.squeeze(0)  # [1, num_heads, head_dim]

            # Store new K, V in cache
            self.cache.set_kv_cache(
                layer_idx=layer_idx,
                seq_id=seq_id,
                key=k_cache,
                value=v_cache,
                start_pos=current_len
            )

            # Prepare for attention computation
            q_attn = q.squeeze(0)  # [1, num_heads, head_dim]

            # Get block table and context length
            block_table = torch.tensor([self.cache.get_block_table(seq_id)], device=self.device)
            context_lens = torch.tensor([current_len + 1], device=self.device)

            # Compute attention using our paged cache
            attn_output = self.cache.compute_attention(
                layer_idx=layer_idx,
                query=q_attn,
                block_tables=block_table,
                context_lens=context_lens
            )  # [1, num_heads, head_dim]

            # Reshape back to [1, 1, hidden_size]
            attn_output = attn_output.transpose(0, 1).unsqueeze(0)  # [1, 1, num_heads, head_dim]
            attn_output = attn_output.reshape(1, 1, hidden_size)

            # Output projection
            attn_output = layer.attn.c_proj(attn_output)

            # Residual connection
            hidden_states = hidden_states + attn_output

            # MLP
            hidden_states = hidden_states + layer.mlp(layer.ln_2(hidden_states))

        # Final layer norm
        hidden_states = self.model.transformer.ln_f(hidden_states)

        # Get logits
        logits = self.model.lm_head(hidden_states)  # [1, 1, vocab_size]
        logits = logits.squeeze(1)  # [1, vocab_size]

        # Sample next token
        if do_sample:
            # Apply temperature
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze()
        else:
            # Greedy
            next_token_id = torch.argmax(logits, dim=-1).squeeze()

        return next_token_id


def create_adapter(model: PreTrainedModel, cache_config: Optional[CacheConfig] = None) -> LiteCacheAdapterBase:
    ADAPTER_REGISTRY = {
        "gpt2": GPT2Adapter,
    }
    model_type = model.config.model_type
    if model_type not in ADAPTER_REGISTRY:
        supported = ", ".join(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Model type '{model_type}' is not supported. "
            f"Supported types: {supported}"
        )

    adapter_class = ADAPTER_REGISTRY[model_type]
    return adapter_class(model, cache_config)
