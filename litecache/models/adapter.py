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

        hf_config = model.config
        self.model_config = ModelConfig.from_pretrained_config(hf_config)

        if cache_config is None:
            cache_config = self.model_config.to_cache_config(
                num_blocks=256,
                block_size=16,
                device=str(self.device),
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
            _ = self.model(input_ids=input_ids, use_cache=True)

        for layer_idx in range(self.model_config.num_hidden_layers):
            if layer_idx not in self.captured_kv:
                raise RuntimeError(
                    f"Layer {layer_idx} not in cache",
                    "Check hook setup"
                )

            key, value = self.captured_kv[layer_idx]
            if key.dim() == 4:
                key = key.squeeze(0).transpose(0, 1)
                value = value.squeeze(0).transpose(0, 1)

    def generate(
            self,
            input_ids: torch.Tensor,
            max_length: int = 50,
            temperature: float = 1.0,
            do_sample: bool = False,
    ) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise ValueError("Only batch size of 1 is supported")

        input_ids = input_ids.to(self.device)
        seq_id = 1
        prompt_len = input_ids.shape[1]
        self._prefill(input_ids, seq_id)

        generated_ids = input_ids.clone()
        max_new_tokens = max_length - prompt_len
        for i in range(max_new_tokens):
            next_token_id = self._decode_one_token(seq_id, temperature, do_sample)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            if hasattr(self.model_config, 'eos_token_id'):
                if next_token_id.item() == self.model_config.eos_token_id:
                    break

        self.cache.free_sequence(seq_id)
        return generated_ids

    @abstractmethod
    def _decode_one_token(self, seq_id: int, temperature: float, do_sample: bool) -> torch.Tensor:
        raise NotImplementedError("Work In Progress")
