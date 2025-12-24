from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CacheConfig:
    # Block Configuration
    block_size: int = 16
    num_blocks: int = 1024

    # Model Architecture
    num_heads: int = 32
    num_kv_heads: Optional[int] = None
    head_dim: int = 128
    num_layers: int = 32

    # Device Config
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    # Optional features
    use_triton: bool = False
    max_seq_len: int = 2048

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads {self.num_heads} must be divisible by num_kv_heads for Grouped Query Attention")

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        if self.device.startswith("cuda"):
            try:
                import triton
                self.use_triton = True
            except ImportError:
                self.use_triton = False
                print("Warning: Triton not available, using PyTorch fallback kernels")

        else:
            self.use_triton = False

        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.max_seq_len < self.block_size:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) should be >= block_size ({self.block_size})"
            )

        total_memory_gb = self.total_cache_memory_gb
        if total_memory_gb > 100:
            print(
                f"Warning: Total cache memory is {total_memory_gb:.2f} GB. "
                f"This may exceed available memory. Consider reducing num_blocks or num_layers."
            )

    @property
    def total_cache_memory_gb(self) -> float:
        bytes_per_element = torch.tensor([], dtype=self.dtype).element_size()
        memory_per_layer = (
                self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim * 2 * bytes_per_element)
        total_memory = memory_per_layer * self.num_layers
        return total_memory / (1024 ** 3)

    @property
    def total_cache_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def get_max_blocks_per_sequence(self) -> int:
        return (self.max_seq_len + self.block_size - 1) // self.block_size

    def validate_sequence_length(self, seq_len: int) -> None:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )
        if seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, got {seq_len}")

    def get_num_query_heads_per_kv_head(self) -> int:
        return self.num_heads // self.num_kv_heads

    def __repr__(self) -> str:
        """Pretty print configuration summary."""
        gqa_ratio = self.get_num_query_heads_per_kv_head()
        attention_type = "MHA" if gqa_ratio == 1 else f"GQA ({gqa_ratio}:1)"

        return (
            f"CacheConfig(\n"
            f"  Memory: {self.total_cache_memory_gb:.2f} GB, "
            f"Capacity: {self.total_cache_tokens:,} tokens\n"
            f"  Blocks: {self.num_blocks} × {self.block_size} tokens/block\n"
            f"  Architecture: {self.num_layers} layers, {attention_type}\n"
            f"    Heads: {self.num_heads} query, {self.num_kv_heads} KV, dim={self.head_dim}\n"
            f"  Compute: device={self.device}, dtype={self.dtype}, triton={self.use_triton}\n"
            f"  Limits: max_seq_len={self.max_seq_len}, "
            f"max_blocks_per_seq={self.get_max_blocks_per_sequence()}\n"
            f")"
        )


@dataclass
class ModelConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    num_hidden_layers: int = 32
    max_position_embeddings: int = 2048

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained_config(cls, config) -> 'ModelConfig':
        return cls(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_kv_heads", None),
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=getattr(config, "max_position_embeddings", 2048),
        )

    def to_cache_config(self,
                        num_blocks: int = 1024,
                        block_size: int = 16,
                        dtype: torch.dtype = torch.bfloat16,
                        device: Optional[str] = None,
                        **kwargs) -> CacheConfig:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        return CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks,
            dtype=dtype,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_layers=self.num_hidden_layers,
            device=device,
            max_seq_len=self.max_position_embeddings,
            **kwargs
        )

    def __repr__(self) -> str:
        """Pretty print model configuration."""
        gqa_ratio = self.num_attention_heads // self.num_key_value_heads
        attention_type = "MHA" if gqa_ratio == 1 else f"GQA ({gqa_ratio}:1)"

        return (
            f"ModelConfig(\n"
            f"  hidden_size={self.hidden_size}, head_dim={self.head_dim}\n"
            f"  {attention_type}: {self.num_attention_heads} query heads, "
            f"{self.num_key_value_heads} KV heads\n"
            f"  layers={self.num_hidden_layers}, "
            f"max_pos={self.max_position_embeddings}\n"
            f")"
        )


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("Example 1: Basic CacheConfig")
    print("=" * 80)
    config = CacheConfig(
        block_size=16,
        num_blocks=1024,
        num_heads=32,
        num_kv_heads=8,  # GQA with 4:1 ratio
        head_dim=128,
        num_layers=32,
        dtype=torch.float16,
        max_seq_len=2048,
    )
    print(config)
    print()

    print("=" * 80)
    print("Example 2: ModelConfig from scratch")
    print("=" * 80)
    model_config = ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
        max_position_embeddings=4096,
    )
    print(model_config)
    print()

    print("=" * 80)
    print("Example 3: Convert ModelConfig to CacheConfig")
    print("=" * 80)
    cache_config_from_model = model_config.to_cache_config(
        num_blocks=2048,
        block_size=32,
        dtype=torch.bfloat16,
    )
    print(cache_config_from_model)
    print()

    print("=" * 80)
    print("Example 4: Validation examples")
    print("=" * 80)
    try:
        # This should work
        config.validate_sequence_length(1024)
        print("✓ Sequence length 1024 is valid")

        # This should fail
        config.validate_sequence_length(3000)
    except ValueError as e:
        print(f"✗ Validation error: {e}")
    print()

    print("=" * 80)
    print("Example 5: CPU configuration")
    print("=" * 80)
    cpu_config = CacheConfig(
        block_size=16,
        num_blocks=256,  # Smaller for CPU
        num_heads=12,
        head_dim=64,
        num_layers=12,
        dtype=torch.float32,  # FP32 for CPU
        device="cpu",
        max_seq_len=1024,
    )
    print(cpu_config)
