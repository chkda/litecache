from typing import List

import torch

from litecache.block_manager import BlockManager
from litecache.cache.base import CacheBase
from litecache.config import CacheConfig
from litecache.kernels.torch_ops import paged_attention_torch


class PagedAttentionCache(CacheBase):

    def __init__(self, config: CacheConfig):
        self.config = config
        self.block_size = config.block_size
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.num_query_heads = config.num_heads
        self.head_dim = config.head_dim

        self.block_manager = BlockManager(config)

        self.key_caches = []
        self.value_caches = []

        for _ in range(self.num_layers):
            key_cache = torch.zeros(
                config.num_blocks,
                config.block_size,
                config.num_kv_heads,
                config.head_dim,
                dtype=config.dtype,
                device=config.device,
            )

            value_cache = torch.zeros(
                config.num_blocks,
                config.block_size,
                config.num_kv_heads,
                config.head_dim,
                dtype=config.dtype,
                device=config.device,
            )

            self.key_caches.append(key_cache)
            self.value_caches.append(value_cache)

    def allocate_sequence(self, seq_id: int, num_tokens: int):
        self.block_manager.allocate(seq_id, num_tokens)

    def free_sequence(self, seq_id: int):
        self.block_manager.free_sequence(seq_id)

    def append_slots(self, seq_id: int, num_tokens: int):
        self.block_manager.append_slots(seq_id, num_tokens)

    def set_kv_cache(self,
                     layer_idx: int,
                     seq_id: int,
                     key: torch.Tensor,
                     value: torch.Tensor,
                     start_pos: int = 0):
        num_tokens = key.shape[0]
        key_cache = self.key_caches[layer_idx]
        value_cache = self.value_caches[layer_idx]
        block_table = self.block_manager.get_block_table(seq_id)

        for i in range(num_tokens):
            token_pos = start_pos + i
            logical_block_idx = token_pos // self.block_size
            physical_block_idx = block_table[logical_block_idx]
            block_offset = token_pos % self.block_size
            key_cache[physical_block_idx][block_offset] = key[i]
            value_cache[physical_block_idx][block_offset] = value[i]

    def compute_attention(self,
                          layer_idx: int,
                          query: torch.Tensor,
                          block_tables: torch.Tensor,
                          context_lens: torch.Tensor,
                          ) -> torch.Tensor:
        keys = self.key_caches[layer_idx]
        values = self.value_caches[layer_idx]
        return paged_attention_torch(
            query,
            keys,
            values,
            block_tables,
            context_lens,
            self.block_size,
            self.num_kv_heads,
            self.num_query_heads,
            self.head_dim
        )

    def get_block_table(self, seq_id: int) -> List[int]:
        return self.block_manager.get_block_table(seq_id)

    def get_num_free_blocks(self) -> int:
        return self.block_manager.get_num_free_blocks()


if __name__ == "__main__":
    from litecache.config import CacheConfig

    # Create config
    config = CacheConfig(
        block_size=16,
        num_blocks=100,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        num_layers=2,
        dtype=torch.float32,
        device='cpu'
    )

    # Create cache
    cache = PagedAttentionCache(config)
    print(f"Created cache: {len(cache.key_caches)} layers")

    # Allocate sequence
    cache.allocate_sequence(seq_id=1, num_tokens=50)
    print(f"Allocated sequence 1 with 50 tokens")
    print(f"Block table: {cache.get_block_table(1)}")

    # Create dummy KV
    key = torch.randn(50, 8, 128)
    value = torch.randn(50, 8, 128)

    # Write to cache
    cache.set_kv_cache(layer_idx=0, seq_id=1, key=key, value=value)
    print("Written KV to cache")

    # Prepare for attention
    query = torch.randn(1, 32, 128)
    block_table = torch.tensor([cache.get_block_table(1)])
    context_lens = torch.tensor([50])

    # Compute attention
    output = cache.compute_attention(
        layer_idx=0,
        query=query,
        block_tables=block_table,
        context_lens=context_lens
    )

    print(f"✓ Output shape: {output.shape}")  # Should be [1, 32, 128]
    print(f"✓ Free blocks: {cache.get_num_free_blocks()}")
