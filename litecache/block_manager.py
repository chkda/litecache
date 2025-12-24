from typing import List

from litecache.config import CacheConfig
from litecache.memory.allocator import BlockAllocator
from litecache.memory.sequence import SequenceManager


class BlockManager:

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.block_size = config.block_size
        self.allocator = BlockAllocator(num_blocks=config.num_blocks)
        self.sequence_manager = SequenceManager(
            block_size=config.block_size,
            allocator=self.allocator,
        )

    def allocate(self, seq_id: int, num_tokens: int):
        self.sequence_manager.allocate_sequence(seq_id, num_tokens)

    def append_slots(self, seq_id: int, num_tokens: int):
        self.sequence_manager.append_tokens(seq_id, num_tokens)

    def free_sequence(self, seq_id: int):
        self.sequence_manager.free_sequence(seq_id)

    def get_block_table(self, seq_id: int) -> List[int]:
        return self.sequence_manager.get_block_table(seq_id)

    def can_allocate_sequence(self, num_tokens: int) -> bool:
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return self.allocator.can_allocate(num_blocks)

    def can_append_slots(self, seq_id: int, num_tokens: int) -> bool:
        if not self.sequence_manager.sequence_exists(seq_id):
            return False
        
        curr_tokens = self.sequence_manager.get_num_tokens(seq_id)
        curr_blocks = len(self.get_block_table(seq_id))
        total_tokens = num_tokens + curr_tokens
        num_blocks = (total_tokens + self.block_size - 1) // self.block_size
        return self.allocator.can_allocate(num_blocks - curr_blocks)

    def get_num_free_blocks(self) -> int:
        return self.allocator.get_num_free_blocks()

    def get_num_total_blocks(self) -> int:
        return self.allocator.num_blocks

    def get_num_sequences(self) -> int:
        return self.sequence_manager.get_num_sequences()

    def get_block_utilization(self) -> float:
        return self.allocator.get_num_used_blocks() / self.allocator.num_blocks
