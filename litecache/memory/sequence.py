from dataclasses import dataclass
from typing import List, Dict

from litecache.memory.allocator import BlockAllocator


@dataclass
class Sequence:
    seq_id: int
    num_tokens: int
    block_table: List[int]


class SequenceManager:

    def __init__(self, block_size: int, allocator: BlockAllocator):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.block_size = block_size
        self.allocator = allocator
        self.__sequence: Dict[int, Sequence] = {}

    def allocate_sequence(self, seq_id: int, num_tokens: int):
        if self.sequence_exists(seq_id):
            raise ValueError("Sequence already allocated")

        if num_tokens < 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")

        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if not self.allocator.can_allocate(num_blocks):
            raise RuntimeError("Cannot allocate sequence. Not enough free blocks")

        block_table = []
        for _ in range(num_blocks):
            block_id = self.allocator.allocate()
            block_table.append(block_id)
        seq = Sequence(seq_id, num_tokens, block_table)
        self.__sequence[seq_id] = seq

    def append_tokens(self, seq_id: int, num_new_tokens: int):
        if not self.sequence_exists(seq_id):
            raise ValueError("Sequence doesn't exist")

        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be positive, got {num_new_tokens}")

        seq = self.__sequence[seq_id]
        # Check if we need a new block
        old_num_blocks = len(seq.block_table)
        seq.num_tokens += num_new_tokens
        num_blocks = (seq.num_tokens + self.block_size - 1) // self.block_size

        num_new_blocks = num_blocks - old_num_blocks
        if num_new_blocks == 0:
            return

        if not self.allocator.can_allocate(num_new_blocks):
            raise RuntimeError("Cannot allocate sequence. Not enough free blocks")

        for _ in range(num_new_blocks):
            block_id = self.allocator.allocate()
            seq.block_table.append(block_id)

    def free_sequence(self, seq_id: int):
        if not self.sequence_exists(seq_id):
            raise ValueError("Sequence doesn't exist")

        seq = self.__sequence.pop(seq_id)
        for block_id in seq.block_table:
            self.allocator.free(block_id)

    def get_block_table(self, seq_id: int) -> List[int]:
        if not self.sequence_exists(seq_id):
            raise ValueError("Sequence doesn't exist")

        return self.__sequence[seq_id].block_table

    def get_num_tokens(self, seq_id: int) -> int:
        if not self.sequence_exists(seq_id):
            raise ValueError("Sequence doesn't exist")

        return self.__sequence[seq_id].num_tokens

    def sequence_exists(self, seq_id: int) -> bool:
        return seq_id in self.__sequence

    def get_num_sequences(self) -> int:
        return len(self.__sequence)

    def __repr__(self) -> str:
        return (
            f"SequenceManager(block_size={self.block_size}, "
            f"sequences={self.get_num_sequences()}, "
            f"total_blocks_used={self.allocator.get_num_used_blocks()})"
        )
