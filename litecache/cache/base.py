from abc import ABC, abstractmethod
from typing import List

import torch


class CacheBase(ABC):

    @abstractmethod
    def allocate_sequence(self, seq_id: int, num_tokens: int):
        pass

    @abstractmethod
    def free_sequence(self, seq_id: int):
        pass

    @abstractmethod
    def append_slots(self, seq_id: int, num_tokens: int):
        pass

    @abstractmethod
    def set_kv_cache(self,
                     layer_idx: int,
                     seq_id: int,
                     key: torch.Tensor,
                     value: torch.Tensor,
                     start_pos: int = 0):
        pass

    @abstractmethod
    def compute_attention(self,
                          layer_idx: int,
                          query: torch.Tensor,
                          block_tables: torch.Tensor,
                          context_lens: torch.Tensor,
                          ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_block_table(self, seq_id: int) -> List[int]:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass
