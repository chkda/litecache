class BlockAllocator:
    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.num_blocks = num_blocks
        self.num_blocks = num_blocks
        self.__free_blocks = {i for i in range(num_blocks)}
        self.__allocated_blocks = set()

    def allocate(self) -> int:
        if len(self.__free_blocks) == 0:
            raise ValueError("No free blocks")
        block_id = self.__free_blocks.pop()
        self.__allocated_blocks.add(block_id)
        return block_id

    def free(self, block_id: int):
        if not (0 <= block_id < self.num_blocks):
            raise ValueError(f"Block id {block_id} is out of range")

        if block_id not in self.__allocated_blocks:
            raise ValueError(f"Block id {block_id} is not allocated")

        self.__allocated_blocks.remove(block_id)
        self.__free_blocks.add(block_id)

    def get_num_free_blocks(self) -> int:
        return len(self.__free_blocks)

    def get_num_used_blocks(self) -> int:
        return len(self.__allocated_blocks)

    def can_allocate(self, num_blocks: int) -> bool:
        return num_blocks <= self.get_num_free_blocks()

    def reset(self):
        self.__free_blocks = {i for i in range(self.num_blocks)}
        self.__allocated_blocks = set()
