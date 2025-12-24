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


if __name__ == "__main__":
    print("=" * 60)
    print("Testing BlockAllocator")
    print("=" * 60)

    # Test 1: Basic allocation and free
    print("\n1. Basic allocation and free")
    allocator = BlockAllocator(num_blocks=4)
    print(f"Initial state: {allocator.get_num_free_blocks()} free, {allocator.get_num_used_blocks()} used")

    b1 = allocator.allocate()
    b2 = allocator.allocate()
    print(f"After 2 allocations: {allocator.get_num_free_blocks()} free, {allocator.get_num_used_blocks()} used")

    allocator.free(b1)
    print(f"After freeing one: {allocator.get_num_free_blocks()} free, {allocator.get_num_used_blocks()} used")

    # Test 2: Allocate all blocks
    print("\n2. Allocate all blocks")
    allocator.reset()
    blocks = [allocator.allocate() for _ in range(4)]
    print(f"Allocated all blocks: {blocks}")
    print(f"State: {allocator.get_num_free_blocks()} free, {allocator.get_num_used_blocks()} used")

    # Test 3: Try to allocate when full
    print("\n3. Try to allocate when full (should fail)")
    try:
        allocator.allocate()
        print("❌ ERROR: Should have raised exception!")
    except ValueError as e:
        print(f"✅ Correctly raised: {e}")

    # Test 4: Double-free detection
    print("\n4. Double-free detection")
    allocator.reset()
    b = allocator.allocate()
    allocator.free(b)
    try:
        allocator.free(b)
        print("❌ ERROR: Double-free not detected!")
    except ValueError as e:
        print(f"✅ Correctly caught double-free: {e}")

    # Test 5: Free never-allocated block
    print("\n5. Free never-allocated block")
    try:
        allocator.free(2)  # Block 2 is free, never allocated
        print("❌ ERROR: Should have raised exception!")
    except ValueError as e:
        print(f"✅ Correctly raised: {e}")

    # Test 6: Invalid block ID
    print("\n6. Invalid block ID")
    try:
        allocator.free(999)
        print("❌ ERROR: Should have raised exception!")
    except ValueError as e:
        print(f"✅ Correctly raised: {e}")

    # Test 7: can_allocate()
    print("\n7. can_allocate() method")
    allocator.reset()
    print(f"Can allocate 2 blocks? {allocator.can_allocate(2)}")  # True
    print(f"Can allocate 5 blocks? {allocator.can_allocate(5)}")  # False
    allocator.allocate()
    allocator.allocate()
    print(f"After 2 allocations, can allocate 3? {allocator.can_allocate(3)}")  # False
    print(f"Can allocate 2? {allocator.can_allocate(2)}")  # True

    # Test 8: Block reuse
    print("\n8. Block reuse after free")
    allocator.reset()
    b1 = allocator.allocate()
    print(f"First allocation: block {b1}")
    allocator.free(b1)
    b2 = allocator.allocate()
    print(f"After free and re-allocate: block {b2}")
    print(f"✅ Block was reused: {b1 == b2}")

    # Test 9: Reset functionality
    print("\n9. Reset functionality")
    allocator.allocate()
    allocator.allocate()
    print(f"Before reset: {allocator.get_num_free_blocks()} free")
    allocator.reset()
    print(f"After reset: {allocator.get_num_free_blocks()} free")

    # Test 10: Invalid initialization
    print("\n10. Invalid initialization")
    try:
        bad_allocator = BlockAllocator(num_blocks=0)
        print("❌ ERROR: Should have raised exception!")
    except ValueError as e:
        print(f"✅ Correctly raised: {e}")

    try:
        bad_allocator = BlockAllocator(num_blocks=-5)
        print("❌ ERROR: Should have raised exception!")
    except ValueError as e:
        print(f"✅ Correctly raised: {e}")

    print("\n" + "=" * 60)
    print("✅ All tests passed! BlockAllocator is working correctly.")
    print("=" * 60)
