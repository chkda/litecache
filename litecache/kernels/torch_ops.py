from typing import Optional

import torch


def paged_attention_torch(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_size: int,
        num_kv_heads: int,
        num_query_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
) -> torch.Tensor:
    """
        Compute paged attention using PyTorch.

        Reference implementation - correct but not optimized.

        Args:
            query: [num_seqs, num_query_heads, head_dim]
            key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            block_tables: [num_seqs, max_num_blocks]
            context_lens: [num_seqs]
            block_size: Tokens per block
            num_kv_heads: Number of KV heads
            num_query_heads: Number of query heads
            head_dim: Dimension of each head
            scale: Attention scale (default: 1/sqrt(head_dim))

        Returns:
            Attention output [num_seqs, num_query_heads, head_dim]
        """
    if scale is None:
        scale = 1 / (head_dim ** 0.5)

    batch_size = query.shape[0]

    # GQA
    num_queries_per_kv = num_query_heads // num_kv_heads

    outputs = []

    for seq_idx in range(batch_size):
        seq_query = query[seq_idx]
        seq_block_table = block_tables[seq_idx]
        seq_context_len = context_lens[seq_idx].item()

        if seq_context_len == 0:
            output = torch.zeros(num_query_heads, head_dim, dtype=query.dtype, device=query.device)
            outputs.append(output)
            continue

        num_blocks = (seq_context_len + block_size - 1) // block_size

        keys_list = []
        values_list = []

        for block_idx in range(num_blocks):
            physical_block_id = seq_block_table[block_idx].item()
            keys_list.append(key_cache[physical_block_id])
            values_list.append(value_cache[physical_block_id])

        keys = torch.cat(keys_list, dim=0)
        values = torch.cat(values_list, dim=0)

        keys = keys[:seq_context_len]
        values = values[:seq_context_len]

        if num_query_heads != num_kv_heads:
            keys = keys.repeat_interleave(num_queries_per_kv, dim=1)
            values = values.repeat_interleave(num_queries_per_kv, dim=1)

        keys_t = keys.permute(1, 2, 0)

        attn_scores = torch.bmm(
            seq_query.unsqueeze(1),
            keys_t
        ).squeeze(1)

        attn_scores = attn_scores * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        values_t = values.permute(1, 0, 2)

        output = torch.bmm(
            attn_weights.unsqueeze(1),
            values_t
        ).squeeze(1)
        outputs.append(output)

    return torch.stack(outputs, dim=0)


if __name__ == "__main__":
    # Test the implementation
    print("=" * 70)
    print("Testing paged_attention_pytorch")
    print("=" * 70)

    # Test 1: Single sequence, MHA (num_query_heads == num_kv_heads)
    print("\nTest 1: Single sequence, MHA")
    block_size = 16
    num_blocks = 10
    num_kv_heads = 8
    num_query_heads = 8
    head_dim = 64

    query = torch.randn(1, num_query_heads, head_dim)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
    block_tables = torch.tensor([[0, 1, 2]])  # Use 3 blocks
    context_lens = torch.tensor([45])  # 45 tokens (3 blocks, last partially filled)

    output = paged_attention_torch(
        query, key_cache, value_cache, block_tables, context_lens,
        block_size=block_size, num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads, head_dim=head_dim
    )

    print(f"Query shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, num_query_heads, head_dim), "Wrong output shape!"
    print("✓ Test 1 passed")

    # Test 2: Batch of sequences, GQA
    print("\nTest 2: Batch of sequences, GQA")
    num_query_heads = 32
    num_kv_heads = 8  # 4 query heads share 1 KV head

    query = torch.randn(3, num_query_heads, head_dim)  # 3 sequences
    block_tables = torch.tensor([
        [0, 1, 2, 3],  # Seq 0: 4 blocks
        [4, 5, 0, 0],  # Seq 1: 2 blocks (rest unused)
        [6, 7, 8, 0],  # Seq 2: 3 blocks
    ])
    context_lens = torch.tensor([50, 20, 35])

    output = paged_attention_torch(
        query, key_cache, value_cache, block_tables, context_lens,
        block_size=block_size, num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads, head_dim=head_dim
    )

    print(f"Query shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (3, num_query_heads, head_dim), "Wrong output shape!"
    print("✓ Test 2 passed")

    # Test 3: Edge case - single token
    print("\nTest 3: Edge case - single token")
    context_lens = torch.tensor([1])
    block_tables = torch.tensor([[0, 0, 0]])
    query = torch.randn(1, num_query_heads, head_dim)

    output = paged_attention_torch(
        query, key_cache, value_cache, block_tables, context_lens,
        block_size=block_size, num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads, head_dim=head_dim
    )

    print(f"Output shape: {output.shape}")
    assert output.shape == (1, num_query_heads, head_dim), "Wrong output shape!"
    print("✓ Test 3 passed")

    # Test 4: Verify attention output is valid (no NaN, reasonable range)
    print("\nTest 4: Sanity checks")
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"
    print(f"Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
    print("✓ Test 4 passed")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
