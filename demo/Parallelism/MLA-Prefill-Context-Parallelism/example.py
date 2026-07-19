"""Deterministic demonstration of PCP load balancing and DCP KV ownership."""

from pcp_mini import (
    contiguous_partition,
    dual_chunk_swap_partition,
    kv_tokens_per_rank,
    naive_coupled_kv_tokens_per_rank,
    rank_summary,
)


def print_partition(name, result):
    print(f"\n{name}")
    for row in rank_summary(result):
        print(
            f"  rank {row['rank']}: work={row['work']:>12,}  "
            + ", ".join(row["chunks"])
        )
    print(f"  imbalance: {result.imbalance:.3f}")


def main() -> None:
    requests = [("long", 100_000), ("medium", 32_768), ("short", 8_192)]
    pcp_size = 4

    contiguous = contiguous_partition(requests, pcp_size)
    balanced = dual_chunk_swap_partition(requests, pcp_size)

    print("Prefill Context Parallelism miniature")
    print(f"requests={requests}, PCP={pcp_size}")
    print_partition("Contiguous baseline", contiguous)
    print_partition("DualChunkSwap", balanced)

    total_kv_tokens = sum(length for _, length in requests)
    dcp_size = 2
    orthogonal = kv_tokens_per_rank(total_kv_tokens, dcp_size)
    coupled = naive_coupled_kv_tokens_per_rank(total_kv_tokens, pcp_size, dcp_size)

    print("\nKV ownership")
    print(f"  total scheduled tokens: {total_kv_tokens:,}")
    print(f"  correct DCP-only ownership (DCP={dcp_size}): {orthogonal:,} tokens/rank")
    print(
        "  incorrect PCP×DCP assumption "
        f"(PCP={pcp_size}, DCP={dcp_size}): {coupled:,} tokens/rank"
    )
    print(
        "  lesson: PCP reduces per-rank prefill compute, but does not by itself "
        "create additional KV shards."
    )


if __name__ == "__main__":
    main()
