from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random


class Tier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class KVBlock:
    block_id: str
    active: bool
    reuse_score: float
    size_mb: float
    tier: Tier | None = None
    codec: str | None = None


def update_tier(active: bool, reuse_score: float) -> Tier:
    if active:
        return Tier.HOT
    if reuse_score > 0.7:
        return Tier.WARM
    return Tier.COLD


def choose_codec(tier: Tier) -> str:
    if tier == Tier.HOT:
        return "low_overhead_codec"
    if tier == Tier.WARM:
        return "balanced_codec"
    return "high_compression_codec"


def estimated_compressed_size_mb(block: KVBlock) -> float:
    ratio = {
        "low_overhead_codec": 0.5,
        "balanced_codec": 0.3,
        "high_compression_codec": 0.15,
    }[block.codec]
    return block.size_mb * ratio


def build_demo_blocks(n: int = 10) -> list[KVBlock]:
    random.seed(42)
    blocks = []
    for i in range(n):
        active = random.random() < 0.3
        reuse_score = random.random()
        size_mb = round(random.uniform(32, 256), 1)
        block = KVBlock(
            block_id=f"block_{i}",
            active=active,
            reuse_score=reuse_score,
            size_mb=size_mb,
        )
        block.tier = update_tier(block.active, block.reuse_score)
        block.codec = choose_codec(block.tier)
        blocks.append(block)
    return blocks


def main() -> None:
    blocks = build_demo_blocks()

    print("=== KV lifecycle-aware codec demo ===\n")
    total_raw = 0.0
    total_compressed = 0.0

    for b in blocks:
        compressed = estimated_compressed_size_mb(b)
        total_raw += b.size_mb
        total_compressed += compressed

        print(
            f"{b.block_id:8s} "
            f"active={str(b.active):5s} "
            f"reuse={b.reuse_score:.2f} "
            f"tier={b.tier.value:4s} "
            f"codec={b.codec:22s} "
            f"raw={b.size_mb:6.1f}MB "
            f"compressed={compressed:6.1f}MB"
        )

    print("\nSummary:")
    print(f"raw total        : {total_raw:.1f} MB")
    print(f"compressed total : {total_compressed:.1f} MB")
    print(f"effective ratio  : {total_compressed / total_raw:.3f}")

    print("\nInterpretation:")
    print("- Hot blocks get lower-overhead compression.")
    print("- Warm blocks get balanced compression.")
    print("- Cold blocks get stronger storage-oriented compression.")
    print("- Real systems would use richer policies, but the lifecycle idea is the same.")


if __name__ == "__main__":
    main()
