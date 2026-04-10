from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List
import random


# ------------------------------------------------------------
# HillInfer backbone demo
# ------------------------------------------------------------
# Simplified architecture / algorithm implemented here:
#   1) importance-aware KV block scoring
#   2) hierarchical placement across GPU / CPU / SmartSSD
#   3) adaptive prefetch based on predicted future access
#
# This is a simplified implementation of the core HillInfer idea:
# SmartSSD-assisted hierarchical KV management for long-context edge inference.
# ------------------------------------------------------------


class Tier(str, Enum):
    GPU = "GPU"
    CPU = "CPU"
    SMARTSSD = "SMARTSSD"


@dataclass
class KVBlock:
    block_id: int
    importance: float
    predicted_next_use: int
    size_mb: float
    tier: Tier | None = None


@dataclass
class TierStore:
    name: Tier
    capacity_mb: float
    used_mb: float = 0.0
    blocks: Dict[int, KVBlock] = field(default_factory=dict)

    def can_fit(self, block: KVBlock) -> bool:
        return self.used_mb + block.size_mb <= self.capacity_mb

    def add(self, block: KVBlock) -> None:
        self.blocks[block.block_id] = block
        self.used_mb += block.size_mb
        block.tier = self.name

    def remove(self, block_id: int) -> KVBlock:
        block = self.blocks.pop(block_id)
        self.used_mb -= block.size_mb
        return block


class HillInferBackbone:
    def __init__(self):
        self.gpu = TierStore(Tier.GPU, capacity_mb=128)
        self.cpu = TierStore(Tier.CPU, capacity_mb=256)
        self.smartssd = TierStore(Tier.SMARTSSD, capacity_mb=2048)

    def score_importance_in_storage(self, block: KVBlock) -> float:
        """
        Simplified SmartSSD-side lightweight importance evaluation.
        In the real paper, this is the key architectural move:
        do lightweight importance work near storage to reduce host-side data movement.
        """
        return block.importance

    def place_block(self, block: KVBlock) -> None:
        score = self.score_importance_in_storage(block)

        if score > 0.75:
            self._place_with_fallback(block, preferred=Tier.GPU)
        elif score > 0.35:
            self._place_with_fallback(block, preferred=Tier.CPU)
        else:
            self._place_with_fallback(block, preferred=Tier.SMARTSSD)

    def _place_with_fallback(self, block: KVBlock, preferred: Tier) -> None:
        stores = {
            Tier.GPU: self.gpu,
            Tier.CPU: self.cpu,
            Tier.SMARTSSD: self.smartssd,
        }

        order = {
            Tier.GPU: [self.gpu, self.cpu, self.smartssd],
            Tier.CPU: [self.cpu, self.smartssd],
            Tier.SMARTSSD: [self.smartssd],
        }[preferred]

        for store in order:
            if store.can_fit(block):
                store.add(block)
                return

        # simplified eviction / demotion policy
        if preferred == Tier.GPU and len(self.gpu.blocks) > 0:
            victim_id = min(self.gpu.blocks,
                            key=lambda k: self.gpu.blocks[k].importance)
            victim = self.gpu.remove(victim_id)
            self._place_with_fallback(victim, preferred=Tier.CPU)
            self._place_with_fallback(block, preferred=Tier.GPU)
            return

        if preferred in (Tier.GPU, Tier.CPU) and len(self.cpu.blocks) > 0:
            victim_id = min(self.cpu.blocks,
                            key=lambda k: self.cpu.blocks[k].importance)
            victim = self.cpu.remove(victim_id)
            self._place_with_fallback(victim, preferred=Tier.SMARTSSD)
            self._place_with_fallback(block, preferred=preferred)
            return

        raise RuntimeError("No space available even after fallback/eviction.")

    def adaptive_prefetch(self, future_blocks: List[KVBlock]) -> List[str]:
        """
        Simplified adaptive prefetch backbone:
        pull future-needed blocks closer to compute before they are touched.
        """
        ops = []
        for block in sorted(future_blocks, key=lambda b: b.predicted_next_use):
            if block.tier == Tier.SMARTSSD:
                ops.append(f"prefetch block {block.block_id}: SMARTSSD -> CPU")
                block.tier = Tier.CPU
            elif block.tier == Tier.CPU and block.importance > 0.75:
                ops.append(f"prefetch block {block.block_id}: CPU -> GPU")
                block.tier = Tier.GPU
        return ops

    def report(self) -> None:
        for store in [self.gpu, self.cpu, self.smartssd]:
            print(
                f"{store.name:8s} "
                f"used={store.used_mb:6.1f}/{store.capacity_mb:6.1f} MB "
                f"blocks={[(b.block_id, round(b.importance, 2)) for b in store.blocks.values()]}"
            )


def make_blocks(n: int = 12) -> List[KVBlock]:
    random.seed(42)
    blocks = []
    for i in range(n):
        blocks.append(
            KVBlock(
                block_id=i,
                importance=random.random(),
                predicted_next_use=random.randint(1, 20),
                size_mb=random.choice([8, 12, 16, 24]),
            )
        )
    return blocks


def main() -> None:
    system = HillInferBackbone()
    blocks = make_blocks()

    print("=== HillInfer backbone demo ===\n")
    print("Initial hierarchical placement:\n")

    for block in blocks:
        system.place_block(block)

    system.report()

    print("\nAdaptive prefetch plan:\n")
    future = sorted(blocks, key=lambda b: b.predicted_next_use)[:6]
    ops = system.adaptive_prefetch(future)
    for op in ops:
        print(op)

    print("\nAfter prefetch decisions:\n")
    # report logical tier view after prefetch planning
    gpu_like = [b.block_id for b in blocks if b.tier == Tier.GPU]
    cpu_like = [b.block_id for b in blocks if b.tier == Tier.CPU]
    ssd_like = [b.block_id for b in blocks if b.tier == Tier.SMARTSSD]

    print(f"GPU-resident blocks      : {gpu_like}")
    print(f"CPU-resident blocks      : {cpu_like}")
    print(f"SmartSSD-resident blocks : {ssd_like}")

    print("\nInterpretation:")
    print("- Important blocks are kept closer to the GPU.")
    print("- Colder blocks are demoted into SmartSSD-backed storage.")
    print("- Lightweight importance evaluation conceptually happens near storage.")
    print("- Adaptive prefetch moves likely-needed blocks up the hierarchy before decode touches them.")
    print("- This is the core backbone of HillInfer's hierarchical KV management idea.")


if __name__ == "__main__":
    main()
