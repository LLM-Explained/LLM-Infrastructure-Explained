from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import hashlib
import random


class Tier(str, Enum):
    GPU = "GPU_HBM"
    CPU = "CPU_RAM"
    SSD = "LOCAL_SSD"


@dataclass
class KVBlock:
    session_id: str
    prefix_key: str
    tokens: int
    active: bool = False
    last_touch: int = 0
    reuse_score: float = 0.0
    tier: Tier = Tier.SSD

    @property
    def size_units(self) -> int:
        return max(1, self.tokens // 1024)


@dataclass
class TierStore:
    name: Tier
    capacity: int
    used: int = 0
    blocks: Dict[str, KVBlock] = field(default_factory=dict)

    def can_fit(self, block: KVBlock) -> bool:
        return self.used + block.size_units <= self.capacity

    def add(self, block: KVBlock) -> None:
        if not self.can_fit(block):
            raise RuntimeError(f"{self.name} is full")
        self.blocks[block.session_id] = block
        self.used += block.size_units
        block.tier = self.name

    def remove(self, session_id: str) -> KVBlock:
        block = self.blocks.pop(session_id)
        self.used -= block.size_units
        return block


class MultiTierKVCache:
    def __init__(self, gpu_cap: int, cpu_cap: int, ssd_cap: int):
        self.time = 0
        self.gpu = TierStore(Tier.GPU, gpu_cap)
        self.cpu = TierStore(Tier.CPU, cpu_cap)
        self.ssd = TierStore(Tier.SSD, ssd_cap)
        self.directory: Dict[str, Tier] = {}
        self.prefix_index: Dict[str, List[str]] = {}

    def _store_for(self, tier: Tier) -> TierStore:
        return {
            Tier.GPU: self.gpu,
            Tier.CPU: self.cpu,
            Tier.SSD: self.ssd,
        }[tier]

    def _touch(self, block: KVBlock, reuse_boost: float = 0.05) -> None:
        self.time += 1
        block.last_touch = self.time
        block.reuse_score = min(1.0, block.reuse_score + reuse_boost)

    def _pick_victim(self, store: TierStore) -> Optional[KVBlock]:
        candidates = [b for b in store.blocks.values() if not b.active]
        if not candidates:
            return None
        candidates.sort(key=lambda b: (b.reuse_score, b.last_touch))
        return candidates[0]

    def _place(self, block: KVBlock, tier: Tier) -> None:
        store = self._store_for(tier)
        while not store.can_fit(block):
            victim = self._pick_victim(store)
            if victim is None:
                raise RuntimeError(f"No evictable blocks in {tier}")
            self._demote(victim.session_id)
        store.add(block)
        self.directory[block.session_id] = tier

    def _demote(self, session_id: str) -> None:
        current_tier = self.directory[session_id]
        current_store = self._store_for(current_tier)
        block = current_store.remove(session_id)

        next_tier = {
            Tier.GPU: Tier.CPU,
            Tier.CPU: Tier.SSD,
            Tier.SSD: Tier.SSD,
        }[current_tier]

        self._place(block, next_tier)

    def _promote(self, session_id: str, target: Tier) -> None:
        current_tier = self.directory[session_id]
        if current_tier == target:
            return
        current_store = self._store_for(current_tier)
        block = current_store.remove(session_id)
        self._place(block, target)

    def add_session(self, session_id: str, prefix: str, tokens: int, active: bool) -> None:
        prefix_key = self.prefix_hash(prefix)
        block = KVBlock(
            session_id=session_id,
            prefix_key=prefix_key,
            tokens=tokens,
            active=active,
            reuse_score=0.25 if active else 0.05,
        )
        self._touch(block)
        initial_tier = Tier.GPU if active else Tier.CPU
        self._place(block, initial_tier)
        self.prefix_index.setdefault(prefix_key, []).append(session_id)

    def access_session(self, session_id: str, active: bool = True) -> None:
        tier = self.directory[session_id]
        block = self._store_for(tier).blocks[session_id]
        block.active = active
        self._touch(block, reuse_boost=0.15 if active else 0.02)

        if active and block.tier != Tier.GPU:
            self._promote(session_id, Tier.GPU)
        elif not active and block.tier == Tier.GPU and block.reuse_score < 0.4:
            self._demote(session_id)

    def finish_decode_step(self, session_id: str) -> None:
        tier = self.directory[session_id]
        block = self._store_for(tier).blocks[session_id]
        block.active = False
        self._touch(block, reuse_boost=0.01)

    def route_by_prefix(self, prefix: str) -> Optional[str]:
        prefix_key = self.prefix_hash(prefix)
        session_ids = self.prefix_index.get(prefix_key, [])
        if not session_ids:
            return None

        def score(sid: str):
            tier = self.directory[sid]
            tier_score = {Tier.GPU: 3, Tier.CPU: 2, Tier.SSD: 1}[tier]
            block = self._store_for(tier).blocks[sid]
            return (tier_score, block.reuse_score, block.last_touch)

        return max(session_ids, key=score)

    @staticmethod
    def prefix_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def stats(self) -> str:
        return (
            f"GPU {self.gpu.used}/{self.gpu.capacity} blocks={list(self.gpu.blocks)}\n"
            f"CPU {self.cpu.used}/{self.cpu.capacity} blocks={list(self.cpu.blocks)}\n"
            f"SSD {self.ssd.used}/{self.ssd.capacity} blocks={list(self.ssd.blocks)}"
        )


def main() -> None:
    random.seed(42)
    cache = MultiTierKVCache(gpu_cap=12, cpu_cap=20, ssd_cap=100)

    shared_prefix = "You are a coding assistant.\nRepository summary...\n"

    for i in range(8):
        cache.add_session(
            session_id=f"sess_{i}",
            prefix=shared_prefix if i < 5 else f"user-specific-{i}",
            tokens=random.randint(3000, 12000),
            active=(i % 3 == 0),
        )

    print("=== Initial placement ===")
    print(cache.stats(), end="\n\n")

    chosen = cache.route_by_prefix(shared_prefix)
    print(f"Prefix-aware routing chose: {chosen}")

    if chosen:
        cache.access_session(chosen, active=True)

    for sid in ["sess_1", "sess_2", "sess_3", "sess_4"]:
        cache.access_session(sid, active=True)
        cache.finish_decode_step(sid)

    print("\n=== After accesses ===")
    print(cache.stats(), end="\n\n")

    cache.add_session(
        session_id="sess_big",
        prefix=shared_prefix,
        tokens=18000,
        active=True,
    )

    print("=== After adding a large active session ===")
    print(cache.stats(), end="\n\n")

    chosen2 = cache.route_by_prefix(shared_prefix)
    print(f"Best session for shared prefix now: {chosen2}")


if __name__ == "__main__":
    main()
