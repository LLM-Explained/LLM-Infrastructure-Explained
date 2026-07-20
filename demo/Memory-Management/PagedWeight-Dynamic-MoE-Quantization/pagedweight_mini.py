"""Educational miniature of PagedWeight's quality-aware runtime planner.

The real system stores MoE expert linear blocks in an Any-Precision bit-plane
representation and moves pages asynchronously between CPU and GPU. This module
models only the planning layer: choose precision reductions that free enough
memory while minimizing predicted quality damage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp
from typing import Iterable, Sequence


SUPPORTED_BITS: tuple[int, ...] = (16, 8, 4, 2)


@dataclass(frozen=True)
class ExpertBlock:
    """One routed-expert linear block tracked by the miniature page table."""

    name: str
    parameter_count: int
    bitwidth: int
    bitwidth_floor: int
    global_sensitivity: float
    routing_mass: float
    prompt_log_residual: float = 0.0

    def __post_init__(self) -> None:
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        if self.bitwidth not in SUPPORTED_BITS:
            raise ValueError(f"unsupported bitwidth: {self.bitwidth}")
        if self.bitwidth_floor not in SUPPORTED_BITS:
            raise ValueError(f"unsupported bitwidth floor: {self.bitwidth_floor}")
        if self.bitwidth < self.bitwidth_floor:
            raise ValueError("bitwidth cannot be below bitwidth_floor")
        if self.global_sensitivity < 0 or self.routing_mass < 0:
            raise ValueError("sensitivity and routing mass must be non-negative")

    def memory_bytes(self, bitwidth: int | None = None) -> int:
        """Idealized packed storage, excluding metadata and alignment."""

        bits = self.bitwidth if bitwidth is None else bitwidth
        return (self.parameter_count * bits + 7) // 8


@dataclass(frozen=True)
class QuantizationAction:
    block_name: str
    from_bits: int
    to_bits: int
    released_bytes: int
    predicted_damage: float

    @property
    def damage_per_released_byte(self) -> float:
        return self.predicted_damage / self.released_bytes


@dataclass(frozen=True)
class PlanResult:
    blocks: tuple[ExpertBlock, ...]
    actions: tuple[QuantizationAction, ...]
    requested_bytes: int
    released_bytes: int
    predicted_damage: float

    @property
    def target_met(self) -> bool:
        return self.released_bytes >= self.requested_bytes


def next_lower_bitwidth(bitwidth: int, floor: int) -> int | None:
    """Return the next supported lower precision that respects the floor."""

    current_index = SUPPORTED_BITS.index(bitwidth)
    for candidate in SUPPORTED_BITS[current_index + 1 :]:
        if candidate >= floor:
            return candidate
    return None


def next_higher_bitwidth(bitwidth: int) -> int | None:
    """Return the next supported higher precision."""

    current_index = SUPPORTED_BITS.index(bitwidth)
    if current_index == 0:
        return None
    return SUPPORTED_BITS[current_index - 1]


def routing_multiplier(routing_mass: float, strength: float = 3.0) -> float:
    """Protect frequently routed experts with a smooth multiplicative penalty."""

    return 1.0 + strength * routing_mass


def estimate_damage(block: ExpertBlock, from_bits: int, to_bits: int) -> float:
    """Synthetic analogue of global sensitivity × routing × prompt residual."""

    if to_bits >= from_bits:
        raise ValueError("to_bits must be lower than from_bits")
    precision_drop = (from_bits - to_bits) / from_bits
    return (
        block.global_sensitivity
        * precision_drop
        * routing_multiplier(block.routing_mass)
        * exp(block.prompt_log_residual)
    )


def candidate_action(block: ExpertBlock) -> QuantizationAction | None:
    to_bits = next_lower_bitwidth(block.bitwidth, block.bitwidth_floor)
    if to_bits is None:
        return None
    released = block.memory_bytes(block.bitwidth) - block.memory_bytes(to_bits)
    return QuantizationAction(
        block_name=block.name,
        from_bits=block.bitwidth,
        to_bits=to_bits,
        released_bytes=released,
        predicted_damage=estimate_damage(block, block.bitwidth, to_bits),
    )


def _replace_block(
    blocks: Sequence[ExpertBlock], block_name: str, new_bitwidth: int
) -> tuple[ExpertBlock, ...]:
    replaced_blocks = []
    found = False
    for block in blocks:
        if block.name == block_name:
            replaced_blocks.append(replace(block, bitwidth=new_bitwidth))
            found = True
        else:
            replaced_blocks.append(block)
    if not found:
        raise KeyError(block_name)
    return tuple(replaced_blocks)


def quality_aware_plan(
    blocks: Iterable[ExpertBlock], requested_bytes: int
) -> PlanResult:
    """Greedily minimize predicted quality damage per released byte.

    This mirrors the paper's planning principle, not its exact calibrated policy.
    The candidate set is refreshed after every selected transition so a block can
    move through multiple bitwidth levels only when that remains cost-effective.
    """

    if requested_bytes < 0:
        raise ValueError("requested_bytes must be non-negative")

    state = tuple(blocks)
    actions: list[QuantizationAction] = []
    released = 0
    damage = 0.0

    while released < requested_bytes:
        candidates = [
            action
            for block in state
            if (action := candidate_action(block)) is not None
        ]
        if not candidates:
            break
        chosen = min(
            candidates,
            key=lambda action: (
                action.damage_per_released_byte,
                action.predicted_damage,
                action.block_name,
            ),
        )
        state = _replace_block(state, chosen.block_name, chosen.to_bits)
        actions.append(chosen)
        released += chosen.released_bytes
        damage += chosen.predicted_damage

    return PlanResult(
        blocks=state,
        actions=tuple(actions),
        requested_bytes=requested_bytes,
        released_bytes=released,
        predicted_damage=damage,
    )


def uniform_plan(blocks: Iterable[ExpertBlock], requested_bytes: int) -> PlanResult:
    """Static-like baseline: lower all legal blocks one level in lockstep."""

    if requested_bytes < 0:
        raise ValueError("requested_bytes must be non-negative")

    state = tuple(blocks)
    actions: list[QuantizationAction] = []
    released = 0
    damage = 0.0

    while released < requested_bytes:
        round_actions = [
            action
            for block in state
            if (action := candidate_action(block)) is not None
        ]
        if not round_actions:
            break
        for action in sorted(round_actions, key=lambda item: item.block_name):
            state = _replace_block(state, action.block_name, action.to_bits)
            actions.append(action)
            released += action.released_bytes
            damage += action.predicted_damage

    return PlanResult(
        blocks=state,
        actions=tuple(actions),
        requested_bytes=requested_bytes,
        released_bytes=released,
        predicted_damage=damage,
    )


def restore_with_headroom(
    blocks: Iterable[ExpertBlock], available_bytes: int
) -> tuple[tuple[ExpertBlock, ...], tuple[str, ...], int]:
    """Restore the highest-value pages while respecting available headroom.

    The value proxy is the damage avoided per added byte. The real PagedWeight
    system also coordinates page state and commits only after GPU residency.
    """

    if available_bytes < 0:
        raise ValueError("available_bytes must be non-negative")

    state = tuple(blocks)
    restored: list[str] = []
    used = 0

    while True:
        candidates: list[tuple[float, ExpertBlock, int, int]] = []
        for block in state:
            higher = next_higher_bitwidth(block.bitwidth)
            if higher is None:
                continue
            added = block.memory_bytes(higher) - block.memory_bytes(block.bitwidth)
            if used + added > available_bytes:
                continue
            avoided_damage = estimate_damage(block, higher, block.bitwidth)
            candidates.append((avoided_damage / added, block, higher, added))
        if not candidates:
            break
        _, block, higher, added = max(
            candidates, key=lambda item: (item[0], item[1].name)
        )
        state = _replace_block(state, block.name, higher)
        restored.append(f"{block.name}:{block.bitwidth}->{higher}")
        used += added

    return state, tuple(restored), used


def total_memory_bytes(blocks: Iterable[ExpertBlock]) -> int:
    return sum(block.memory_bytes() for block in blocks)


def mib(value: int) -> float:
    return value / (1024 * 1024)
