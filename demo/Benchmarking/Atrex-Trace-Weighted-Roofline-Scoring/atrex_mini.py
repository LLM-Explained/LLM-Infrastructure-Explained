"""A dependency-free miniature of Atrex-Bench's deployment-weighted scoring.

This module reproduces the paper's central evaluation idea:

1. derive a per-shape speed-of-light latency from compute and memory bounds;
2. gate performance on correctness;
3. aggregate shapes into operators with a median;
4. weight operators by their share of production GPU time.

It is an educational model, not a replacement for the full Atrex-Bench evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class HardwareProfile:
    """Peak hardware capabilities used by the roofline lower bound."""

    compute_tflops: float
    memory_bandwidth_gbps: float

    def __post_init__(self) -> None:
        if self.compute_tflops <= 0 or self.memory_bandwidth_gbps <= 0:
            raise ValueError("hardware peaks must be positive")


@dataclass(frozen=True)
class Shape:
    """One operator/shape evaluation unit."""

    operator: str
    shape_id: str
    flops: float
    bytes_moved: float

    def __post_init__(self) -> None:
        if not self.operator or not self.shape_id:
            raise ValueError("operator and shape_id must be non-empty")
        if self.flops < 0 or self.bytes_moved < 0:
            raise ValueError("semantic work and traffic cannot be negative")
        if self.flops == 0 and self.bytes_moved == 0:
            raise ValueError("a shape must perform work or move data")


@dataclass(frozen=True)
class Measurement:
    """A candidate kernel result for one shape."""

    latency_ms: float
    compiled: bool = True
    correct: bool = True
    target_dsl_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.latency_ms <= 0:
            raise ValueError("latency_ms must be positive")
        if not 0.0 <= self.target_dsl_fraction <= 1.0:
            raise ValueError("target_dsl_fraction must lie in [0, 1]")


@dataclass(frozen=True)
class CandidateScore:
    name: str
    compile_rate: float
    correctness_rate: float
    target_dsl_fraction: float
    unweighted_operator_score: float
    production_weighted_score: float
    operator_scores: Mapping[str, float]


def roofline_latency_ms(shape: Shape, hardware: HardwareProfile) -> float:
    """Return max(F/P, M/B), expressed in milliseconds."""

    compute_seconds = shape.flops / (hardware.compute_tflops * 1e12)
    memory_seconds = shape.bytes_moved / (hardware.memory_bandwidth_gbps * 1e9)
    return max(compute_seconds, memory_seconds) * 1e3


def shape_achievement(
    shape: Shape,
    measurement: Measurement,
    hardware: HardwareProfile,
    *,
    tolerance: float = 1e-9,
) -> float | None:
    """Return T_roofline / T_candidate for a correct unit, otherwise None.

    Atrex treats a score above one as an evaluation error rather than clipping it,
    because the semantic bound, units, or measurement are then inconsistent.
    """

    if not measurement.compiled or not measurement.correct:
        return None

    roofline_ms = roofline_latency_ms(shape, hardware)
    score = roofline_ms / measurement.latency_ms
    if score > 1.0 + tolerance:
        raise ValueError(
            f"{shape.operator}/{shape.shape_id}: measured latency "
            f"{measurement.latency_ms:.6g} ms is below roofline "
            f"{roofline_ms:.6g} ms"
        )
    return min(score, 1.0)


def _normalize_weights(weights: Mapping[str, float], operators: Iterable[str]) -> dict[str, float]:
    selected = {operator: float(weights.get(operator, 0.0)) for operator in operators}
    if any(weight < 0 for weight in selected.values()):
        raise ValueError("production weights cannot be negative")
    total = sum(selected.values())
    if total <= 0:
        raise ValueError("at least one retained operator must have positive weight")
    return {operator: weight / total for operator, weight in selected.items()}


def score_candidate(
    name: str,
    shapes: Sequence[Shape],
    measurements: Mapping[tuple[str, str], Measurement],
    production_weights: Mapping[str, float],
    hardware: HardwareProfile,
) -> CandidateScore:
    """Score a candidate using the paper's shape -> operator -> fleet aggregation."""

    if not shapes:
        raise ValueError("shapes cannot be empty")

    operators = sorted({shape.operator for shape in shapes})
    weights = _normalize_weights(production_weights, operators)
    achievements: dict[str, list[float]] = {operator: [] for operator in operators}

    compiled = 0
    correct = 0
    dsl_time_proxy = 0.0

    for shape in shapes:
        key = (shape.operator, shape.shape_id)
        measurement = measurements.get(
            key,
            Measurement(latency_ms=1.0, compiled=False, correct=False, target_dsl_fraction=0.0),
        )
        compiled += int(measurement.compiled)
        correct += int(measurement.compiled and measurement.correct)
        dsl_time_proxy += measurement.target_dsl_fraction if measurement.compiled else 0.0

        achievement = shape_achievement(shape, measurement, hardware)
        if achievement is not None:
            achievements[shape.operator].append(achievement)

    operator_scores = {
        operator: median(values) if values else 0.0
        for operator, values in achievements.items()
    }
    unweighted = sum(operator_scores.values()) / len(operator_scores)
    weighted = sum(weights[operator] * operator_scores[operator] for operator in operators)
    count = len(shapes)

    return CandidateScore(
        name=name,
        compile_rate=compiled / count,
        correctness_rate=correct / count,
        target_dsl_fraction=dsl_time_proxy / count,
        unweighted_operator_score=unweighted,
        production_weighted_score=weighted,
        operator_scores=operator_scores,
    )
