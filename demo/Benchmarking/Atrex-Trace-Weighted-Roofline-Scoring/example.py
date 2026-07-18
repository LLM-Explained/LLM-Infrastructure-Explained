"""Deterministic ranking-reversal example for the Atrex-style metric."""

from __future__ import annotations

from atrex_mini import HardwareProfile, Measurement, Shape, score_candidate


def build_demo():
    hardware = HardwareProfile(compute_tflops=100.0, memory_bandwidth_gbps=1000.0)

    # The head operators dominate serving time; the tail operators are numerous
    # but individually cheap. Each shape has a 0.01 ms roofline lower bound.
    shapes = [
        Shape("unified_attention", "prefill", flops=1e9, bytes_moved=2e6),
        Shape("fused_moe", "decode", flops=8e8, bytes_moved=1e6),
        Shape("rms_norm", "short", flops=2e7, bytes_moved=1e7),
        Shape("silu_and_mul", "short", flops=2e7, bytes_moved=1e7),
        Shape("reshape_and_cache", "decode", flops=1e6, bytes_moved=1e7),
    ]
    weights = {
        "unified_attention": 0.55,
        "fused_moe": 0.25,
        "rms_norm": 0.08,
        "silu_and_mul": 0.07,
        "reshape_and_cache": 0.05,
    }

    # Candidate A is correct on every operator, but delegates the production-heavy
    # head to slow fallbacks. Raw correctness makes it look excellent.
    fallback_friendly = {
        ("unified_attention", "prefill"): Measurement(0.50, target_dsl_fraction=0.05),
        ("fused_moe", "decode"): Measurement(0.40, target_dsl_fraction=0.10),
        ("rms_norm", "short"): Measurement(0.020, target_dsl_fraction=1.00),
        ("silu_and_mul", "short"): Measurement(0.018, target_dsl_fraction=1.00),
        ("reshape_and_cache", "decode"): Measurement(0.016, target_dsl_fraction=1.00),
    }

    # Candidate B misses one rare tail operator, but writes much faster native
    # kernels for attention and MoE. Production weighting should prefer it.
    production_focused = {
        ("unified_attention", "prefill"): Measurement(0.025, target_dsl_fraction=0.98),
        ("fused_moe", "decode"): Measurement(0.030, target_dsl_fraction=0.96),
        ("rms_norm", "short"): Measurement(0.040, target_dsl_fraction=0.95),
        ("silu_and_mul", "short"): Measurement(0.050, target_dsl_fraction=0.94),
        ("reshape_and_cache", "decode"): Measurement(
            0.020, compiled=False, correct=False, target_dsl_fraction=0.0
        ),
    }

    return hardware, shapes, weights, fallback_friendly, production_focused


def main() -> None:
    hardware, shapes, weights, candidate_a, candidate_b = build_demo()
    scores = [
        score_candidate("Fallback-friendly", shapes, candidate_a, weights, hardware),
        score_candidate("Production-focused", shapes, candidate_b, weights, hardware),
    ]

    print("Atrex-style miniature: correctness is not deployability\n")
    print(
        f"{'Candidate':<22} {'Correct':>8} {'DSL share':>10} "
        f"{'Uniform S':>10} {'Weighted S':>11}"
    )
    print("-" * 66)
    for score in scores:
        print(
            f"{score.name:<22} "
            f"{score.correctness_rate:>8.1%} "
            f"{score.target_dsl_fraction:>10.1%} "
            f"{score.unweighted_operator_score:>10.3f} "
            f"{score.production_weighted_score:>11.3f}"
        )

    raw_winner = max(scores, key=lambda item: item.correctness_rate)
    production_winner = max(scores, key=lambda item: item.production_weighted_score)
    print()
    print(f"Raw-correctness winner:   {raw_winner.name}")
    print(f"Production-score winner: {production_winner.name}")
    print("\nPer-operator roofline achievement:")
    for score in scores:
        details = ", ".join(
            f"{operator}={value:.3f}" for operator, value in score.operator_scores.items()
        )
        print(f"  {score.name}: {details}")


if __name__ == "__main__":
    main()
