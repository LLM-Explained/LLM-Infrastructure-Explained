from __future__ import annotations

import unittest

from atrex_mini import HardwareProfile, Measurement, Shape, roofline_latency_ms, score_candidate
from example import build_demo


class AtrexMiniTests(unittest.TestCase):
    def test_roofline_uses_slower_bound(self) -> None:
        hardware = HardwareProfile(compute_tflops=100.0, memory_bandwidth_gbps=1000.0)
        compute_bound = Shape("gemm", "a", flops=2e9, bytes_moved=1e6)
        memory_bound = Shape("norm", "b", flops=1e6, bytes_moved=20e6)
        self.assertAlmostEqual(roofline_latency_ms(compute_bound, hardware), 0.02)
        self.assertAlmostEqual(roofline_latency_ms(memory_bound, hardware), 0.02)

    def test_ranking_reverses_under_production_weighting(self) -> None:
        hardware, shapes, weights, candidate_a, candidate_b = build_demo()
        score_a = score_candidate("A", shapes, candidate_a, weights, hardware)
        score_b = score_candidate("B", shapes, candidate_b, weights, hardware)
        self.assertGreater(score_a.correctness_rate, score_b.correctness_rate)
        self.assertGreater(score_b.production_weighted_score, score_a.production_weighted_score)
        self.assertGreater(score_b.target_dsl_fraction, score_a.target_dsl_fraction)

    def test_operator_with_no_correct_shape_scores_zero(self) -> None:
        hardware = HardwareProfile(100.0, 1000.0)
        shapes = [Shape("attention", "x", 1e9, 1e6)]
        score = score_candidate(
            "broken",
            shapes,
            {("attention", "x"): Measurement(1.0, compiled=False, correct=False)},
            {"attention": 1.0},
            hardware,
        )
        self.assertEqual(score.operator_scores["attention"], 0.0)
        self.assertEqual(score.production_weighted_score, 0.0)

    def test_sub_roofline_measurement_is_rejected(self) -> None:
        hardware = HardwareProfile(100.0, 1000.0)
        shape = Shape("gemm", "x", 1e9, 1e6)  # 0.01 ms compute bound
        with self.assertRaises(ValueError):
            score_candidate(
                "impossible",
                [shape],
                {("gemm", "x"): Measurement(0.005)},
                {"gemm": 1.0},
                hardware,
            )


if __name__ == "__main__":
    unittest.main()
