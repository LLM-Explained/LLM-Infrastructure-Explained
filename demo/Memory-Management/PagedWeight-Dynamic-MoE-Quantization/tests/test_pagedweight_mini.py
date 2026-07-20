import unittest

from pagedweight_mini import (
    ExpertBlock,
    quality_aware_plan,
    restore_with_headroom,
    total_memory_bytes,
    uniform_plan,
)


PARAMS = 8 * 1024 * 1024


def blocks():
    return (
        ExpertBlock("hot", PARAMS, 16, 8, 1.0, 0.50, 0.2),
        ExpertBlock("cold", PARAMS, 16, 2, 0.1, 0.01, -0.2),
        ExpertBlock("medium", PARAMS, 16, 4, 0.4, 0.10, 0.0),
    )


class PagedWeightMiniTests(unittest.TestCase):
    def test_plan_meets_releasable_target(self):
        result = quality_aware_plan(blocks(), 8 * 1024 * 1024)
        self.assertTrue(result.target_met)
        self.assertGreaterEqual(result.released_bytes, 8 * 1024 * 1024)

    def test_cold_expert_is_quantized_before_hot_expert(self):
        result = quality_aware_plan(blocks(), 4 * 1024 * 1024)
        self.assertEqual(result.actions[0].block_name, "cold")
        final = {block.name: block.bitwidth for block in result.blocks}
        self.assertEqual(final["hot"], 16)

    def test_quality_aware_plan_has_lower_damage_than_uniform(self):
        target = 12 * 1024 * 1024
        adaptive = quality_aware_plan(blocks(), target)
        uniform = uniform_plan(blocks(), target)
        self.assertTrue(adaptive.target_met)
        self.assertTrue(uniform.target_met)
        self.assertLess(adaptive.predicted_damage, uniform.predicted_damage)

    def test_bitwidth_floor_is_respected(self):
        result = quality_aware_plan(blocks(), 1 << 60)
        final = {block.name: block.bitwidth for block in result.blocks}
        self.assertEqual(final["hot"], 8)
        self.assertEqual(final["cold"], 2)
        self.assertEqual(final["medium"], 4)
        self.assertFalse(result.target_met)

    def test_restore_uses_headroom_and_increases_memory(self):
        quantized = quality_aware_plan(blocks(), 12 * 1024 * 1024)
        before = total_memory_bytes(quantized.blocks)
        restored, transitions, used = restore_with_headroom(
            quantized.blocks, 8 * 1024 * 1024
        )
        self.assertGreater(len(transitions), 0)
        self.assertLessEqual(used, 8 * 1024 * 1024)
        self.assertGreater(total_memory_bytes(restored), before)


if __name__ == "__main__":
    unittest.main()
