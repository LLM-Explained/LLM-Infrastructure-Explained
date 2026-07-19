from __future__ import annotations

import unittest

from pcp_mini import (
    contiguous_partition,
    covered_positions,
    decode_replica_ranks,
    dual_chunk_swap_partition,
    kv_shards,
    kv_tokens_per_rank,
    restore_global_order,
)


class PCPMiniTests(unittest.TestCase):
    def test_dual_chunk_swap_covers_every_token_once(self) -> None:
        result = dual_chunk_swap_partition([("r", 101)], pcp_size=4)
        self.assertEqual(covered_positions(result, "r"), list(range(101)))

    def test_dual_chunk_swap_reduces_causal_work_imbalance(self) -> None:
        requests = [("long", 100_000), ("medium", 32_768)]
        contiguous = contiguous_partition(requests, pcp_size=4)
        balanced = dual_chunk_swap_partition(requests, pcp_size=4)
        self.assertLess(balanced.imbalance, contiguous.imbalance)
        self.assertLess(balanced.imbalance, 0.10)

    def test_restore_global_order(self) -> None:
        result = dual_chunk_swap_partition([("r", 16)], pcp_size=2)
        outputs = []
        for chunks in result.assignments:
            values = []
            for chunk in chunks:
                values.extend(range(chunk.start, chunk.end))
            outputs.append(values)
        self.assertEqual(restore_global_order(result, "r", outputs), list(range(16)))

    def test_decode_rows_are_replicated(self) -> None:
        self.assertEqual(decode_replica_ranks(4), (0, 1, 2, 3))

    def test_kv_sharding_is_independent_of_pcp(self) -> None:
        self.assertEqual(kv_shards(2), 2)
        self.assertEqual(kv_tokens_per_rank(100_000, 2), 50_000)
        # There is intentionally no PCP parameter in the correct API.
        self.assertEqual(kv_tokens_per_rank(100_000, 2), 50_000)


if __name__ == "__main__":
    unittest.main()
