import unittest

from c2kv_demo import compose, evaluate, extract_document, make_documents, rope


class C2KVDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = make_documents()
        self.caches = {doc.name: extract_document(doc, 4) for doc in self.documents}

    def test_compression_ratio(self) -> None:
        result = evaluate(self.documents, block_size=4, queries_per_order=4)
        self.assertEqual(result.compression_ratio, 4.0)

    def test_position_agnostic_composition_matches_recompute(self) -> None:
        order = tuple(reversed([doc.name for doc in self.documents]))
        expected = compose(self.caches, order, rerotate=True)
        actual = compose(self.caches, order, rerotate=True)
        self.assertEqual(actual, expected)

    def test_stale_position_cache_diverges_after_reordering(self) -> None:
        order = tuple(reversed([doc.name for doc in self.documents]))
        ideal = compose(self.caches, order, rerotate=True)
        stale = compose(self.caches, order, rerotate=False)
        self.assertNotEqual(ideal, stale)

    def test_c2kv_preserves_attention_better_than_stale_baseline(self) -> None:
        result = evaluate(self.documents, block_size=4, queries_per_order=16)
        self.assertEqual(result.c2kv_top1_agreement, 1.0)
        self.assertGreater(result.c2kv_top1_agreement, result.stale_top1_agreement)
        self.assertLess(result.c2kv_mean_kl, result.stale_mean_kl)

    def test_rope_is_identity_at_zero(self) -> None:
        vector = (1.0, 2.0, -3.0, 4.0)
        self.assertEqual(rope(vector, 0), vector)


if __name__ == "__main__":
    unittest.main()
