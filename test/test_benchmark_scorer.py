import unittest

from benchmark.scorer import (
    aggregate_records,
    assert_comparable,
    hamming_fraction,
    score_record,
    validate_adapter_record,
)


def adapter_record(native="AAAAAA", samples=None):
    samples = samples if samples is not None else ["AAAAAA", "AACAAT"]
    return {
        "example_id": "example-1",
        "native": native,
        "native_length": len(native),
        "samples": samples,
        "attempts": 3,
        "valid_attempts": 2,
        "invalid_attempts": 1,
        "length_matches": 2,
        "length_mismatches": 0,
        "complete": True,
        "runtime_seconds": 0.6,
    }


class BenchmarkScorerTests(unittest.TestCase):
    def test_scores_whole_core_anchor_and_duplicate_preserving_design_metrics(self):
        record = adapter_record(samples=["AAAAAA", "AACAAT"])
        metrics = score_record(record, anchor_trim=2)

        self.assertAlmostEqual(metrics["whole_span_aar_mean"], 5 / 6)
        self.assertAlmostEqual(metrics["core_aar_mean"], 0.75)
        self.assertAlmostEqual(metrics["anchor_aar_mean"], 0.875)
        self.assertAlmostEqual(metrics["diversity_hamming"], 1 / 3)
        self.assertEqual(metrics["unique_fraction"], 1.0)
        self.assertAlmostEqual(metrics["valid_generation_rate"], 2 / 3)
        self.assertEqual(metrics["length_match_rate"], 1.0)
        self.assertAlmostEqual(metrics["runtime_seconds_per_retained_sample"], 0.3)

    def test_short_loop_has_no_core_and_anchors_cover_each_position_once(self):
        record = adapter_record(native="AAAA", samples=["AAAT", "AAAA"])
        record.update(attempts=2, valid_attempts=2, invalid_attempts=0)
        metrics = score_record(record, anchor_trim=2)

        self.assertIsNone(metrics["core_aar_mean"])
        self.assertFalse(metrics["core_available"])
        self.assertEqual(metrics["anchor_length"], 4)
        self.assertAlmostEqual(metrics["anchor_aar_mean"], 0.875)

    def test_duplicate_draws_reduce_unique_fraction_and_pairwise_diversity(self):
        record = adapter_record(samples=["AAAAAA", "AAAAAA", "TTTTTT"])
        record.update(attempts=3, valid_attempts=3, invalid_attempts=0, length_matches=3)
        metrics = score_record(record, anchor_trim=2)

        self.assertAlmostEqual(metrics["unique_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["diversity_hamming"], 2 / 3)
        self.assertEqual(hamming_fraction("AAA", "AAT"), 1 / 3)

    def test_rejects_inconsistent_adapter_accounting_or_bad_samples(self):
        record = adapter_record()
        manifest = {"native": "AAAAAA", "native_length": 6}
        validate_adapter_record(record, manifest)

        with self.assertRaisesRegex(ValueError, "attempt accounting"):
            validate_adapter_record({**record, "attempts": 4}, manifest)
        with self.assertRaisesRegex(ValueError, "wrong length"):
            validate_adapter_record({**record, "samples": ["AAA"]}, manifest)

    def test_comparison_requires_same_examples_and_sampling_controls(self):
        base = {
            "sampling": {"num_samples": 5, "temperature": 1.0, "top_p": 0.9, "base_seed": 42}
        }
        assert_comparable([("a", base, ["x"]), ("b", base, ["x"])])
        with self.assertRaisesRegex(ValueError, "framework IDs"):
            assert_comparable([("a", base, ["x"]), ("b", base, ["y"])])
        changed = {"sampling": {**base["sampling"], "top_p": 1.0}}
        with self.assertRaisesRegex(ValueError, "top_p"):
            assert_comparable([("a", base, ["x"]), ("b", changed, ["x"])])

    def test_aggregate_is_macro_average_over_examples(self):
        first = adapter_record(samples=["AAAAAA"])
        first.update(attempts=1, valid_attempts=1, invalid_attempts=0, length_matches=1)
        second = adapter_record(samples=["TTTTTT", "TTTTTT", "TTTTTT"])
        second.update(attempts=3, valid_attempts=3, invalid_attempts=0, length_matches=3)
        scored = []
        for index, record in enumerate((first, second)):
            metrics = score_record(record, anchor_trim=2)
            scored.append({
                "metrics": metrics,
                "accounting": {
                    "attempts": record["attempts"],
                    "valid_attempts": record["valid_attempts"],
                    "length_matches": record["length_matches"],
                },
            })
        aggregate = aggregate_records(scored)
        self.assertEqual(aggregate["whole_span_aar_mean"]["macro_mean"], 0.5)
        self.assertEqual(aggregate["micro_accounting"]["retained_samples"], 4)


if __name__ == "__main__":
    unittest.main()
