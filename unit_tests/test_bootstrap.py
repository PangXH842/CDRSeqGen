"""Tests for the paired bootstrap in benchmark/bootstrap.py.

The three properties worth pinning are the ones that decide whether an interval means anything:

  * **pairing** — one index draw must be applied to both arms. An unpaired bootstrap of the same
    data gives a much wider interval, so a bug here silently converts resolved findings into
    unresolved ones (or, with the error in the other direction, manufactures significance).
  * **the paired subset** — a metric may be missing on different examples for each model
    (positional AAR is defined only where a candidate matched the native length), and the two arms
    must be averaged over the *same* examples or they are being scored on different problems.
  * **direction** — for a lower-is-better metric like perplexity, a positive difference means the
    first model is *worse*. Getting this backwards would name the losing model as the winner while
    every number on the row stays correct.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmark"))

from bootstrap import compare, load_runs, paired_bootstrap, paired_values  # noqa: E402


def _scores(records_a, records_b):
    """A minimal two-run scores payload; each record list is (example_id, {metric: value})."""
    return {
        "manifest": {"id": "test_manifest"},
        "judge": {"model": "test-judge"},
        "runs": [
            {"label": "cdrseqgen",
             "records": [{"example_id": i, "metrics": m} for i, m in records_a]},
            {"label": "iglm",
             "records": [{"example_id": i, "metrics": m} for i, m in records_b]},
        ],
    }


def _write(payload):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(payload, handle)
    handle.close()
    return handle.name


class PairedSubsetTests(unittest.TestCase):
    def test_only_examples_both_runs_report_are_compared(self):
        payload = _scores(
            [("a", {"core_aar_mean": 0.2}), ("b", {"core_aar_mean": None}),
             ("c", {"core_aar_mean": 0.4})],
            [("a", {"core_aar_mean": 0.1}), ("b", {"core_aar_mean": 0.9}),
             ("c", {"core_aar_mean": None})],
        )
        _, runs = load_runs(_write(payload))
        ids, values_a, values_b = paired_values(runs, "core_aar_mean")
        # Only "a" has a finite value on both sides; b and c are each missing on one arm.
        self.assertEqual(ids, ["a"])
        self.assertEqual((values_a, values_b), ([0.2], [0.1]))

    def test_a_metric_missing_everywhere_is_dropped_not_zeroed(self):
        payload = _scores([("a", {"core_aar_mean": None})], [("a", {"core_aar_mean": None})])
        _, runs = load_runs(_write(payload))
        self.assertEqual(paired_values(runs, "core_aar_mean"), ([], [], []))
        self.assertIsNone(paired_bootstrap([], []))


class PairingTests(unittest.TestCase):
    def test_a_constant_offset_gives_a_zero_width_interval(self):
        # Every example differs by exactly +0.1, so the difference has no variance and pairing must
        # cancel the (large) between-example spread entirely. An unpaired bootstrap would not.
        values_a = [0.1, 0.5, 0.9, 0.3, 0.7]
        values_b = [v - 0.1 for v in values_a]
        stats = paired_bootstrap(values_a, values_b, iterations=2000, seed=1)
        self.assertAlmostEqual(stats["difference"], 0.1, places=9)
        self.assertAlmostEqual(stats["ci_low"], 0.1, places=9)
        self.assertAlmostEqual(stats["ci_high"], 0.1, places=9)
        self.assertTrue(stats["excludes_zero"])

    def test_the_interval_brackets_the_observed_difference(self):
        stats = paired_bootstrap([0.5, 0.2, 0.9, 0.1], [0.4, 0.4, 0.4, 0.4],
                                 iterations=2000, seed=7)
        self.assertLessEqual(stats["ci_low"], stats["difference"])
        self.assertGreaterEqual(stats["ci_high"], stats["difference"])

    def test_noise_around_zero_does_not_resolve(self):
        values_a = [0.30, 0.50, 0.20, 0.60, 0.40]
        values_b = [0.50, 0.30, 0.60, 0.20, 0.40]
        stats = paired_bootstrap(values_a, values_b, iterations=4000, seed=3)
        self.assertFalse(stats["excludes_zero"])

    def test_resampling_is_deterministic_for_a_seed(self):
        args = ([0.1, 0.4, 0.9], [0.2, 0.2, 0.2])
        first = paired_bootstrap(*args, iterations=500, seed=11)
        second = paired_bootstrap(*args, iterations=500, seed=11)
        self.assertEqual(first, second)


class DirectionTests(unittest.TestCase):
    def test_lower_is_better_names_the_smaller_value_as_the_winner(self):
        # cdrseqgen perplexity is higher, so iglm must be credited despite the positive difference.
        payload = _scores(
            [(str(i), {"judge_perplexity_mean": 9.0}) for i in range(8)],
            [(str(i), {"judge_perplexity_mean": 4.0}) for i in range(8)],
        )
        result = compare(_write(payload), iterations=500)
        row = next(r for r in result["results"] if r["metric"] == "judge_perplexity_mean")
        self.assertGreater(row["difference"], 0)
        self.assertEqual(row["better"], "iglm")

    def test_higher_is_better_names_the_larger_value_as_the_winner(self):
        payload = _scores(
            [(str(i), {"diversity_hamming": 0.8}) for i in range(8)],
            [(str(i), {"diversity_hamming": 0.6}) for i in range(8)],
        )
        result = compare(_write(payload), iterations=500)
        row = next(r for r in result["results"] if r["metric"] == "diversity_hamming")
        self.assertEqual(row["better"], "cdrseqgen")

    def test_an_unresolved_metric_names_no_winner(self):
        payload = _scores(
            [("a", {"core_aar_mean": 0.30}), ("b", {"core_aar_mean": 0.50}),
             ("c", {"core_aar_mean": 0.20})],
            [("a", {"core_aar_mean": 0.50}), ("b", {"core_aar_mean": 0.20}),
             ("c", {"core_aar_mean": 0.35})],
        )
        result = compare(_write(payload), iterations=2000)
        row = next(r for r in result["results"] if r["metric"] == "core_aar_mean")
        self.assertIsNone(row["better"])


class PayloadTests(unittest.TestCase):
    def test_a_payload_without_exactly_two_runs_is_refused(self):
        payload = _scores([("a", {"core_aar_mean": 0.2})], [("a", {"core_aar_mean": 0.1})])
        payload["runs"] = payload["runs"][:1]
        with self.assertRaises(SystemExit):
            load_runs(_write(payload))


if __name__ == "__main__":
    unittest.main()
