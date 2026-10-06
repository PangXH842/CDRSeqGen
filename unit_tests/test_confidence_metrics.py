import unittest

from cdrseqgen.eval.metrics import confidence_metrics, spearman


class SpearmanTests(unittest.TestCase):
    def test_perfect_and_inverted_rank_agreement(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_undefined_cases_return_none_rather_than_zero(self):
        """A constant score ranks nothing — reporting 0.0 would imply 'measured, no effect'."""
        self.assertIsNone(spearman([1, 2], [1, 2]))          # n < 3
        self.assertIsNone(spearman([5, 5, 5, 5], [1, 2, 3, 4]))  # no variance in x
        self.assertIsNone(spearman([1, 2, 3, 4], [7, 7, 7, 7]))  # no variance in y

    def test_ties_take_averaged_ranks(self):
        self.assertAlmostEqual(spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)


class ConfidenceMetricTests(unittest.TestCase):
    """Perplexity is lower-is-better, so orientation is the thing most likely to be wrong."""

    def test_perplexity_is_oriented_low_equals_confident(self):
        recoveries = [0.9, 0.5, 0.1]
        perplexities = [1.0, 5.0, 9.0]          # best candidate has the LOWEST perplexity
        out = confidence_metrics(recoveries, perplexities, prefix="c")
        self.assertAlmostEqual(out["c_spearman"], 1.0)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)

    def test_higher_is_better_scores_are_supported(self):
        out = confidence_metrics([0.9, 0.5, 0.1], [1.0, 0.5, 0.0],
                                 lower_is_better=False, prefix="c")
        self.assertAlmostEqual(out["c_spearman"], 1.0)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)

    def test_baseline_is_the_mean_over_the_same_pairs(self):
        """Lift must be measured against the same population, not against aar_mean."""
        out = confidence_metrics([1.0, 0.0, 0.5], [1.0, 2.0, 3.0], prefix="c")
        self.assertAlmostEqual(out["c_baseline_recovery"], 0.5)
        self.assertAlmostEqual(out["c_top1_recovery"], 1.0)

    def test_missing_values_drop_the_pair_not_the_alignment(self):
        # candidate 2's judge score is missing; it must not shift the others' pairing
        out = confidence_metrics([0.9, 0.4, 0.1, 0.8], [1.0, None, 9.0, 2.0], prefix="c")
        self.assertEqual(out["c_n"], 3)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)

    def test_too_few_pairs_reports_n_only(self):
        out = confidence_metrics([0.5, 0.4], [1.0, 2.0], prefix="c")
        self.assertEqual(out, {"c_n": 2})

    def test_unmatched_length_candidates_are_excluded_by_none_recovery(self):
        out = confidence_metrics([None, 0.5, 0.9, 0.2], [1.0, 2.0, 3.0, 4.0], prefix="c")
        self.assertEqual(out["c_n"], 3)

    def test_unequal_lengths_raise_before_dropping_missing_values(self):
        for recoveries, scores in (([0.9, 0.5, 0.1], [1.0, 2.0]),
                                   ([0.9, 0.5], [1.0, 2.0, 3.0]),
                                   ([], [None])):
            with self.subTest(recoveries=recoveries, scores=scores):
                with self.assertRaisesRegex(ValueError, "equal-length"):
                    confidence_metrics(recoveries, scores)

    def test_nonfinite_scores_drop_pairs_without_shifting_candidates(self):
        out = confidence_metrics(
            [0.9, 0.4, 0.1, 0.8, 0.6, 0.7],
            [1.0, float("nan"), 9.0, 2.0, float("inf"), float("-inf")],
            prefix="c",
        )
        self.assertEqual(out["c_n"], 3)
        self.assertAlmostEqual(out["c_spearman"], 1.0)
        self.assertAlmostEqual(out["c_baseline_recovery"], 0.6)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)
        self.assertAlmostEqual(out["c_top3_recovery"], 0.6)

    def test_nonfinite_recoveries_are_excluded(self):
        out = confidence_metrics(
            [0.9, float("nan"), 0.5, float("inf"), 0.1, float("-inf")],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            prefix="c",
        )
        self.assertEqual(out["c_n"], 3)
        self.assertAlmostEqual(out["c_spearman"], 1.0)
        self.assertAlmostEqual(out["c_baseline_recovery"], 0.5)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)

    def test_reporting_minimum_uses_only_finite_pairs(self):
        out = confidence_metrics([0.5, 0.4, 0.9], [1.0, 2.0, float("nan")], prefix="c")
        self.assertEqual(out, {"c_n": 2})

    def test_no_eligible_pairs_reports_zero(self):
        out = confidence_metrics(
            [None, float("nan"), float("inf"), float("-inf"), 0.5, 0.6, 0.7],
            [1.0, 1.0, 1.0, 1.0, None, float("nan"), float("inf")],
            prefix="c",
        )
        self.assertEqual(out, {"c_n": 0})


if __name__ == "__main__":
    unittest.main()
