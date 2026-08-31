import unittest

from cdrseqgen.visualization.confidence_plot import (
    baseline_recovery,
    rho_distribution,
    selection_curve,
)


def record(scores, recoveries, key="self_perplexity"):
    return {"candidates": [{"recovery": r, key: s} for s, r in zip(scores, recoveries)]}


class SelectionCurveTests(unittest.TestCase):
    def test_low_perplexity_is_selected_first(self):
        """Perplexity is lower-is-better; sorting the wrong way would invert every conclusion."""
        r = record([1.0, 5.0, 9.0], [0.9, 0.5, 0.1])
        curve = selection_curve([r], "self_perplexity", 3)
        self.assertAlmostEqual(curve[0], 0.9)                 # best candidate first
        self.assertAlmostEqual(curve[2], (0.9 + 0.5 + 0.1) / 3)  # k = N -> the mean

    def test_curve_converges_to_the_random_baseline_at_k_equals_n(self):
        r = record([3.0, 1.0, 2.0], [0.2, 0.8, 0.4])
        self.assertAlmostEqual(selection_curve([r], "self_perplexity", 3)[-1],
                               baseline_recovery([r]))

    def test_frameworks_shorter_than_k_contribute_all_they_have(self):
        """Dropping them instead would change the population as k grows, so the curve would
        measure which frameworks survive rather than how well the score ranks."""
        short, long = record([1.0, 2.0], [1.0, 0.0]), record([1.0, 2.0, 3.0], [0.6, 0.3, 0.0])
        curve = selection_curve([short, long], "self_perplexity", 3)
        self.assertAlmostEqual(curve[2], (0.5 + 0.3) / 2)     # short still averaged over its 2

    def test_candidates_without_a_defined_recovery_are_excluded(self):
        r = {"candidates": [{"recovery": None, "self_perplexity": 0.1},
                            {"recovery": 0.4, "self_perplexity": 5.0},
                            {"recovery": 0.6, "self_perplexity": 6.0}]}
        # the wrong-length candidate has the best perplexity but must not be picked
        self.assertAlmostEqual(selection_curve([r], "self_perplexity", 1)[0], 0.4)

    def test_missing_scores_do_not_shift_the_ranking(self):
        r = {"candidates": [{"recovery": 0.9, "judge_perplexity": 1.0},
                            {"recovery": 0.1, "judge_perplexity": None},
                            {"recovery": 0.5, "judge_perplexity": 2.0}]}
        self.assertAlmostEqual(selection_curve([r], "judge_perplexity", 1)[0], 0.9)


class RhoDistributionTests(unittest.TestCase):
    def test_one_rho_per_framework_oriented_for_perplexity(self):
        good = record([1.0, 2.0, 3.0], [0.9, 0.5, 0.1])   # confident == accurate
        bad = record([1.0, 2.0, 3.0], [0.1, 0.5, 0.9])    # confident == wrong
        self.assertAlmostEqual(rho_distribution([good], "self_perplexity")[0], 1.0)
        self.assertAlmostEqual(rho_distribution([bad], "self_perplexity")[0], -1.0)

    def test_frameworks_with_too_few_candidates_are_skipped(self):
        self.assertEqual(rho_distribution([record([1.0, 2.0], [0.5, 0.4])], "self_perplexity"), [])


if __name__ == "__main__":
    unittest.main()


class AutoFigureTests(unittest.TestCase):
    """evaluate.py / post-training generation write the figure beside the JSON automatically."""

    def _payload(self, with_scores=True):
        cands = [{"recovery": 0.9, "self_perplexity": 1.0 if with_scores else None,
                  "judge_perplexity": None},
                 {"recovery": 0.5, "self_perplexity": 5.0 if with_scores else None,
                  "judge_perplexity": None},
                 {"recovery": 0.1, "self_perplexity": 9.0 if with_scores else None,
                  "judge_perplexity": None}]
        return {"records": [{"candidates": cands}]}

    def test_figure_is_written_next_to_the_json(self):
        import tempfile, os
        from cdrseqgen.methods.mlm.evaluation import write_confidence_figure
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "generation_eval.json")
            path = write_confidence_figure(self._payload(), out)
            self.assertEqual(path, os.path.join(tmp, "generation_eval_confidence.png"))
            self.assertGreater(os.path.getsize(path), 0)

    def test_nothing_to_plot_is_skipped_not_failed(self):
        """--skip_perplexity with no judge leaves no scores; that's a config choice, not an error."""
        import tempfile, os
        from cdrseqgen.methods.mlm.evaluation import write_confidence_figure
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "generation_eval.json")
            self.assertIsNone(write_confidence_figure(self._payload(with_scores=False), out))
            self.assertEqual(os.listdir(tmp), [])

    def test_no_output_path_writes_nothing(self):
        from cdrseqgen.methods.mlm.evaluation import write_confidence_figure
        self.assertIsNone(write_confidence_figure(self._payload(), None))
