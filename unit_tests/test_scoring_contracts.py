"""Keep candidate populations intact when generator/judge scores are unavailable."""
import unittest
from unittest.mock import patch

import torch

from cdrseqgen.eval.generative import Draw, normalize_scores, run_eval_loop


class Strategy:
    def __init__(self, scores):
        self.scores = scores
        self.self_calls = 0

    def draw(self, index):
        return Draw(native="ACD", seqs=["ACD", "AQD", "AC"], spans=[None] * 3,
                    target_len=3, length_policy={"mode": "inferred"})

    def self_scores(self, drawn):
        self.self_calls += 1
        return self.scores


def evaluate(strategy, judge_scores=None, *, skip_self=False):
    with patch("cdrseqgen.eval.generative.score_with_judge", return_value=judge_scores):
        return run_eval_loop(
            strategy, [7], model=None, tokenizer=None, device="cpu",
            skip_perplexity=skip_self, judge_model=object() if judge_scores is not None else None,
            judge_tokenizer=None,
        )


class ScoreContractTests(unittest.TestCase):
    def test_ordered_scalar_iterators_are_normalized_without_changing_finite_values(self):
        values = iter([2.5, None, torch.tensor(4.0), 8])
        self.assertEqual(normalize_scores(values, 4, label="custom scorer"),
                         [2.5, None, 4.0, 8.0])
        self.assertEqual(normalize_scores([], 0, label="empty scorer"), [])

    def test_noniterable_and_nonnumeric_scores_have_a_labelled_error(self):
        with self.assertRaisesRegex(ValueError, "custom scorer.*one score"):
            normalize_scores(None, 1, label="custom scorer")
        for invalid in (True, "2.0", object(), {}, [2.0], torch.tensor([2.0]),
                        torch.tensor(True), 2 + 3j):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "custom scorer score 0.*scalar"):
                    normalize_scores([invalid], 1, label="custom scorer")

    def test_short_and_long_self_score_lists_fail_before_returning_partial_records(self):
        for scores in ([2.0], [2.0, 3.0, 4.0, 5.0]):
            with self.subTest(count=len(scores)):
                with self.assertRaisesRegex(ValueError, r"self scorer at eval index 7.*scores for 3 candidates"):
                    evaluate(Strategy(scores))

    def test_judge_score_count_is_checked_even_when_generator_scoring_is_skipped(self):
        strategy = Strategy(None)
        with self.assertRaisesRegex(ValueError, r"judge scorer at eval index 7.*scores for 3 candidates"):
            evaluate(strategy, [2.0], skip_self=True)
        self.assertEqual(strategy.self_calls, 0)

    def test_nonfinite_scores_do_not_remove_candidates_or_inflate_judge_coverage(self):
        strategy = Strategy([2.0, float("nan"), float("inf")])
        with self.assertLogs("cdrseqgen.eval.generative", level="WARNING"):
            records = evaluate(strategy, [4.0, float("-inf"), 8.0])
        record = records[0]
        self.assertEqual(record["samples"], ["ACD", "AQD", "AC"])
        self.assertEqual([c["seq"] for c in record["candidates"]], record["samples"])
        self.assertEqual([c["self_perplexity"] for c in record["candidates"]], [2.0, None, None])
        self.assertEqual([c["judge_perplexity"] for c in record["candidates"]], [4.0, None, 8.0])
        self.assertEqual(record["metrics"]["self_perplexity_mean"], 2.0)
        self.assertEqual(record["metrics"]["judge_perplexity_mean"], 6.0)
        self.assertEqual(record["metrics"]["judge_candidates_scored"], 2)
        self.assertEqual(record["metrics"]["judge_coverage"], 2 / 3)

    def test_all_unavailable_scores_leave_no_misleading_mean(self):
        with self.assertLogs("cdrseqgen.eval.generative", level="WARNING"):
            record = evaluate(Strategy([float("nan"), None, float("inf")]),
                              [None, float("nan"), float("inf")])[0]
        self.assertEqual(len(record["candidates"]), 3)
        self.assertNotIn("self_perplexity_mean", record["metrics"])
        self.assertNotIn("judge_perplexity_mean", record["metrics"])
        self.assertEqual(record["metrics"]["judge_candidates_scored"], 0)
        self.assertEqual(record["metrics"]["judge_coverage"], 0.0)

    def test_finite_scores_keep_the_existing_mean_and_candidate_order(self):
        record = evaluate(Strategy([2.0, 4.0, 6.0]))[0]
        self.assertEqual(record["metrics"]["self_perplexity_mean"], 4.0)
        self.assertEqual([c["self_perplexity"] for c in record["candidates"]], [2.0, 4.0, 6.0])
        self.assertEqual([c["judge_perplexity"] for c in record["candidates"]], [None] * 3)


class CaseScoreContractTests(unittest.TestCase):
    def test_eos_case_rejects_short_scores_and_retains_candidates_with_nonfinite_scores(self):
        from types import SimpleNamespace
        from cdrseqgen.methods.common.eos_decoding import eos_case_record
        args = SimpleNamespace(max_len=64, partner_chain=None, antigen=None, num_samples=2,
            temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=5, skip_perplexity=False, seed=42)
        options = dict(mode="causal", build_context=lambda *_: ([1, 2], {}),
            sample_loops=lambda *a, **kw: [dict(loop="AC", complete=True), dict(loop="AD", complete=True)])
        tokenizer = SimpleNamespace(encode=lambda seq: list(range(len(seq))))
        case = dict(pt1="AC", pt2="DE", target_truth="AC")
        with self.assertRaisesRegex(ValueError, "case self scorer"):
            eos_case_record(None, tokenizer, "cpu", args, None, case, lambda *a: None,
                            perplexities=lambda *a: [2.0], **options)
        record = eos_case_record(None, tokenizer, "cpu", args, None, case, lambda *a: None,
                                perplexities=lambda *a: [float("inf"), 2.0], **options)
        self.assertEqual(len(record["samples"]), 2)
        self.assertNotIn("perplexity", record["samples"][0])
        self.assertEqual(record["metrics"]["perplexity_mean"], 2.0)

    def test_mlm_case_excludes_nonfinite_scores_from_samples_and_summary(self):
        from types import SimpleNamespace
        from cdrseqgen.methods.mlm import generation
        args = SimpleNamespace(max_len=64, partner_chain=None, antigen=None, num_samples=1,
            temperature=1.0, top_k=0, top_p=1.0, decoding="independent", skip_perplexity=False, seed=42)
        tokenizer = SimpleNamespace(decode=lambda ids: "AC", special_ids={0}, mask_id=4)
        span = SimpleNamespace(token_ids=[5, 6], input_ids=[4, 4], mask_positions=[0, 1])
        with patch.object(generation, "sample_length_conditioned_spans", return_value=([span], {})), \
                patch.object(generation, "pseudo_log_likelihood", return_value=(0.0, float("nan"))):
            record = generation.generate_case(None, tokenizer, "cpu", args, None,
                dict(pt1="AC", pt2="DE", target_length=2), None, lambda *a: None)
        self.assertEqual(len(record["samples"]), 1)
        self.assertNotIn("pseudo_perplexity", record["samples"][0])
        self.assertNotIn("pseudo_perplexity_mean", record["metrics"])


if __name__ == "__main__":
    unittest.main()
