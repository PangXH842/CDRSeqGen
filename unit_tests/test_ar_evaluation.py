"""The causal evaluation strategy: the four things the shared loop cannot do for a causal model.

The index-resolution test exists because that bug was real and silent-ish: `framework_for_dataset`
already unwraps a Subset, so resolving the Subset index *before* calling it maps the index twice.
On a small eval subset it raises; on a large one it would quietly score the wrong framework.
"""
import json
import os
import tempfile
import unittest

from unittest.mock import patch
import torch

from cdrseqgen.datasets.dataset_ar import CausalInfillingDataset, causal_adapter
from cdrseqgen.eval.generative import build_masked_judge_prompt
from cdrseqgen.methods.ar.evaluation import (CausalStrategy,
                                             run_generative_eval)

CHAIN = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
         "fwr3": "FFFF", "cdr3": "GGGGG", "fwr4": "HHHH"}


def _data(n=6):
    rec = {"heavy": dict(CHAIN, full_aa="x"), "light": dict(CHAIN, full_aa="x"), "split": "train"}
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump([rec] * n, handle)
    handle.close()
    return handle.name


class StubModel:
    """Emits a fixed loop then <eos>, and returns flat logits for perplexity scoring."""

    def __init__(self, vocab_size, loop_ids, eos_id):
        self.vocab_size, self.loop_ids, self.eos_id = vocab_size, loop_ids, eos_id
        self.step = 0

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, **_):
        token = self.loop_ids[self.step] if self.step < len(self.loop_ids) else self.eos_id
        self.step += 1
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], self.vocab_size))
        logits[:, -1, token] = 10.0
        return {"logits": logits}


def _strategy(path, model=None, **kw):
    adapter = causal_adapter()
    ds = CausalInfillingDataset(path, adapter, target_region="h3")
    subset = torch.utils.data.Subset(ds, [2, 3, 4])
    loop_ids = [adapter.encode("AC")[0], adapter.encode("AC")[1]]
    model = model or StubModel(adapter.vocab_size, loop_ids, adapter.eos_id)
    defaults = dict(model=model, tokenizer=adapter, device="cpu", generator=None,
                    eval_dataset=subset, judge_dataset=ds, judge_tokenizer=adapter,
                    num_samples=2, temperature=0, top_k=0, top_p=1.0, max_new_tokens=8)
    defaults.update(kw)
    return CausalStrategy(**defaults), subset, ds, adapter


class IndexResolutionTests(unittest.TestCase):
    def test_the_subset_index_is_resolved_exactly_once(self):
        # framework_for_dataset unwraps the Subset itself; resolving first as well maps twice.
        path = _data()
        strategy, subset, full, _ = _strategy(path)
        self.assertEqual(strategy.framework(0), full.framework_parts(2))
        os.unlink(path)

    def test_the_judge_side_resolves_against_the_full_dataset(self):
        path = _data()
        strategy, subset, full, _ = _strategy(path)
        self.assertEqual(strategy._judge_framework(0), full.framework_parts(2))
        os.unlink(path)

    def test_a_judge_dataset_that_is_the_eval_subset_is_not_double_resolved(self):
        path = _data()
        adapter = causal_adapter()
        ds = CausalInfillingDataset(path, adapter, target_region="h3")
        subset = torch.utils.data.Subset(ds, [1, 2])
        strategy = CausalStrategy(model=None, tokenizer=adapter, device="cpu", generator=None,
                                  eval_dataset=subset, judge_dataset=subset,
                                  judge_tokenizer=adapter, num_samples=1, temperature=0,
                                  top_k=0, top_p=1.0)
        self.assertEqual(strategy._judge_framework(0), ds.framework_parts(1))
        os.unlink(path)

    def test_nested_eval_subsets_build_the_judge_prompt_from_the_original_record(self):
        path = _data()
        self.addCleanup(os.unlink, path)
        with open(path) as handle:
            records = json.load(handle)
        for index, record in enumerate(records):
            record["heavy"]["fwr1"] = "A" * (index + 1)
        with open(path, "w") as handle:
            json.dump(records, handle)

        adapter = causal_adapter()
        full = CausalInfillingDataset(path, adapter, target_region="h3")
        judge = CausalInfillingDataset(path, adapter, target_region="h3")
        subset = torch.utils.data.Subset(
            torch.utils.data.Subset(full, [4, 1, 5, 2]), [2, 0])
        strategy, *_ = _strategy(path, eval_dataset=subset, judge_dataset=judge)

        for subset_index, original_index in enumerate((5, 4)):
            with self.subTest(subset_index=subset_index):
                expected_framework = judge.framework_parts(original_index)
                self.assertEqual(strategy.framework(subset_index), expected_framework)
                self.assertEqual(
                    strategy.judge_inputs(subset_index, "ACD"),
                    build_masked_judge_prompt(adapter, expected_framework, 3))


class StrategyProtocolTests(unittest.TestCase):
    def test_length_policy_reports_eos_not_a_length_head(self):
        # Findings §10's gap-sentinel classifier has no counterpart on this side.
        path = _data()
        strategy, *_ = _strategy(path)
        self.assertEqual(strategy.payload_length_policy(),
                         {"mode": "inferred", "source": "eos_token"})
        self.assertFalse(strategy.supplies_native_length)
        os.unlink(path)

    def test_draw_returns_candidates_and_the_native(self):
        path = _data()
        strategy, *_ = _strategy(path)
        drawn = strategy.draw(0)
        self.assertEqual(drawn.native, "GGGGG")
        self.assertTrue(all(s == "AC" for s in drawn.seqs))
        self.assertEqual(drawn.length_policy["source"], "eos_token")
        os.unlink(path)

    def test_incomplete_draws_are_dropped_not_scored_as_short_loops(self):
        # A truncated decode has no length the model chose.
        class NeverStops(StubModel):
            def __call__(self, input_ids=None, attention_mask=None, **_):
                logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], self.vocab_size))
                logits[:, -1, self.loop_ids[0]] = 10.0
                return {"logits": logits}
        path = _data()
        adapter = causal_adapter()
        strategy, *_ = _strategy(path, model=NeverStops(adapter.vocab_size,
                                                        adapter.encode("A"), adapter.eos_id))
        self.assertIsNone(strategy.draw(0))
        os.unlink(path)

    def test_self_scores_are_finite_perplexities_one_per_candidate(self):
        path = _data()
        strategy, *_ = _strategy(path)
        drawn = strategy.draw(0)
        scores = strategy.self_scores(drawn)
        self.assertEqual(len(scores), len(drawn.seqs))
        self.assertTrue(all(s is not None and s > 0 for s in scores))
        os.unlink(path)


class JudgePromptTests(unittest.TestCase):
    def test_the_prompt_carries_exactly_the_candidate_length_in_masks(self):
        # score_with_judge checks this against the tokenized candidate before scoring.
        path = _data()
        adapter = causal_adapter()
        ds = CausalInfillingDataset(path, adapter, target_region="h3")
        for length in (3, 5, 11):
            _, positions = build_masked_judge_prompt(adapter, ds.framework_parts(0), length)
            self.assertEqual(len(positions), length)
        os.unlink(path)

    def test_judge_inputs_tracks_the_candidate_not_the_native_length(self):
        # Causal lengths vary by construction, so a native-length prompt would misalign.
        path = _data()
        strategy, *_ = _strategy(path)
        _, positions = strategy.judge_inputs(0, "ACDEFGH")
        self.assertEqual(len(positions), 7)
        os.unlink(path)


class ValidityAccountingTests(unittest.TestCase):
    def test_empty_and_incomplete_attempts_do_not_disappear(self):
        path = _data()
        strategy, *_ = _strategy(path)
        results = [
            {"loop": "", "length": 0, "complete": True},
            {"loop": "A", "length": 1, "complete": False},
            {"loop": "AC", "length": 2, "complete": True},
        ]
        with patch("cdrseqgen.methods.ar.evaluation.sample_loops", return_value=results):
            drawn = strategy.draw(0)

        self.assertEqual(drawn.seqs, ["AC"])
        self.assertEqual(drawn.record_extra, {
            "attempts": 3, "incomplete": 1, "empty": 1, "length_matches": 0,
            "length_absolute_error_sum_capped": 12,
        })
        self.assertEqual(strategy.attempts, 3)
        self.assertEqual(strategy.complete_nonempty, 1)
        self.assertEqual(strategy.complete_empty, 1)
        self.assertEqual(strategy.incomplete, 1)
        self.assertEqual(strategy.length_matches, 0)
        self.assertEqual(strategy.length_absolute_error_capped, 12)
        os.unlink(path)

    def test_payload_matches_the_shared_schema_and_records_validity(self):
        path = _data()
        strategy, subset, _, adapter = _strategy(path)
        payload = run_generative_eval(
            strategy.model, adapter, subset, device="cpu", num_examples=1,
            num_samples=2, temperature=0, max_new_tokens=8, seed=17,
            skip_perplexity=True, judge_name="",
            meta={"target_region": "h3"},
        )

        self.assertEqual(payload["schema_version"], "cdrseqgen.generation_eval.v1")
        self.assertEqual(payload["method"], "ar")
        self.assertEqual(payload["target_region"], "h3")
        self.assertEqual(payload["decoding"]["seed"], 17)
        self.assertEqual(payload["n_selected"], 1)
        self.assertEqual(payload["n_generated"], 1)
        self.assertIn("aar_mean", payload["summary"])
        validity = payload["generation_validity"]
        self.assertEqual(validity["attempts"], 1)
        self.assertEqual(validity["valid_generation_rate"], 1.0)
        self.assertEqual(validity["fixed_length_acceptance_rate"], 0.0)
        self.assertEqual(validity["length_absolute_error_mean_capped"], 3.0)
        os.unlink(path)

    def test_a_zero_validity_run_still_reports_its_attempt_budget(self):
        class NeverStops:
            def __init__(self, vocab_size, token):
                self.vocab_size, self.token = vocab_size, token

            def eval(self):
                return self

            def __call__(self, input_ids=None, attention_mask=None, **_):
                logits = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], self.vocab_size))
                logits[:, -1, self.token] = 10.0
                return {"logits": logits}

        path = _data()
        _, subset, _, adapter = _strategy(path)
        model = NeverStops(adapter.vocab_size, adapter.encode("A")[0])
        payload = run_generative_eval(
            model, adapter, subset, device="cpu", num_examples=1,
            temperature=0, max_new_tokens=3, skip_perplexity=True)

        self.assertEqual(payload["n_generated"], 0)
        validity = payload["generation_validity"]
        self.assertEqual(validity["attempts"], 1)
        self.assertEqual(validity["incomplete"], 1)
        self.assertEqual(validity["valid_generation_rate"], 0.0)
        self.assertEqual(validity["fixed_length_acceptance_rate"], 0.0)
        self.assertEqual(validity["length_absolute_error_mean_capped"], 2.0)
        os.unlink(path)


class JudgeScoringTests(unittest.TestCase):
    """`score_with_judge` had no test at all, which is how a NameError reached a GPU run.

    The extraction of the shared eval loop left `pseudo_log_likelihood` unimported in
    `eval/generative.py`. Every existing test ran with the judge disabled, so the whole suite passed
    while the one metric §11 says decides the AR-versus-MLM comparison could not be computed.
    """

    def test_the_judge_path_runs_end_to_end(self):
        import torch
        from cdrseqgen.eval.generative import score_with_judge

        adapter = causal_adapter()

        class StubJudge:
            """A masked LM standing in for IgBert: uniform logits over the vocabulary."""
            def __call__(self, input_ids=None, **_):
                return {"logits": torch.zeros(input_ids.shape[0], input_ids.shape[1],
                                              adapter.vocab_size)}
            def eval(self):
                return self

        class StubStrategy:
            def judge_inputs(self, i, seq):
                # A prompt whose mask count matches the candidate, as the real ones do.
                return [adapter.mask_id] * len(seq), list(range(len(seq)))

        scores = score_with_judge(StubStrategy(), 0, ["ACD", "ACDE"], StubJudge(), adapter, "cpu")
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(s is not None and s > 0 for s in scores),
                        f"judge returned {scores}")

    def test_a_tokenizer_length_disagreement_scores_none_rather_than_mis_scoring(self):
        import torch
        from cdrseqgen.eval.generative import score_with_judge

        adapter = causal_adapter()

        class StubJudge:
            def __call__(self, input_ids=None, **_):
                return {"logits": torch.zeros(input_ids.shape[0], input_ids.shape[1],
                                              adapter.vocab_size)}
            def eval(self):
                return self

        class MismatchedStrategy:
            def judge_inputs(self, i, seq):
                return [adapter.mask_id] * 2, [0, 1]      # always 2 masks, whatever the candidate

        scores = score_with_judge(MismatchedStrategy(), 0, ["ACDEF"], StubJudge(), adapter, "cpu")
        self.assertEqual(scores, [None])   # positional, so a skipped candidate stays aligned


if __name__ == "__main__":
    unittest.main()
