"""Tests merged from: test_ar_backbones.py, test_ar_evaluation.py, test_ar_model.py, test_ar_sampling.py, test_ar_training.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import unittest
import torch
from cdrseqgen.datasets.dataset_ar import CausalTokenizer
from cdrseqgen.methods.ar import backbones
from cdrseqgen.methods.ar.backbones import (
    _prepare_hf_tokenizer,
    REQUIRED_SENTINELS,
    missing_sentinels,
    resolve,
)
import json
import os
import tempfile
from unittest.mock import patch
from cdrseqgen.datasets.dataset_ar import CausalInfillingDataset, causal_adapter
from cdrseqgen.eval.generative import build_masked_judge_prompt
from cdrseqgen.methods.ar.evaluation import CausalStrategy, run_generative_eval
from cdrseqgen.methods.ar.models.transformer import CausalAntibodyModel, build_model
from cdrseqgen.datasets.dataset_ar import LOOP_LENGTH_RESERVE, CausalTokenizer, causal_adapter
from cdrseqgen.methods.ar.sampling import _left_pad, sample_loops
import numpy as np
from cdrseqgen.datasets.dataset_common import select_split_indices
from cdrseqgen.methods.ar.backbones import resolve
from cdrseqgen.methods.ar.training import compute_metrics, get_datasets


# ====================================================================================================
# from test_ar_backbones.py
# The causal backbone registry: dispatch, vocab agreement, and the pretrained-sentinel problem.
#
# Two failure modes are silent and expensive, so both are pinned:
#
#   * **the fallback shadowing a named backbone.** `hf_decoder` matches any string containing "/", so
#     if it were registered before a named entry every hub id would resolve to the generic loader and
#     look like it worked.
#   * **vocab disagreement between tokenizer and model.** The causal vocab is 27 (the shared 25 plus
#     <bos>/<eos>); building a model sized 25 makes <eos> unrepresentable, which would silently
#     disable the very mechanism that gives an autoregressive model its length.
# ====================================================================================================

class DispatchTests(unittest.TestCase):
    def test_scratch_resolves_to_the_custom_decoder(self):
        self.assertEqual(resolve("scratch").id, "scratch")

    def test_a_hub_id_falls_back_to_the_generic_causal_loader(self):
        for name in ("nferruz/ProtGPT2", "hugohrban/progen2-small", "./local_run"):
            self.assertEqual(resolve(name).id, "hf_decoder")

    def test_an_unknown_bare_name_is_refused_with_the_registry_listed(self):
        with self.assertRaises(ValueError) as ctx:
            resolve("not_a_backbone")
        self.assertIn("scratch", str(ctx.exception))

    def test_the_generic_fallback_is_registered_last(self):
        # Registered earlier, it would shadow every named backbone whose id contains a slash.
        flags = [b.fallback for b in backbones.REGISTRY]
        self.assertEqual(flags.count(True), 1)
        self.assertTrue(flags[-1])


class VocabAgreementTests(unittest.TestCase):
    def test_the_scratch_model_is_sized_for_the_causal_vocab_not_the_masked_one(self):
        backbone = resolve("scratch")
        adapter = backbone.build_tokenizer("scratch")
        model = backbone.load_training_model("scratch", hidden_size=32, num_layers=1,
                                             num_attention_heads=4)
        self.assertEqual(adapter.vocab_size, len(CausalTokenizer().vocab))
        self.assertEqual(model.config.vocab_size, adapter.vocab_size)

    def test_eos_is_inside_the_model_vocabulary(self):
        # Sizing to the masked vocab (25) would put <eos> (26) out of range and silently remove the
        # model's ability to end a loop at all.
        backbone = resolve("scratch")
        adapter = backbone.build_tokenizer("scratch")
        model = backbone.load_training_model("scratch", hidden_size=32, num_layers=1,
                                             num_attention_heads=4)
        self.assertLess(adapter.eos_id, model.config.vocab_size)

    def test_the_scratch_adapter_uses_relative_positions(self):
        # Rotary; Findings §8 is why absolute positions are the wrong default for varying prompts.
        self.assertEqual(resolve("scratch").position_type, "relative")
        self.assertEqual(resolve("scratch").build_tokenizer("scratch").position_type, "relative")


class SentinelTests(unittest.TestCase):
    def test_a_plain_lm_vocabulary_is_reported_as_missing_every_sentinel(self):
        class FakeTok:
            def get_vocab(self):
                return {"A": 0, "C": 1, "<|endoftext|>": 2}
        self.assertEqual(missing_sentinels(FakeTok()), list(REQUIRED_SENTINELS))

    def test_a_vocabulary_that_already_has_them_needs_nothing_added(self):
        class FakeTok:
            def get_vocab(self):
                return {t: i for i, t in enumerate(REQUIRED_SENTINELS)}
        self.assertEqual(missing_sentinels(FakeTok()), [])

    def test_the_causal_tokenizer_supplies_all_three_sentinels(self):
        vocab = CausalTokenizer().tok_to_id
        for token in REQUIRED_SENTINELS:
            self.assertIn(token, vocab)

class FakeHFTokenizer:
    def __init__(self, existing_eos=False):
        self.vocab = {"A": 0, "C": 1}
        self.eos_token = None
        self.eos_token_id = None
        self.pad_token_id = None
        if existing_eos:
            self.vocab["</s>"] = len(self.vocab)
            self.eos_token = "</s>"
            self.eos_token_id = self.vocab["</s>"]

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def add_special_tokens(self, mapping):
        added = 0
        if "eos_token" in mapping:
            token = mapping["eos_token"]
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
            self.eos_token = token
            self.eos_token_id = self.vocab[token]
        for token in mapping.get("additional_special_tokens", []):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    @property
    def pad_token(self):
        return None if self.pad_token_id is None else self.eos_token

    @pad_token.setter
    def pad_token(self, token):
        self.pad_token_id = self.vocab[token]


class HFTokenizerPreparationTests(unittest.TestCase):
    def test_missing_eos_is_registered_as_the_actual_terminator(self):
        tokenizer = FakeHFTokenizer()
        added = _prepare_hf_tokenizer(tokenizer)

        self.assertEqual(added, ["<eos>", "<mask>", "<sep>"])
        self.assertEqual(tokenizer.eos_token_id, tokenizer.vocab["<eos>"])
        self.assertEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)

    def test_an_existing_semantic_eos_is_not_replaced_by_a_literal_token(self):
        tokenizer = FakeHFTokenizer(existing_eos=True)
        added = _prepare_hf_tokenizer(tokenizer)

        self.assertEqual(added, ["<mask>", "<sep>"])
        self.assertNotIn("<eos>", tokenizer.vocab)
        self.assertEqual(tokenizer.eos_token, "</s>")


# ====================================================================================================
# from test_ar_evaluation.py
# The causal evaluation strategy: the four things the shared loop cannot do for a causal model.
#
# The index-resolution test exists because that bug was real and silent-ish: `framework_for_dataset`
# already unwraps a Subset, so resolving the Subset index *before* calling it maps the index twice.
# On a small eval subset it raises; on a large one it would quietly score the wrong framework.
# ====================================================================================================

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


# ====================================================================================================
# from test_ar_model.py
# The causal decoder: the properties that make autoregressive training meaningful.
#
# Causality is the one that matters most. If position t can attend to t+1 the model sees the answer it
# is meant to predict, training collapses into copying, and every downstream number is meaningless
# while looking entirely healthy. It gets tested on both attention paths, because SDPA cannot combine
# `is_causal=True` with an explicit mask — padded batches take a different branch, and that branch is
# the one training actually uses.
# ====================================================================================================

def _model(**kw):
    torch.manual_seed(0)
    return build_model(vocab_size=27, hidden_size=64, num_layers=2, num_attention_heads=4,
                       **kw).eval()


class CausalityTests(unittest.TestCase):
    def test_a_later_token_cannot_change_an_earlier_prediction(self):
        model = _model()
        ids = torch.randint(5, 25, (1, 12))
        with torch.no_grad():
            base = model(ids)["logits"]
            for t in (4, 7, 11):
                alt = ids.clone()
                alt[0, t] = (alt[0, t] + 3) % 20 + 5
                out = model(alt)["logits"]
                self.assertTrue(torch.allclose(base[0, :t], out[0, :t], atol=1e-6),
                                f"perturbing t={t} changed an earlier position")
                self.assertFalse(torch.allclose(base[0, t:], out[0, t:], atol=1e-6),
                                 f"perturbing t={t} changed nothing at or after it")

    def test_causality_also_holds_on_the_padded_branch(self):
        # SDPA takes a different code path when an explicit mask is present, and that is the branch
        # every real training batch uses.
        model = _model()
        ids = torch.randint(5, 25, (1, 12))
        mask = torch.ones_like(ids)
        mask[0, 9:] = 0
        with torch.no_grad():
            base = model(ids, attention_mask=mask)["logits"]
            alt = ids.clone()
            alt[0, 6] = (alt[0, 6] + 5) % 20 + 5
            out = model(alt, attention_mask=mask)["logits"]
        self.assertTrue(torch.allclose(base[0, :6], out[0, :6], atol=1e-6))

    def test_padded_positions_cannot_influence_real_ones(self):
        model = _model()
        ids = torch.randint(5, 25, (1, 12))
        mask = torch.ones_like(ids)
        mask[0, 9:] = 0
        with torch.no_grad():
            base = model(ids, attention_mask=mask)["logits"]
            alt = ids.clone()
            alt[0, 10] = (alt[0, 10] + 7) % 20 + 5       # inside the padded region
            out = model(alt, attention_mask=mask)["logits"]
        self.assertTrue(torch.allclose(base[0, :9], out[0, :9], atol=1e-6))


class LossWiringTests(unittest.TestCase):
    def test_initial_loss_matches_a_uniform_guess_over_the_vocab(self):
        # ln(27) ~ 3.296. A wildly different value means the shift or the ignore_index is wrong.
        model = _model()
        ids = torch.randint(5, 25, (2, 16))
        labels = ids.clone()
        with torch.no_grad():
            loss = model(ids, labels=labels)["loss"].item()
        self.assertAlmostEqual(loss, torch.log(torch.tensor(27.0)).item(), delta=0.25)

    def test_the_next_token_shift_drops_no_supervision(self):
        # Positions 0-1 are the supplied prompt; positions 2-4 are the answer.
        # Compare with its three next-token log probabilities, excluding prompt supervision.
        model = _model()
        ids = torch.tensor([[5, 6, 7, 8, 9]])
        labels = torch.tensor([[-100, -100, 7, 8, 9]])
        with torch.no_grad():
            output = model(ids, labels=labels)
        log_probs = torch.log_softmax(output["logits"][0], dim=-1)
        expected = -(log_probs[1, 7] + log_probs[2, 8] + log_probs[3, 9]) / 3
        torch.testing.assert_close(output["loss"], expected)

    def test_a_fully_ignored_batch_produces_no_finite_loss_demand(self):
        model = _model()
        ids = torch.randint(5, 25, (1, 8))
        labels = torch.full_like(ids, -100)
        loss = model(ids, labels=labels)["loss"]
        self.assertTrue(torch.isnan(loss))   # CrossEntropyLoss over zero targets; caller must guard


class ArchitectureTests(unittest.TestCase):
    def test_heads_must_divide_the_width(self):
        with self.assertRaises(ValueError):
            build_model(hidden_size=100, num_attention_heads=8)

    def test_ffn_defaults_to_four_times_width(self):
        # Same rule as the masked model, so an architecture sweep is comparable across methods.
        self.assertEqual(build_model(hidden_size=96, num_attention_heads=8).config.intermediate_size,
                         4 * 96)

    def test_rotary_positions_mean_no_learned_position_table(self):
        # Findings §8: learned absolute positions are why a prepended partner chain shifts every
        # position and locks a checkpoint to its context regime. Causal prompts vary in length by
        # construction, so there must be no absolute table to shift.
        model = build_model()
        self.assertFalse(any("pos_embedding" in name for name, _ in model.named_parameters()))
        self.assertIsInstance(model, CausalAntibodyModel)


# ====================================================================================================
# from test_ar_sampling.py
# Causal decoding: the properties that make a sampled loop trustworthy.
#
# Uses a stub model returning controlled logits, so each behaviour is asserted exactly rather than
# inferred from a real model's noise.
# ====================================================================================================

class ArSamplingStubModel:
    """Emits a fixed token sequence, then <eos> forever. `forced` is a list of token ids."""

    def __init__(self, vocab_size, forced, eos_id):
        self.vocab_size, self.forced, self.eos_id = vocab_size, list(forced), eos_id
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, **_):
        batch = input_ids.shape[0]
        token = self.forced[self.calls] if self.calls < len(self.forced) else self.eos_id
        self.calls += 1
        logits = torch.full((batch, input_ids.shape[1], self.vocab_size), -10.0)
        logits[:, -1, token] = 10.0
        return {"logits": logits}


class NeverStops:
    """Always predicts the same residue — used to exercise the max_new_tokens cutoff."""

    def __init__(self, vocab_size, token):
        self.vocab_size, self.token = vocab_size, token

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, **_):
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.vocab_size), -10.0)
        logits[:, -1, self.token] = 10.0
        return {"logits": logits}


class LeftPadTests(unittest.TestCase):
    def test_every_rows_last_token_lands_at_index_minus_one(self):
        # Right padding would put the newest token at a different index per row, so a single
        # logits[:, -1] read would be wrong for all but the longest prompt.
        ids, mask = _left_pad([[1, 2, 3], [4, 5], [6]], pad_id=0, device="cpu")
        self.assertEqual(ids.shape, (3, 3))
        self.assertEqual(ids[:, -1].tolist(), [3, 5, 6])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [0, 1, 1], [0, 0, 1]])


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.tok = CausalTokenizer()
        self.adapter = causal_adapter(self.tok)
        self.aa = [self.tok.tok_to_id[c] for c in "ACDE"]

    def test_the_loop_is_the_tokens_before_eos(self):
        model = ArSamplingStubModel(len(self.tok.vocab), self.aa, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2, 3], device="cpu", temperature=0,
                           max_new_tokens=20)
        self.assertEqual(out[0]["loop"], "ACDE")
        self.assertEqual(out[0]["length"], 4)
        self.assertTrue(out[0]["complete"])

    def test_default_budget_can_finish_the_full_reserved_loop(self):
        residue = self.tok.tok_to_id["A"]
        model = ArSamplingStubModel(
            len(self.tok.vocab), [residue] * LOOP_LENGTH_RESERVE, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", temperature=0)
        self.assertTrue(out[0]["complete"])
        self.assertEqual(out[0]["length"], LOOP_LENGTH_RESERVE)

    def test_greedy_returns_exactly_one_candidate(self):
        # Matches the masked sampler's convention so --temperature 0 behaves the same either side.
        model = ArSamplingStubModel(len(self.tok.vocab), self.aa, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", num_samples=8,
                           temperature=0, max_new_tokens=20)
        self.assertEqual(len(out), 1)

    def test_sampling_returns_the_requested_number_of_candidates(self):
        model = ArSamplingStubModel(len(self.tok.vocab), self.aa, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", num_samples=5,
                           temperature=1.0, max_new_tokens=20,
                           generator=torch.Generator().manual_seed(0))
        self.assertEqual(len(out), 5)

    def test_a_draw_that_never_emits_eos_is_marked_incomplete(self):
        # A truncated loop is not a length the model chose, so it must not pass as a short loop.
        model = NeverStops(len(self.tok.vocab), self.tok.tok_to_id["A"])
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", temperature=0,
                           max_new_tokens=6)
        self.assertFalse(out[0]["complete"])
        self.assertEqual(out[0]["length"], 6)

    def test_special_tokens_cannot_appear_inside_a_loop(self):
        # The model is pushed hard toward <mask>; banning must override it, leaving only residues.
        model = ArSamplingStubModel(len(self.tok.vocab), [self.tok.mask_id] * 3, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", temperature=0,
                           max_new_tokens=10)
        residues = set("ACDEFGHIKLMNPQRSTVWY")
        self.assertTrue(set(out[0]["loop"]) <= residues, out[0]["loop"])

    def test_eos_stays_reachable_despite_the_ban(self):
        # Banning every special would make termination impossible and every draw incomplete.
        model = ArSamplingStubModel(len(self.tok.vocab), self.aa[:2], self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", temperature=0,
                           max_new_tokens=10)
        self.assertTrue(out[0]["complete"])
        self.assertEqual(out[0]["length"], 2)


# ====================================================================================================
# from test_ar_training.py
# The causal training pipeline: split selection, and the metric's alignment.
#
# The alignment test is the one that matters. `compute_metrics` receives predictions and
# labels UNSHIFTED and must redo the next-token shift itself; an off-by-one there reports a different
# quantity than the masked side's `compute_metrics` while looking perfectly healthy, which would make
# every AR-vs-MLM comparison wrong in a way no other test would catch.
# ====================================================================================================

class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


AR_TRAINING_CHAIN = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
         "fwr3": "FFFF", "cdr3": "GGGGGG", "fwr4": "HHHH"}


def _ar_training_data(splits):
    records = [{"heavy": dict(AR_TRAINING_CHAIN, full_aa="x"), "light": dict(AR_TRAINING_CHAIN, full_aa="x"), "split": s}
               for s in splits]
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(records, handle)
    handle.close()
    return handle.name


# labels as the model saw them: 3 prompt positions, a 6-residue loop (10..15), then <eos>=26.
LABELS = np.array([[-100, -100, -100, 10, 11, 12, 13, 14, 15, 26]])
PERFECT = np.array([[0, 0, 10, 11, 12, 13, 14, 15, 26, 0]])   # preds[t] == labels[t+1]


class MetricAlignmentTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        m = compute_metrics((PERFECT, LABELS))
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["core_accuracy"], 1.0)
        self.assertEqual(m["anchor_accuracy"], 1.0)

    def test_a_shift_that_is_off_by_one_scores_zero(self):
        # The failure this test exists for: silently measuring position t against residue t+1.
        misaligned = np.array([[0, 10, 11, 12, 13, 14, 15, 26, 0, 0]])
        self.assertEqual(compute_metrics((misaligned, LABELS))["accuracy"], 0.0)

    def test_core_and_anchor_split_at_the_same_boundary_as_the_masked_side(self):
        # Loop length 6, trim 2 -> core is the middle two residues.
        broken = PERFECT.copy()
        broken[0, 4] = 99
        broken[0, 5] = 99
        m = compute_metrics((broken, LABELS))
        self.assertEqual(m["core_accuracy"], 0.0)
        self.assertEqual(m["anchor_accuracy"], 1.0)

    def test_eos_is_excluded_from_recovery(self):
        # Recovery must measure residue prediction, comparable with a masked model that has no eos.
        eos_wrong = PERFECT.copy()
        eos_wrong[0, 8] = 99
        self.assertEqual(compute_metrics((eos_wrong, LABELS))["accuracy"], 1.0)

    def test_logits_are_accepted_as_well_as_argmaxed_ids(self):
        logits = np.zeros((1, 10, 30), dtype=np.float32)
        for t, token in enumerate(PERFECT[0]):
            logits[0, t, token] = 1.0
        self.assertEqual(compute_metrics((logits, LABELS))["accuracy"], 1.0)

    def test_a_row_with_no_supervision_is_skipped_not_counted_as_zero(self):
        labels = np.array([[-100] * 10])
        self.assertEqual(compute_metrics((PERFECT, labels))["accuracy"], 0.0)


class SplitTests(unittest.TestCase):
    """The causal pipeline must inherit the guard, not reimplement it."""

    def _datasets(self, splits, eval_split):
        path = _ar_training_data(splits)
        backbone = resolve("scratch")
        model_args = Args(model_type="scratch")
        data_args = Args(data_path=path, max_len=128, target_region="h3",
                         include_partner_chain=False, include_antigen=False,
                         eval_split=eval_split, context_dropout=0.0, test_size=0.1)
        try:
            return get_datasets(model_args, data_args, Args(seed=42), backbone)
        finally:
            os.unlink(path)

    def test_an_unset_eval_split_is_refused_when_a_test_partition_exists(self):
        with self.assertRaises(ValueError) as ctx:
            self._datasets(["train", "train", "validation", "test"], "")
        self.assertIn("test", str(ctx.exception))

    def test_validation_is_selected_without_touching_test(self):
        train, evaluation, _ = self._datasets(
            ["train", "train", "validation", "validation", "test"], "validation")
        self.assertEqual((len(train), len(evaluation)), (2, 2))

    def test_the_guard_is_the_shared_one(self):
        # If this ever diverges from the masked side, one method's results can be contaminated
        # without the other's tests noticing.
        with self.assertRaises(ValueError):
            select_split_indices([{"split": "train"}, {"split": "test"}], "",
                                 data_path="x.json", seed=0)


class NoMaskingViewTests(unittest.TestCase):
    def test_train_and_eval_share_records_with_separate_context_views(self):
        # Context settings must be isolated even though parsed records and tokenizer are shared.
        path = _ar_training_data(["train", "train", "validation"])
        backbone = resolve("scratch")
        train, evaluation, _ = get_datasets(
            Args(model_type="scratch"),
            Args(data_path=path, max_len=128, target_region="h3", include_partner_chain=False,
                 include_antigen=False, eval_split="validation", context_dropout=0.0, test_size=0.1),
            Args(seed=42), backbone)
        self.assertIsNot(train.dataset, evaluation.dataset)
        self.assertIs(train.dataset.data, evaluation.dataset.data)
        self.assertIs(train.dataset.tokenizer, evaluation.dataset.tokenizer)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
