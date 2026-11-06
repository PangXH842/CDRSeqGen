"""Wiring tests for evaluation/generation hooks that no earlier test executed.

A line trace of the suite on 2026-11-06 showed these never ran: the S2S strategy and case hook, MLM
self-scoring and both judge-prompt builders, the shared case record with a native truth and scores,
and the compatibility checker's blocking branches. test_ar_evaluation records why that matters:
`score_with_judge` stayed untested until a NameError reached a GPU run. Stub models keep every
expectation exact (flat logits give a known pseudo-perplexity), so nothing here needs trained weights.
"""
import io
import json
import os
import re
import shlex
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import evaluate
import generate
import train
from cdrseqgen.artifacts import GENERATION_EVAL_SCHEMA
from cdrseqgen.datasets.dataset_common import unwrap_subsets
from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, MaskedInfillingDataset, TokenizerAdapter
from cdrseqgen.datasets.dataset_s2s import (
    S2S_DECODE_STEPS, Seq2SeqInfillingDataset, build_encoder_input, s2s_adapter,
)
from cdrseqgen.eval.generative import build_masked_judge_prompt
from cdrseqgen.methods.common import evaluation as shared_evaluation
from cdrseqgen.methods.common import generation as shared_generation
from cdrseqgen.methods.common.generation import check_compatibility, load_test_cases, residue_warnings
from cdrseqgen.methods.mlm.evaluation import run_generative_eval as mlm_evaluate
from cdrseqgen.methods.mlm.generation import generate_case as mlm_generate_case
from cdrseqgen.methods.mlm.length import (
    FixedNativeStrategy, VariableLengthStrategy, masked_view, resolve_generation_length_policy,
)
from cdrseqgen.methods.s2s.evaluation import Seq2SeqStrategy, run_generative_eval as s2s_evaluate
from cdrseqgen.methods.s2s.generation import generate_case as s2s_generate_case

CHAIN = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
         "fwr3": "FFFF", "cdr3": "GGGGG", "fwr4": "HHHH"}
CPU = torch.device("cpu")


def corpus(test, n=6):
    """A processed-corpus JSON whose records differ in flank length, so a mixed-up index shows."""
    records = []
    for index in range(n):
        chain = dict(CHAIN, fwr1="A" * (index + 1), full_aa="x")
        records.append({"heavy": chain, "light": dict(chain), "split": "train"})
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(records, handle)
    handle.close()
    test.addCleanup(os.unlink, handle.name)
    return handle.name


class ScriptedSeq2Seq:
    """Emits the scripted token at each decoder length; every other logit is a flat -20.

    Scoring "AC" therefore costs log(27) at the two unscripted positions and ~0 at the final EOS
    position: perplexity 27 ** (2 / 3) = 9 exactly.
    """

    def __init__(self, script, vocab_size):
        self.script, self.vocab_size = list(script), vocab_size

    def eval(self):
        return self

    def encode(self, input_ids, attention_mask=None):
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)

    def __call__(self, *, decoder_input_ids, encoder_hidden_states=None, attention_mask=None, **_):
        rows, steps = decoder_input_ids.shape
        logits = torch.full((rows, steps, self.vocab_size), -20.0)
        logits[:, -1, self.script[min(steps - 1, len(self.script) - 1)]] = 20.0
        return {"logits": logits}


class FlatMasked:
    """A masked LM with uniform logits: every non-special residue is equally likely (ppl = 20)."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def eval(self):
        return self

    def __call__(self, input_ids=None, **_):
        return {"logits": torch.zeros(*input_ids.shape, self.vocab_size)}


class S2SEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tok = s2s_adapter()
        self.path = corpus(self)
        self.full = Seq2SeqInfillingDataset(self.path, self.tok, target_region="h3")
        self.subset = torch.utils.data.Subset(self.full, [2, 3, 4])
        self.model = ScriptedSeq2Seq(self.tok.encode("AC") + [self.tok.eos_id], self.tok.vocab_size)

    def strategy(self, model=None, **overrides):
        options = dict(model=model or self.model, tokenizer=self.tok, device=CPU,
                       generator=torch.Generator().manual_seed(0), eval_dataset=self.subset,
                       judge_dataset=self.full, judge_tokenizer=self.tok, num_samples=3,
                       temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=8)
        options.update(overrides)
        return Seq2SeqStrategy(**options)

    def test_hooks_use_the_training_encoder_input_and_the_s2s_sampler_and_scorer(self):
        strategy = self.strategy()
        framework = strategy.framework(0)
        self.assertEqual(framework, self.full.framework_parts(2))
        context = strategy.context(framework)
        self.assertEqual(context, build_encoder_input(self.tok, framework)[0])
        self.assertEqual(strategy.sample(context),
                         [{"loop": "AC", "length": 2, "complete": True}] * 3)
        (score,) = strategy.score([(context, self.tok.encode("AC"))])
        self.assertAlmostEqual(score, 9.0, places=3)

    def test_draw_counts_every_attempt_against_the_native_length(self):
        strategy = self.strategy()
        drawn = strategy.draw(0)
        self.assertEqual((drawn.native, drawn.seqs), ("GGGGG", ["AC"] * 3))
        self.assertEqual((strategy.attempts, strategy.complete_nonempty, strategy.incomplete), (3, 3, 0))
        self.assertEqual((strategy.length_matches, strategy.length_absolute_error_capped), (0, 9))

    def test_attempts_that_never_terminate_stay_in_the_counts_without_a_draw(self):
        strategy = self.strategy(model=ScriptedSeq2Seq(self.tok.encode("A"), self.tok.vocab_size),
                                 max_new_tokens=5)
        self.assertIsNone(strategy.draw(0))
        self.assertEqual((strategy.attempts, strategy.complete_nonempty, strategy.incomplete), (3, 0, 3))
        # Five unterminated residues sit at the five-residue native's length, yet a failed draw is
        # never free: each costs at least one residue and none counts as a length match.
        self.assertEqual((strategy.length_matches, strategy.length_absolute_error_capped), (0, 3))

    def test_run_generative_eval_returns_the_shared_payload_with_self_scores(self):
        payload = s2s_evaluate(self.model, self.tok, self.subset, device=CPU, num_examples=-1,
                               num_samples=2, temperature=1.0, top_k=0, top_p=1.0, seed=0)
        self.assertEqual((payload["method"], payload["decoding"]["mode"]), ("s2s", "seq2seq"))
        self.assertEqual(payload["decoding"]["max_new_tokens"], S2S_DECODE_STEPS)
        self.assertEqual((payload["n_selected"], payload["n_generated"]), (3, 3))
        validity = payload["generation_validity"]
        self.assertEqual((validity["attempts"], validity["valid_generation_rate"]), (6, 1.0))
        self.assertEqual([r["samples"] for r in payload["records"]], [["AC", "AC"]] * 3)
        self.assertEqual({r["eval_index"] for r in payload["records"]}, {0, 1, 2})
        for record in payload["records"]:
            for candidate in record["candidates"]:
                self.assertAlmostEqual(candidate["self_perplexity"], 9.0, places=3)
        self.assertAlmostEqual(payload["summary"]["self_perplexity_mean"], 9.0, places=3)

    def test_skip_perplexity_leaves_no_self_scores(self):
        payload = s2s_evaluate(self.model, self.tok, self.subset, device=CPU, num_examples=-1,
                               num_samples=2, temperature=1.0, top_k=0, top_p=1.0, seed=0,
                               skip_perplexity=True)
        self.assertIsNone(payload["summary"]["self_perplexity_mean"])
        self.assertTrue(all(c["self_perplexity"] is None
                            for r in payload["records"] for c in r["candidates"]))

    def test_the_masked_lm_judge_scores_s2s_candidates_at_their_own_length(self):
        judge_tok = TokenizerAdapter.from_custom(AntibodyTokenizer())
        judge_data = MaskedInfillingDataset(self.path, judge_tok, target_region="h3")
        payload = s2s_evaluate(
            self.model, self.tok, self.subset, device=CPU, num_examples=-1, num_samples=2,
            temperature=1.0, top_k=0, top_p=1.0, seed=0, skip_perplexity=True,
            judge_model=FlatMasked(judge_tok.vocab_size), judge_tokenizer=judge_tok,
            judge_dataset=judge_data, judge_name="flat")
        self.assertEqual(payload["judge"], "flat")
        self.assertAlmostEqual(payload["summary"]["judge_perplexity_mean"], 20.0, places=3)
        self.assertEqual(payload["summary"]["judge_coverage"], 1.0)


class SharedCaseRecordTests(unittest.TestCase):
    """`eos_case_record` through S2S: the native-truth and score branches nothing else reached."""

    CASE = {"name": "c", "pt1": "ACDEFGHIK", "pt2": "LMNPQRST"}

    def setUp(self):
        self.tok = s2s_adapter()
        self.model = ScriptedSeq2Seq(self.tok.encode("AC") + [self.tok.eos_id], self.tok.vocab_size)
        self.args = dict(max_len=64, partner_chain=None, antigen=None, num_samples=2, temperature=1.0,
                         top_k=0, top_p=1.0, max_new_tokens=8, skip_perplexity=False, seed=0)

    def record(self, case=None, model=None, **overrides):
        callback = Mock()
        record = s2s_generate_case(
            model or self.model, self.tok, CPU, SimpleNamespace(**{**self.args, **overrides}),
            torch.Generator().manual_seed(0), {**self.CASE, **(case or {})}, None, callback)
        return record, callback

    def test_matching_truth_scores_recovery_and_perplexity(self):
        record, callback = self.record({"target_truth": " ac "})
        self.assertEqual(record["native"], "AC")
        self.assertEqual(len(record["samples"]), 2)
        for sample in record["samples"]:
            self.assertEqual((sample["seq"], sample["length"]), ("AC", 2))
            self.assertEqual((sample["length_match"], sample["recovery"]), (True, 1.0))
            self.assertAlmostEqual(sample["perplexity"], 9.0, places=3)
        self.assertEqual(record["metrics"]["aar_mean"], 1.0)
        self.assertEqual(record["target_length"], 2)
        self.assertEqual(record["decoding"]["mode"], "seq2seq")
        self.assertEqual(record["generation_validity"],
                         {"attempts": 2, "complete_nonempty": 2, "incomplete": 0, "complete_empty": 0})
        callback.assert_called_once()
        self.assertEqual(callback.call_args.args[0], "c")

    def test_a_different_length_truth_has_no_positional_recovery(self):
        record, _ = self.record({"target_truth": "GGGGG"})
        for sample in record["samples"]:
            self.assertFalse(sample["length_match"])
            self.assertIsNone(sample["recovery"])
        self.assertNotIn("aar_mean", record["metrics"])

    def test_without_truth_candidates_carry_no_reference_fields(self):
        record, _ = self.record()
        self.assertIsNone(record["native"])
        for sample in record["samples"]:
            self.assertNotIn("length_match", sample)
            self.assertNotIn("recovery", sample)

    def test_skip_perplexity_omits_the_score_key(self):
        record, _ = self.record(skip_perplexity=True)
        self.assertTrue(all("perplexity" not in sample for sample in record["samples"]))

    def test_a_case_where_every_attempt_fails_is_still_a_complete_record(self):
        never_ends = ScriptedSeq2Seq(self.tok.encode("A"), self.tok.vocab_size)
        record, callback = self.record(model=never_ends, max_new_tokens=4)
        self.assertEqual(record["samples"], [])
        self.assertEqual(record["generated_lengths"], [])
        self.assertIsNone(record["target_length"])
        self.assertEqual(record["generation_validity"],
                         {"attempts": 2, "complete_nonempty": 0, "incomplete": 2, "complete_empty": 0})
        self.assertEqual(callback.call_args.args[3], "—")


class MlmScoringAndJudgeMappingTests(unittest.TestCase):
    def setUp(self):
        self.tok = TokenizerAdapter.from_custom(AntibodyTokenizer())
        self.path = corpus(self)
        self.residues = self.tok.vocab_size - len(set(self.tok.special_ids))

    def strategies(self, partner=False):
        full = MaskedInfillingDataset(self.path, self.tok, max_len=64, target_region="h3",
                                      include_partner_chain=partner)
        common = dict(model=FlatMasked(self.tok.vocab_size), tokenizer=self.tok, device=CPU,
                      generator=torch.Generator().manual_seed(0),
                      eval_dataset=torch.utils.data.Subset(full, [2, 3, 4]), judge_dataset=full,
                      judge_tokenizer=self.tok, decoding="independent", num_samples=2,
                      temperature=1.0, top_k=0, top_p=1.0)
        fixed = FixedNativeStrategy(**common)
        variable = VariableLengthStrategy(
            length_predictor=None, length_policy=resolve_generation_length_policy("inferred"),
            length_temperature=1.0, **common)
        return full, fixed, variable

    def test_self_scores_are_pseudo_perplexities_in_candidate_order(self):
        _, fixed, _ = self.strategies()
        drawn = fixed.draw(0)
        scores = fixed.self_scores(drawn)
        self.assertEqual(len(scores), len(drawn.seqs))
        for score in scores:
            self.assertAlmostEqual(score, self.residues, places=3)

    def test_fixed_native_judge_prompt_follows_the_eval_index_to_its_corpus_record(self):
        full, fixed, _ = self.strategies()
        # Revisiting index 0 last proves the one-entry cache does not hand back a neighbour's prompt.
        for eval_index, corpus_index in ((0, 2), (1, 3), (2, 4), (0, 2)):
            framework = full.framework_parts(corpus_index)
            prompt = fixed.judge_inputs(eval_index, "A")
            self.assertEqual(prompt, masked_view(full[corpus_index])[:2])
            self.assertEqual(prompt, build_masked_judge_prompt(self.tok, framework,
                                                               len(framework["target"])))

    def test_variable_length_judge_prompt_uses_the_candidate_length_and_the_shared_builder(self):
        for partner in (False, True):
            full, _, variable = self.strategies(partner=partner)
            for eval_index, corpus_index in ((0, 2), (1, 3), (2, 4)):
                for length in (3, 5, 8):
                    with self.subTest(partner=partner, eval_index=eval_index, length=length):
                        input_ids, positions = variable.judge_inputs(eval_index, "A" * length)
                        self.assertEqual(len(positions), length)
                        self.assertEqual((input_ids, positions), build_masked_judge_prompt(
                            self.tok, full.framework_parts(corpus_index), length))


class MlmGenerateCaseTests(unittest.TestCase):
    def setUp(self):
        self.tok = TokenizerAdapter.from_custom(AntibodyTokenizer())
        self.residues = self.tok.vocab_size - len(set(self.tok.special_ids))
        self.args = dict(partner_chain=None, antigen=None, max_len=64, decoding="independent",
                         num_samples=3, temperature=1.0, top_k=0, top_p=1.0, length_temperature=1.0,
                         skip_perplexity=False, seed=0, min_length=4, max_length=4,
                         target_length=None, use_native_length=False)
        self.case = {"name": "c", "pt1": "ACDEFGHIK", "pt2": "LMNPQRST", "target_truth": "WYVW"}

    def record(self, **overrides):
        return mlm_generate_case(
            FlatMasked(self.tok.vocab_size), self.tok, CPU, SimpleNamespace(**{**self.args, **overrides}),
            torch.Generator().manual_seed(0), self.case, None, Mock())

    def test_each_candidate_is_self_scored_and_summarised(self):
        record = self.record()
        self.assertEqual(len(record["samples"]), 3)
        for sample in record["samples"]:
            self.assertAlmostEqual(sample["pseudo_perplexity"], self.residues, places=3)
        for key in ("pseudo_perplexity_mean", "pseudo_perplexity_min", "pseudo_perplexity_max"):
            self.assertAlmostEqual(record["metrics"][key], self.residues, places=3)
        self.assertEqual((record["native"], record["length_policy"]["mode"]), ("WYVW", "fixed"))

    def test_skip_perplexity_leaves_neither_scores_nor_summary(self):
        record = self.record(skip_perplexity=True)
        self.assertTrue(all("pseudo_perplexity" not in sample for sample in record["samples"]))
        self.assertNotIn("pseudo_perplexity_mean", record["metrics"])


class CompatibilityContractTests(unittest.TestCase):
    """The checks that stop a checkpoint meeting inputs it was not trained for."""

    TRAINED = {"source": "run_config", "target_region": "h3"}

    def check(self, config=None, *, pt1_length=104, partner_chain=None, target_loop=None, case_partner=None):
        args = SimpleNamespace(target_loop=target_loop, partner_chain=partner_chain, antigen=None)
        case = {"name": "case", "pt1": "A" * pt1_length, "pt2": "C" * 10}
        if case_partner:
            case["partner_chain"] = case_partner
        return check_compatibility({**self.TRAINED, **(config or {})}, [case], args)

    def test_a_requested_loop_that_differs_from_the_trained_one_only_warns(self):
        problems, warnings = self.check(target_loop="h1")
        self.assertEqual(problems, [])
        self.assertTrue(any(w.startswith("LOOP:") for w in warnings))

    def test_a_gap_positioned_like_another_cdr_warns_with_the_inferred_number(self):
        problems, warnings = self.check(pt1_length=26)
        self.assertEqual(problems, [])
        shape = [w for w in warnings if w.startswith("SHAPE:")]
        self.assertEqual(len(shape), 1)
        self.assertIn("looks like CDR1", shape[0])
        self.assertEqual(self.check(pt1_length=104)[1], [])

    def test_a_checkpoint_trained_with_partner_context_blocks_cases_without_it(self):
        problems, _ = self.check({"include_partner_chain": True})
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("CONTEXT MISSING"))
        self.assertEqual(self.check({"include_partner_chain": True}, partner_chain="EVQL")[0], [])
        self.assertEqual(self.check({"include_partner_chain": True}, case_partner="EVQL")[0], [])

    def test_a_checkpoint_trained_without_partner_context_blocks_cases_with_it(self):
        problems, _ = self.check({"include_partner_chain": False}, case_partner="EVQL")
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("CONTEXT UNEXPECTED"))

    def test_cli_assumptions_cannot_establish_a_context_problem(self):
        problems, _ = self.check({"source": "cli", "include_partner_chain": True})
        self.assertEqual(problems, [])


class RootScriptDocumentationTests(unittest.TestCase):
    """What a root script's docstring advertises must exist and parse: commands and functions."""

    def test_every_advertised_command_parses_with_its_own_parser(self):
        for module in (evaluate, generate):
            examples = [line.strip()[2:].partition("   #")[0].strip()
                        for line in module.__doc__.splitlines()
                        if line.strip().startswith(f"* python {module.__name__}.py")]
            self.assertGreaterEqual(len(examples), 4)
            for example in examples:
                with self.subTest(example=example):
                    argv = shlex.split(example)
                    self.assertEqual(argv[:2], ["python", f"{module.__name__}.py"])
                    module.parse_args(argv[2:])
                    # argparse also accepts any unambiguous prefix, so a stale spelling would parse.
                    known = module.build_parser()._option_string_actions
                    self.assertEqual([t for t in argv if t.startswith("--") and t not in known], [])

    def test_every_listed_function_exists(self):
        for module in (train, evaluate, generate):
            section = module.__doc__.split("Functions:")[1].split("Workflow:")[0]
            names = re.findall(r"^  \* (\w+)\s+--", section, re.M)
            self.assertGreaterEqual(len(names), 2)
            for name in names:
                with self.subTest(module=module.__name__, name=name):
                    self.assertTrue(callable(getattr(module, name, None)))

    def test_the_root_scripts_share_trains_docstring_sections(self):
        for module in (train, evaluate, generate):
            positions = [module.__doc__.find(heading) for heading in ("To run:", "Functions:", "Workflow:")]
            with self.subTest(module=module.__name__):
                self.assertTrue(all(p >= 0 for p in positions) and positions == sorted(positions))


class PipelineHarness:
    """Runs run_generation_pipeline with loading mocked and a real scratch tokenizer.

    Mix into a TestCase. Each call returns (results, refusal): refusal is the SystemExit a guard
    raised, else None. The last call's mocks and paths stay on self for inspection.
    """

    WITH_PARTNER = {"source": "run_config", "model_type": "scratch", "max_len": 64,
                    "include_partner_chain": True}
    NO_PARTNER = {"source": "run_config", "model_type": "scratch", "max_len": 64,
                  "include_partner_chain": False}

    def run_pipeline(self, *flags, cases=None, output="out.json", config=None, on_load=None,
                     case_record=None):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, True)
        argv = ["--model_dir", self.directory, "--pt1", "A" * 30, "--pt2", "C" * 10]
        self.output = None
        if output is not None:
            self.output = output if os.path.isabs(output) else os.path.join(self.directory, output)
            argv += ["--output", self.output]
        if cases is not None:
            path = os.path.join(self.directory, "cases.json")
            with open(path, "w") as handle:
                json.dump(cases, handle)
            argv += ["--test_file", path]
        self.args = generate.parse_args([*argv, *flags])
        self.backbone = Mock()
        self.backbone.build_tokenizer.return_value = TokenizerAdapter.from_custom(AntibodyTokenizer())
        if on_load is not None:
            self.backbone.load_pretrained_model.side_effect = lambda *_: (on_load(), Mock())[1]
        self.generate_case = Mock(return_value={"name": "x"} if case_record is None else case_record)
        method = SimpleNamespace(name="mlm", module=Mock(return_value=SimpleNamespace(
            resolve=Mock(return_value=self.backbone))))
        with patch.object(shared_generation, "resolve", return_value=method), \
                patch("builtins.print") as printed:
            try:
                results, refusal = shared_generation.run_generation_pipeline(
                    "mlm", self.args, dict(config or self.WITH_PARTNER),
                    prepare_generation=Mock(return_value=None),
                    generation_state=Mock(return_value=None),
                    generate_case=self.generate_case), None
            except SystemExit as exc:
                results, refusal = None, exc
        self.printed = printed
        return results, refusal

    def assert_nothing_loaded(self):
        self.backbone.build_tokenizer.assert_not_called()
        self.backbone.load_pretrained_model.assert_not_called()
        self.generate_case.assert_not_called()


class GenerationPipelineGuardTests(PipelineHarness, unittest.TestCase):
    """Compatibility problems stop the run before any weights load; --force records the override."""

    def test_a_missing_context_refuses_before_loading_or_writing_anything(self):
        with self.assertLogs(shared_generation.logger, "WARNING") as logs:
            results, refusal = self.run_pipeline()
        self.assertTrue(any("CONTEXT MISSING" in line for line in logs.output))
        self.assertIsNone(results)
        self.assertIn("compatibility problem", str(refusal))
        self.assert_nothing_loaded()
        self.assertFalse(os.path.exists(self.output))

    def test_force_runs_and_records_the_override_in_the_saved_output(self):
        with self.assertLogs(shared_generation.logger, "WARNING") as logs:
            results, refusal = self.run_pipeline("--force")
        self.assertTrue(any("--force given" in line for line in logs.output))
        self.assertIsNone(refusal)
        self.assertEqual(results, [{"name": "x"}])
        with open(self.output) as handle:
            saved = json.load(handle)
        compatibility = saved["model"]["compatibility"]
        self.assertTrue(compatibility["forced"])
        self.assertEqual(len(compatibility["problems"]), 1)
        self.assertIn("CONTEXT MISSING", compatibility["problems"][0])

    def test_supplying_the_context_needs_no_override(self):
        with self.assertNoLogs(shared_generation.logger, "WARNING"):
            results, refusal = self.run_pipeline("--partner_chain", "EVQLVESGG")
        self.assertIsNone(refusal)
        with open(self.output) as handle:
            compatibility = json.load(handle)["model"]["compatibility"]
        self.assertEqual((compatibility["problems"], compatibility["forced"]), ([], False))


class LoadTestCasesTests(unittest.TestCase):
    """Supplied cases are validated and their residue strings cleaned where they enter."""

    def load(self, content):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        handle.write(content if isinstance(content, str) else json.dumps(content))
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        self.path = handle.name
        return load_test_cases(SimpleNamespace(test_file=handle.name))

    def test_a_single_case_object_is_a_one_case_list(self):
        self.assertEqual(self.load({"name": "solo", "pt1": "AC", "pt2": "WY"}),
                         [{"name": "solo", "pt1": "AC", "pt2": "WY"}])

    def test_cases_keep_their_other_keys_and_order(self):
        supplied = [{"name": "a", "pt1": "AC", "pt2": "WY", "note": "kept", "source_id": 7},
                    {"pt1": "", "pt2": ""}]
        self.assertEqual(self.load(supplied), supplied)

    def test_residue_strings_are_stripped_and_uppercased_but_never_repaired(self):
        (case,) = self.load({"pt1": " evqL\n", "pt2": "wg-qg", "partner_chain": "dIq",
                             "antigen": "ab x", "target_truth": " ars "})
        # target_truth is not a model input; the methods already normalise it themselves.
        self.assertEqual(case, {"pt1": "EVQL", "pt2": "WG-QG", "partner_chain": "DIQ",
                                "antigen": "AB X", "target_truth": " ars "})

    def test_the_cli_case_is_cleaned_too(self):
        args = SimpleNamespace(test_file=None, pt1="ac ", pt2="\twy", target_truth=None,
                               target_length=None, min_length=3, max_length=3)
        (case,) = load_test_cases(args)
        self.assertEqual((case["name"], case["pt1"], case["pt2"], case["min_length"]),
                         ("CLI Test Case", "AC", "WY", 3))

    def test_malformed_inputs_raise_a_message_naming_the_problem(self):
        base = {"pt1": "A", "pt2": "C"}
        for label, content, expected in (
                ("a bare string", '"cases"', "got str"), ("a number", 3, "got int"),
                ("an empty list", [], "holds no cases"),
                ("a non-object case", [base, ["pt1", "pt2"]], "case 1 must be an object"),
                ("missing pt1", {"pt2": "C"}, "needs 'pt1'"),
                ("null pt2", {**base, "pt2": None}, "needs 'pt2'"),
                ("numeric pt1", {**base, "pt1": 5}, "needs 'pt1'"),
                ("numeric partner", {**base, "partner_chain": 5}, "'partner_chain'"),
                ("list antigen", {**base, "antigen": ["A"]}, "'antigen'"),
                ("numeric truth", {**base, "target_truth": 7}, "'target_truth'"),
                ("invalid JSON", "{not json", "is not valid JSON")):
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, expected) as caught:
                    self.load(content)
                self.assertIn(self.path, str(caught.exception))

    def test_a_named_case_is_named_in_its_error(self):
        with self.assertRaisesRegex(ValueError, "'mine' needs 'pt1'"):
            self.load({"name": "mine", "pt2": "C"})

    def test_every_shipped_showcase_file_loads(self):
        files = sorted((Path(__file__).resolve().parents[1] / "showcase").glob("*.json"))
        self.assertGreaterEqual(len(files), 5)
        for path in files:
            with self.subTest(file=path.name):
                cases = load_test_cases(SimpleNamespace(test_file=str(path)))
                self.assertTrue(cases and all(case["pt1"] and case["pt2"] for case in cases))


class ResidueWarningTests(unittest.TestCase):
    """Characters a tokenizer silently turns into <unk> are named before the weights load."""

    def setUp(self):
        self.tok = TokenizerAdapter.from_custom(AntibodyTokenizer())

    def warnings(self, cases, partner_chain=None, antigen=None):
        args = SimpleNamespace(partner_chain=partner_chain, antigen=antigen)
        return residue_warnings(self.tok, cases, args, model_type="scratch")

    def test_clean_sequences_have_no_warnings(self):
        self.assertEqual(self.warnings([{"name": "a", "pt1": "ACDEF", "pt2": "WY",
                                         "partner_chain": "EVQL"}]), [])

    def test_unencodable_characters_are_named_with_one_based_positions(self):
        (message,) = self.warnings([{"name": "gapped", "pt1": "AC-DE.FG1H", "pt2": "WY"}])
        for part in ("RESIDUES: gapped pt1", "3 character(s)", "'scratch' tokenizer",
                     "'-' at 3", "'.' at 6", "'1' at 9"):
            self.assertIn(part, message)

    def test_more_than_three_are_summarised(self):
        (message,) = self.warnings([{"name": "a", "pt1": "A-B-C-D", "pt2": "WY"}])
        self.assertIn("4 character(s)", message)
        self.assertIn("(+1 more)", message)

    def test_each_affected_sequence_gets_its_own_warning(self):
        messages = self.warnings([{"name": "a", "pt1": "A-C", "pt2": "W Y", "partner_chain": "EVQL"}])
        self.assertEqual([m.split()[1:3] for m in messages], [["a", "pt1"], ["a", "pt2"]])

    def test_an_antigen_chain_separator_is_exempt_but_other_fields_flag_x(self):
        self.assertEqual(self.warnings([{"name": "a", "pt1": "AC", "pt2": "WY", "antigen": "ACXDE"}]), [])
        (message,) = self.warnings([{"name": "a", "pt1": "ACX", "pt2": "WY"}])
        self.assertIn("'X' at 3", message)
        (message,) = self.warnings([{"name": "a", "pt1": "AC", "pt2": "WY", "antigen": "AC-XDE"}])
        self.assertIn("'-' at 3", message)
        self.assertNotIn("'X'", message)

    def test_cli_context_is_checked_once_for_all_cases(self):
        cases = [{"name": f"c{i}", "pt1": "AC", "pt2": "WY"} for i in range(3)]
        (message,) = self.warnings(cases, partner_chain="EV-L")
        self.assertIn("--partner_chain", message)

    def test_nothing_to_check_never_touches_the_tokenizer(self):
        args = SimpleNamespace(partner_chain=None, antigen=None)
        self.assertEqual(residue_warnings(Mock(), [{"name": "a", "pt1": "", "pt2": ""}], args,
                                          model_type="x"), [])

    def test_a_tokenizer_that_raises_for_a_character_counts_it_as_unencodable(self):
        tokenizer = Mock(special_ids=[0], encode=Mock(side_effect=KeyError("?")))
        args = SimpleNamespace(partner_chain=None, antigen=None)
        (message,) = residue_warnings(tokenizer, [{"name": "a", "pt1": "AC", "pt2": ""}], args,
                                      model_type="x")
        self.assertIn("2 character(s)", message)


class CaseHandlingTests(PipelineHarness, unittest.TestCase):
    """Malformed cases stop the run before any weights load; supplied residues are cleaned."""

    def test_a_malformed_case_file_exits_with_its_message(self):
        for content, expected in (({"pt2": "C"}, "needs 'pt1'"), ([], "holds no cases"),
                                  (3, "got int"),
                                  ([{"pt1": "A", "pt2": "C"}, "oops"], "case 1 must be an object")):
            with self.subTest(expected=expected):
                results, refusal = self.run_pipeline(cases=content, config=self.NO_PARTNER)
                self.assertIsNone(results)
                self.assertIn(expected, str(refusal))
                self.assert_nothing_loaded()

    def test_pasted_lowercase_flanks_and_context_reach_the_method_uppercased(self):
        config = dict(self.WITH_PARTNER, include_antigen=True)
        results, refusal = self.run_pipeline(
            "--partner_chain", " dIqm ", "--antigen", "kv", config=config,
            cases=[{"name": "p", "pt1": " evql ", "pt2": "wgqg"}])
        self.assertIsNone(refusal)
        sent = self.generate_case.call_args.args[5]
        self.assertEqual((sent["pt1"], sent["pt2"]), ("EVQL", "WGQG"))
        self.assertEqual((self.args.partner_chain, self.args.antigen), ("DIQM", "KV"))

    def test_unencodable_residues_warn_before_the_weights_load_and_are_recorded(self):
        seen = []
        with self.assertLogs(shared_generation.logger, "WARNING") as logs:
            results, refusal = self.run_pipeline(
                cases=[{"name": "gapped", "pt1": "AC-DE", "pt2": "WY"}], config=self.NO_PARTNER,
                on_load=lambda: seen.append(list(logs.output)))
        self.assertIsNone(refusal)
        self.assertEqual(len(seen), 1)
        self.assertTrue(any("RESIDUES: gapped pt1" in line for line in seen[0]))   # warned first
        with open(self.output) as handle:
            warnings = json.load(handle)["model"]["compatibility"]["warnings"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("RESIDUES: gapped pt1", warnings[0])


class OutputHandlingTests(PipelineHarness, unittest.TestCase):
    """A finished generation run is never lost to its output path."""

    def test_results_go_into_a_directory_that_does_not_exist_yet(self):
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base, True)
        target = os.path.join(base, "new", "deeper", "out.json")
        results, refusal = self.run_pipeline(output=target, config=self.NO_PARTNER)
        self.assertIsNone(refusal)
        with open(target) as handle:
            self.assertEqual(json.load(handle)["cases"], results)
        self.assertEqual(os.listdir(os.path.dirname(target)), ["out.json"])   # no temp leftovers

    def test_the_default_output_sits_beside_the_checkpoint(self):
        self.run_pipeline(output=None, config=self.NO_PARTNER)
        self.assertTrue(os.path.exists(os.path.join(self.directory, "generation_results.json")))

    def test_output_none_writes_nothing(self):
        results, _ = self.run_pipeline("--output", "none", output=None, config=self.NO_PARTNER)
        self.assertEqual(results, [{"name": "x"}])
        self.assertEqual(os.listdir(self.directory), [])

    def test_a_directory_as_the_output_exits_before_any_loading(self):
        target = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, target, True)
        results, refusal = self.run_pipeline(output=target, config=self.NO_PARTNER)
        self.assertIsNone(results)
        self.assertIn("is a directory", str(refusal))
        self.assert_nothing_loaded()

    def test_an_uncreatable_output_directory_exits_before_any_loading(self):
        blocker = tempfile.NamedTemporaryFile(delete=False)
        blocker.close()
        self.addCleanup(os.unlink, blocker.name)
        results, refusal = self.run_pipeline(output=os.path.join(blocker.name, "sub", "out.json"),
                                             config=self.NO_PARTNER)
        self.assertIsNone(results)
        self.assertIn("Cannot create the output directory", str(refusal))
        self.assert_nothing_loaded()

    def test_non_finite_values_are_saved_as_null(self):
        record = {"name": "x", "metrics": {"nan": float("nan"), "inf": float("inf"), "ok": 1.5}}
        self.run_pipeline(config=self.NO_PARTNER, case_record=record)
        with open(self.output) as handle:
            text = handle.read()
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(json.loads(text)["cases"][0]["metrics"], {"nan": None, "inf": None, "ok": 1.5})

    def test_a_write_failure_after_generation_still_returns_the_records(self):
        with patch.object(shared_generation, "atomic_write_json", side_effect=OSError("disk full")):
            results, refusal = self.run_pipeline(config=self.NO_PARTNER)
        self.assertIsNone(refusal)
        self.assertEqual(results, [{"name": "x"}])
        self.assertTrue(any("Could not write results" in str(call)
                            for call in self.printed.call_args_list))


class NestedSubsetTests(unittest.TestCase):
    """Every strategy maps an eval index through all Subset levels, not just the outer one."""

    def test_unwrap_subsets_resolves_every_level(self):
        full = list(range(100, 110))
        inner = torch.utils.data.Subset(full, [4, 1, 5, 7])
        nested = torch.utils.data.Subset(inner, [2, 0])
        self.assertEqual(unwrap_subsets(nested, 0), (full, 5))
        self.assertEqual(unwrap_subsets(nested, 1), (full, 4))
        self.assertEqual(unwrap_subsets(inner, 3), (full, 7))
        self.assertEqual(unwrap_subsets(full, 2), (full, 2))

    def test_mlm_judge_prompts_follow_a_nested_eval_subset(self):
        tok = TokenizerAdapter.from_custom(AntibodyTokenizer())
        path = corpus(self)
        full = MaskedInfillingDataset(path, tok, max_len=64, target_region="h3")
        nested = torch.utils.data.Subset(torch.utils.data.Subset(full, [4, 1, 5, 2]), [2, 0])
        common = dict(model=None, tokenizer=tok, device=CPU, generator=None, eval_dataset=nested,
                      judge_dataset=full, judge_tokenizer=tok, decoding="independent",
                      num_samples=1, temperature=0, top_k=0, top_p=1.0)
        fixed = FixedNativeStrategy(**common)
        variable = VariableLengthStrategy(
            length_predictor=None, length_policy=resolve_generation_length_policy("inferred"),
            length_temperature=1.0, **common)
        for eval_index, corpus_index in ((0, 5), (1, 4)):
            with self.subTest(eval_index=eval_index):
                self.assertEqual(fixed.judge_inputs(eval_index, "A"),
                                 masked_view(full[corpus_index])[:2])
                self.assertEqual(variable.judge_inputs(eval_index, "ACD"), build_masked_judge_prompt(
                    tok, full.framework_parts(corpus_index), 3))

    def test_eos_judge_framework_follows_a_nested_eval_subset(self):
        tok = s2s_adapter()
        path = corpus(self)
        full = Seq2SeqInfillingDataset(path, tok, target_region="h3")
        judge = MaskedInfillingDataset(path, TokenizerAdapter.from_custom(AntibodyTokenizer()),
                                       target_region="h3")
        nested = torch.utils.data.Subset(torch.utils.data.Subset(full, [4, 1, 5, 2]), [2, 0])
        strategy = Seq2SeqStrategy(
            model=None, tokenizer=tok, device=CPU, generator=None, eval_dataset=nested,
            judge_dataset=judge, judge_tokenizer=None, num_samples=1, temperature=0, top_k=0,
            top_p=1.0, max_new_tokens=8)
        self.assertEqual(strategy._judge_framework(0), judge.framework_parts(5))
        self.assertEqual(strategy._judge_framework(1), judge.framework_parts(4))


class PayloadAlignmentTests(unittest.TestCase):
    """MLM and the EOS methods write the same evaluation payload schema."""

    COMMON = {"schema_version", "blosum", "method", "judge", "length_policy", "anchor_trim_each_end",
              "decoding", "n_eval_total", "n_selected", "n_generated", "summary", "records"}

    def payloads(self):
        mlm_tok = TokenizerAdapter.from_custom(AntibodyTokenizer())
        path = corpus(self)
        mlm = mlm_evaluate(
            FlatMasked(mlm_tok.vocab_size), mlm_tok,
            MaskedInfillingDataset(path, mlm_tok, max_len=64, target_region="h3"), device=CPU,
            num_examples=-1, num_samples=2, temperature=1.0, top_k=0, top_p=1.0, seed=0,
            skip_perplexity=True)
        s2s_tok = s2s_adapter()
        s2s = s2s_evaluate(
            ScriptedSeq2Seq(s2s_tok.encode("AC") + [s2s_tok.eos_id], s2s_tok.vocab_size), s2s_tok,
            Seq2SeqInfillingDataset(path, s2s_tok, target_region="h3"), device=CPU,
            num_examples=-1, num_samples=2, temperature=1.0, top_k=0, top_p=1.0, seed=0,
            skip_perplexity=True)
        return mlm, s2s

    def test_both_payloads_carry_the_shared_keys_and_one_schema_version(self):
        mlm, s2s = self.payloads()
        self.assertLessEqual(self.COMMON, set(mlm))
        self.assertLessEqual(self.COMMON, set(s2s))
        self.assertEqual((mlm["schema_version"], s2s["schema_version"]),
                         (GENERATION_EVAL_SCHEMA, GENERATION_EVAL_SCHEMA))
        self.assertEqual((mlm["method"], s2s["method"]), ("mlm", "s2s"))

    def test_n_selected_counts_the_chosen_frameworks_in_both(self):
        for payload in self.payloads():
            self.assertEqual((payload["n_selected"], payload["n_generated"], payload["n_eval_total"]),
                             (6, 6, 6))

    def test_only_the_eos_payload_has_generation_validity(self):
        mlm, s2s = self.payloads()
        self.assertNotIn("generation_validity", mlm)
        self.assertIn("generation_validity", s2s)


class CaseRecordAlignmentTests(unittest.TestCase):
    """MLM and the EOS methods describe a generated case in the same terms."""

    def records(self, truth, skip_perplexity=False):
        mlm_tok = TokenizerAdapter.from_custom(AntibodyTokenizer())
        case = {"name": "c", "pt1": "ACDEFGHIK", "pt2": "LMNPQRST", "target_truth": truth}
        mlm = mlm_generate_case(
            FlatMasked(mlm_tok.vocab_size), mlm_tok, CPU, SimpleNamespace(
                partner_chain=None, antigen=None, max_len=64, decoding="independent", num_samples=2,
                temperature=1.0, top_k=0, top_p=1.0, length_temperature=1.0,
                skip_perplexity=skip_perplexity, seed=0, min_length=2, max_length=2,
                target_length=None, use_native_length=False),
            torch.Generator().manual_seed(0), case, None, Mock())
        s2s_tok = s2s_adapter()
        s2s = s2s_generate_case(
            ScriptedSeq2Seq(s2s_tok.encode("AC") + [s2s_tok.eos_id], s2s_tok.vocab_size), s2s_tok, CPU,
            SimpleNamespace(max_len=64, partner_chain=None, antigen=None, num_samples=2,
                            temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=8,
                            skip_perplexity=skip_perplexity, seed=0),
            torch.Generator().manual_seed(0), case, None, Mock())
        return mlm, s2s

    def test_a_candidate_of_another_length_has_recovery_none_in_both(self):
        for record in self.records("WYV"):
            for sample in record["samples"]:
                self.assertEqual((sample["length_match"], sample["recovery"]), (False, None))

    def test_samples_have_the_same_keys_apart_from_the_score(self):
        for truth in (None, "WYV", "WY"):
            with self.subTest(truth=truth):
                mlm, s2s = self.records(truth)
                keys = [{key for sample in record["samples"] for key in sample}
                        - {"pseudo_perplexity", "perplexity"} for record in (mlm, s2s)]
                self.assertEqual(keys[0], keys[1])

    def test_both_summarise_the_self_scores(self):
        mlm, s2s = self.records("WY")
        for key in ("mean", "min", "max"):
            self.assertAlmostEqual(mlm["metrics"][f"pseudo_perplexity_{key}"], 20.0, places=3)
            self.assertAlmostEqual(s2s["metrics"][f"perplexity_{key}"], 9.0, places=3)

    def test_skipping_perplexity_leaves_no_summary_in_either(self):
        for record in self.records("WY", skip_perplexity=True):
            self.assertFalse([key for key in record["metrics"] if "perplexity" in key])
            self.assertFalse([key for sample in record["samples"] for key in sample if "perplexity" in key])

    def test_both_count_retained_candidates_in_decoding_num_samples(self):
        for record in self.records("WY"):
            self.assertEqual(record["decoding"]["num_samples"], len(record["samples"]))

    def test_eos_candidates_are_uppercased_like_the_evaluation_path(self):
        tok = s2s_adapter()
        real_decode = tok.decode
        with patch.object(tok, "decode", side_effect=lambda ids: real_decode(ids).lower()):
            record = s2s_generate_case(
                ScriptedSeq2Seq(tok.encode("AC") + [tok.eos_id], tok.vocab_size), tok, CPU,
                SimpleNamespace(max_len=64, partner_chain=None, antigen=None, num_samples=2,
                                temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=8,
                                skip_perplexity=True, seed=0),
                torch.Generator().manual_seed(0),
                {"name": "c", "pt1": "ACDEFGHIK", "pt2": "LMNPQRST"}, None, Mock())
        self.assertEqual([sample["seq"] for sample in record["samples"]], ["AC", "AC"])


class PrintCaseTests(unittest.TestCase):
    """The case display runs after a whole generation pass, so a format slip would cost the run."""

    def test_every_sample_and_metric_kind_prints(self):
        args = SimpleNamespace(decoding="causal", temperature=1.0, top_k=0, top_p=0.9, show_samples=2)
        samples = [{"seq": "AC", "length": 2, "recovery": 1.0, "perplexity": 9.0},
                   {"seq": "AD", "length": 2, "recovery": None, "pseudo_perplexity": 20.0},
                   {"seq": "AE", "length": 2}]
        metrics = {"diversity_hamming": 0.5, "n_unique": 3, "n_samples": 3,
                   "per_position_entropy_bits": 1.2, "aar_mean": 0.5, "aar_max": 1.0, "aar_min": 0.0,
                   "novelty_mean": 0.5, "pseudo_perplexity_mean": 20.0, "pseudo_perplexity_min": 20.0,
                   "pseudo_perplexity_max": 20.0, "perplexity_mean": 9.0, "perplexity_min": 8.5,
                   "perplexity_max": 9.5}
        with redirect_stdout(io.StringIO()) as printed:
            shared_generation._print_case("case", "AAA", "CCC", "2", "AC", args, samples, metrics)
        text = printed.getvalue()
        for expected in ("rec=100.0%", "ppl=  9.00", "ppl= 20.00", "(+1 more)", "native: AC",
                         "unique=3/3", "entropy=1.20b", "recovery : mean=50.0%",
                         "pseudo-ppl: mean=20.00", "ppl: mean=9.00  min=8.50  max=9.50"):
            self.assertIn(expected, text)
        self.assertNotIn("AE", text)       # beyond show_samples, only counted


class EvaluationPipelineGuardTests(unittest.TestCase):
    """A mismatched checkpoint/config pair stops standalone evaluation before data or weights load."""

    PROBLEMS = ["target CDR: checkpoint recorded 'h3', but --config supplies 'h1'"]

    def run_pipeline(self, run_config, problems, *flags):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        config_path = os.path.join(directory, "config.json")
        with open(config_path, "w") as handle:
            json.dump({"method": "mlm"}, handle)
        args = evaluate.parse_args(["--config", config_path, "--model_dir", directory,
                                    "--output", "none", *flags])
        tokenizer, dataset = Mock(), [object()]
        backbone = Mock()
        get_datasets = Mock(return_value=([], dataset, tokenizer))
        modules = {"backbones": SimpleNamespace(resolve=Mock(return_value=backbone)),
                   "training": SimpleNamespace(get_datasets=get_datasets),
                   "arguments": SimpleNamespace(ModelArguments=object, DataArguments=object,
                                                GenerationArguments=object)}
        method = SimpleNamespace(name="mlm", module=Mock(side_effect=modules.__getitem__))
        parsed = (SimpleNamespace(model_type="scratch", anchor_trim=3),
                  SimpleNamespace(data_path="/tmp/corpus.json", target_region="h3",
                                  eval_split="validation", include_partner_chain=False,
                                  include_antigen=False, context_dropout=0.0),
                  SimpleNamespace(), SimpleNamespace())
        hook = Mock(return_value={"payload": True})
        with patch.object(shared_evaluation, "resolve", return_value=method), \
                patch.object(shared_evaluation, "HfArgumentParser"), \
                patch.object(shared_evaluation, "parse_config_dict", return_value=parsed), \
                patch.object(shared_evaluation, "compatibility_problems", return_value=problems), \
                patch.object(shared_evaluation, "get_evaluation_source_info", return_value={}), \
                patch.object(shared_evaluation, "evaluation_context_view", return_value=dataset), \
                patch.object(shared_evaluation.torch.cuda, "is_available", return_value=False), \
                patch("builtins.print"):
            try:
                result, refusal = shared_evaluation.run_evaluation_pipeline(
                    "mlm", args, {"method": "mlm"}, run_config, prepare_evaluation=Mock(),
                    evaluation_options=Mock(return_value={}), evaluate=hook), None
            except SystemExit as exc:
                result, refusal = None, exc
        return result, refusal, get_datasets, backbone, hook

    def test_a_mismatch_refuses_before_loading_data_or_weights(self):
        result, refusal, get_datasets, backbone, hook = self.run_pipeline({"method": "mlm"}, self.PROBLEMS)
        self.assertIsNone(result)
        self.assertIn("Refusing to evaluate", str(refusal))
        get_datasets.assert_not_called()
        backbone.load_pretrained_model.assert_not_called()
        hook.assert_not_called()

    def test_force_runs_and_records_every_mismatch_in_the_payload_metadata(self):
        result, refusal, _, _, hook = self.run_pipeline({"method": "mlm"}, self.PROBLEMS, "--force")
        self.assertIsNone(refusal)
        self.assertEqual(result, {"payload": True})
        self.assertEqual(hook.call_args.kwargs["meta"]["compatibility"],
                         {"problems": self.PROBLEMS, "forced": True})

    def test_a_checkpoint_without_a_run_config_cannot_be_verified_and_does_not_block(self):
        result, refusal, _, _, hook = self.run_pipeline(None, [])
        self.assertIsNone(refusal)
        self.assertEqual(hook.call_args.kwargs["meta"]["compatibility"],
                         {"problems": [], "forced": False})


if __name__ == "__main__":
    unittest.main()
