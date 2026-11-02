"""Contracts of the small shared helpers behind the MLM/common presentation cleanup.

  * MLM decoder names are validated wherever a run can name one (training config, standalone
    evaluation, standalone generation, the sampler itself) instead of falling back to iterative.
  * Both MLM decoders run without autograd.
  * anchor_trim validation lives in one place.
  * The context-regime helpers behind the generative-evaluation payload keep their key order.
  * evaluate.py and generate.py define their shared flags once, those flags mirror the
    training-time Generation fields, and each command's public flag set cannot change silently.
"""
import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import TrainingArguments

import evaluate
import generate
from cdrseqgen.methods.common.arguments import validate_anchor_trim
from cdrseqgen.methods.common.evaluation import _context_meta, _partner_context
from cdrseqgen.methods.mlm import arguments as mlm_arguments
from cdrseqgen.methods.mlm.arguments import validate_inference_args
from cdrseqgen.methods.mlm.evaluation import prepare_evaluation
from cdrseqgen.methods.mlm.generation import prepare_generation
from cdrseqgen.methods.mlm.sampling import sample_masked_span

CASE = [{"pt1": "AC", "pt2": "WY"}]


def _args(**overrides):
    values = dict(decoding=None, max_new_tokens=None, length_mode=None,
                  min_length=None, max_length=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class DecoderNameTests(unittest.TestCase):
    def test_only_the_mlm_decoders_are_accepted(self):
        for name in (None, "independent", "iterative"):
            validate_inference_args(_args(decoding=name))
        for name in ("causal", "seq2seq", "iterativ"):
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "independent or iterative"):
                validate_inference_args(_args(decoding=name))

    def test_the_ar_only_cap_keeps_its_own_message(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            validate_inference_args(_args(max_new_tokens=64))

    def test_standalone_evaluation_and_generation_share_the_check(self):
        with self.assertRaisesRegex(ValueError, "independent or iterative"):
            prepare_evaluation(_args(decoding="seq2seq"), {})
        with self.assertRaisesRegex(ValueError, "independent or iterative"):
            prepare_generation(_args(decoding="seq2seq"), CASE, {})

    def test_generation_adopts_a_recorded_decoder_only_for_mixed_masking(self):
        for masking, expected in [("mixed", "iterative"), ("full_span", "independent"),
                                  (None, "independent")]:
            with self.subTest(masking=masking):
                args = _args()
                config = {"masking_policy": masking, "gen_decoding": "iterative"}
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare_generation(args, CASE, config)
                self.assertEqual(args.decoding, expected)

    def test_an_explicit_decoder_wins_over_the_recorded_one(self):
        args = _args(decoding="independent")
        config = {"masking_policy": "mixed", "gen_decoding": "iterative"}
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_generation(args, CASE, config)
        self.assertEqual(args.decoding, "independent")


class TrainingConfigDecoderTests(unittest.TestCase):
    def _groups(self, **generation):
        return (mlm_arguments.ModelArguments(), mlm_arguments.DataArguments(),
                mlm_arguments.GenerationArguments(**generation),
                TrainingArguments(output_dir="unused"))

    def test_gen_decoding_is_normalized_like_the_other_mode_strings(self):
        groups = self._groups(gen_decoding=" Iterative ")
        with patch.object(mlm_arguments, "parse_config_and_cli", return_value=groups), \
                patch.object(mlm_arguments, "initialize_run"):
            _, _, gen_args, _ = mlm_arguments.setup_args()
        self.assertEqual(gen_args.gen_decoding, "iterative")

    def test_a_bad_decoder_fails_before_the_run_directory_exists(self):
        for bad in ("iterativ", "causal", "seq2seq", ""):
            with self.subTest(bad=bad):
                groups = self._groups(gen_decoding=bad)
                with patch.object(mlm_arguments, "parse_config_and_cli", return_value=groups), \
                        patch.object(mlm_arguments, "initialize_run") as initialise:
                    with self.assertRaisesRegex(ValueError, "gen_decoding"):
                        mlm_arguments.setup_args()
                initialise.assert_not_called()


class _GradProbe(torch.nn.Module):
    """Uniform logits that remember whether autograd was on during each forward."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(25))
        self.grad_modes = []

    def forward(self, input_ids):
        self.grad_modes.append(torch.is_grad_enabled())
        batch, length = input_ids.shape
        return {"logits": self.weight.expand(batch, length, 25) + 0.0}


class SamplerTests(unittest.TestCase):
    tokenizer = SimpleNamespace(special_ids=[0, 1, 2, 3], mask_id=3)

    def _draw(self, model, decoding):
        return sample_masked_span(
            model, self.tokenizer, "cpu", torch.Generator().manual_seed(0), [4, 3, 3, 5], [1, 2],
            decoding=decoding, num_samples=2, temperature=1.0, top_k=0, top_p=1.0)

    def test_both_decoders_run_without_autograd(self):
        for decoding in ("independent", "iterative"):
            with self.subTest(decoding=decoding):
                model = _GradProbe()
                self._draw(model, decoding)
                self.assertTrue(model.grad_modes)
                self.assertFalse(any(model.grad_modes))

    def test_an_unknown_decoder_is_an_error_not_a_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, "unknown decoding"):
            self._draw(_GradProbe(), "iterativ")


class AnchorTrimTests(unittest.TestCase):
    def test_non_negative_integers_pass(self):
        for value in (0, 1, 2, 7):
            validate_anchor_trim(value)

    def test_everything_else_is_rejected(self):
        for value in (-1, 1.5, 2.0, "2", None, True, False):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "non-negative integer"):
                validate_anchor_trim(value)


class ContextHelperTests(unittest.TestCase):
    def test_partner_context_resolution(self):
        for setting, trained, expected in [("trained", True, True), ("trained", False, False),
                                           ("with", False, True), ("without", True, False)]:
            with self.subTest(setting=setting, trained=trained):
                self.assertIs(_partner_context(setting, trained), expected)

    def test_context_meta_keeps_the_payload_key_order(self):
        data = SimpleNamespace(include_partner_chain=True, include_antigen=False,
                               context_dropout=0.3)
        meta = _context_meta(data, partner=False, judge_partner=True)
        self.assertEqual(list(meta), [
            "include_partner_chain", "training_include_partner_chain",
            "training_context_dropout", "eval_context_dropout",
            "judge_include_partner_chain", "include_antigen",
        ])
        self.assertEqual(list(meta.values()), [False, True, 0.3, 0.0, True, False])


SHARED_FLAGS = ("decoding", "num_samples", "temperature", "top_k", "top_p", "seed",
                "skip_perplexity", "min_length", "max_length", "length_temperature",
                "max_new_tokens")

# Standalone flag -> the training-time Generation field it mirrors, wherever both have a default.
# Not listed on purpose: --decoding, --max_new_tokens, --length_mode and --judge are None here
# ("take it from the checkpoint" / "no judge") while the fields carry trained defaults.
MIRRORED_DEFAULTS = {
    "num_examples": "gen_num_examples",
    "num_samples": "gen_num_samples", "temperature": "gen_temperature", "top_k": "gen_top_k",
    "top_p": "gen_top_p", "seed": "gen_seed", "skip_perplexity": "gen_skip_perplexity",
    "stratify": "gen_stratify", "judge_path": "gen_judge_path",
    "judge_partner_context": "gen_judge_partner_context",
    "length_temperature": "gen_length_temperature", "min_length": "gen_min_length",
    "max_length": "gen_max_length",
}

EVALUATE_OPTIONS = [
    "--config", "--decoding", "--force", "--judge", "--judge_partner_context", "--judge_path",
    "--length_mode", "--length_temperature", "--max_length", "--max_new_tokens", "--min_length",
    "--model_dir", "--model_path", "--num_examples", "--num_samples", "--output", "--overwrite",
    "--partner_context", "--seed", "--skip_perplexity", "--stratify", "--temperature", "--top_k",
    "--top_p",
]
GENERATE_OPTIONS = [
    "--antigen", "--decoding", "--force", "--length_temperature", "--max_len", "--max_length",
    "--max_new_tokens", "--method", "--min_length", "--model_dir", "--model_path", "--model_type",
    "--num_samples", "--output", "--partner_chain", "--pt1", "--pt2", "--seed", "--show_samples",
    "--skip_perplexity", "--target_length", "--target_loop", "--target_truth", "--temperature",
    "--test_file", "--top_k", "--top_p", "--use_native_length",
]


def _flags(parser):
    return {action.dest: action for action in parser._actions if action.dest != "help"}


def _options(parser):
    return sorted(option for action in parser._actions for option in action.option_strings
                  if option not in ("-h", "--help"))


class StandaloneCommandTests(unittest.TestCase):
    def test_the_commands_share_one_definition_of_the_sampling_flags(self):
        evaluation, generation = _flags(evaluate.build_parser()), _flags(generate.build_parser())
        for dest in SHARED_FLAGS:
            with self.subTest(flag=dest):
                left, right = evaluation[dest], generation[dest]
                self.assertEqual(
                    (left.option_strings, type(left), left.type, left.default, left.choices,
                     left.nargs, left.help),
                    (right.option_strings, type(right), right.type, right.default, right.choices,
                     right.nargs, right.help))

    def test_standalone_defaults_mirror_the_training_time_fields(self):
        training = {field.name: field.default
                    for field in dataclasses.fields(mlm_arguments.GenerationArguments)}
        flags = {**_flags(generate.build_parser()), **_flags(evaluate.build_parser())}
        for dest, field_name in MIRRORED_DEFAULTS.items():
            with self.subTest(flag=dest):
                self.assertEqual(flags[dest].default, training[field_name])

    def test_public_flag_sets_are_pinned(self):
        self.assertEqual(_options(evaluate.build_parser()), EVALUATE_OPTIONS)
        self.assertEqual(_options(generate.build_parser()), GENERATE_OPTIONS)

    def test_both_commands_accept_either_spelling_of_the_checkpoint_directory(self):
        for spelling in ("--model_dir", "--model_path"):
            with self.subTest(spelling=spelling):
                evaluated = evaluate.parse_args(["--config", "c.json", spelling, "/tmp/run"])
                generated = generate.parse_args([spelling, "/tmp/run"])
                self.assertEqual(evaluated.model_dir, "/tmp/run")
                self.assertEqual(generated.model_dir, "/tmp/run")

    def test_commands_parse_sys_argv_when_called_without_arguments(self):
        method = Mock()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"method": "mlm"}))
            argv = ["evaluate.py", "--config", str(config), "--model_dir", directory, "--seed", "7"]
            with patch("sys.argv", argv), patch("evaluate.read_run_config", return_value=None), \
                    patch("evaluate.resolve", return_value=method):
                evaluate.main()
            parsed = method.module.return_value.main.call_args.args[0]
            self.assertEqual((parsed.model_dir, parsed.seed), (directory, 7))

        method = Mock()
        argv = ["generate.py", "--model_path", "/tmp/run", "--seed", "7"]
        with patch("sys.argv", argv), patch("generate.read_run_config", return_value={}), \
                patch("generate.checkpoint_method", return_value=method):
            generate.main()
        parsed = method.module.return_value.main.call_args.args[0]
        self.assertEqual((parsed.model_dir, parsed.seed), ("/tmp/run", 7))


if __name__ == "__main__":
    unittest.main()
