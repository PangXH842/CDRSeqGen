"""Cross-method contracts introduced by the dispatching training stack."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.datasets.dataset_ar import CAUSAL_DECODE_STEPS, LOOP_LENGTH_RESERVE
from cdrseqgen.methods.ar.arguments import (DataArguments as ARDataArguments,
                                            GenerationArguments as ARGenerationArguments,
                                            ModelArguments as ARModelArguments,
                                            validate_args as validate_ar_args)
from cdrseqgen.methods.common.arguments import (
    BaseDataArguments,
    BaseGenerationArguments,
    BaseModelArguments,
    parse_config_and_cli,
    parse_config_dict,
    validate_common_args,
)
from cdrseqgen.methods.common.training import run_training_pipeline
from cdrseqgen.methods.mlm.arguments import GenerationArguments as MLMGenerationArguments
from train import resolve_method


class DispatcherTests(unittest.TestCase):
    def test_cli_method_override_remains_for_the_method_parser(self):
        argv = ["train.py", "configs/mlm/heavy/h3_scratch.json", "--method", "ar"]
        method, routed = resolve_method(argv)
        self.assertEqual(method, "ar")
        self.assertEqual(routed, argv)

    def test_equals_form_and_last_override_win(self):
        method, _ = resolve_method(
            ["train.py", "--method", "mlm", "--method=AR"])
        self.assertEqual(method, "ar")

    def test_config_method_is_normalized(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump({"method": " AR "}, handle)
            handle.flush()
            method, _ = resolve_method(["train.py", handle.name])
        self.assertEqual(method, "ar")

    def test_override_is_also_recorded_by_the_method_parser(self):
        parser = HfArgumentParser((
            ARModelArguments, ARDataArguments, ARGenerationArguments, TrainingArguments))
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump({"method": "mlm", "output_dir": "/tmp/unused"}, handle)
            handle.flush()
            argv = ["train.py", handle.name, "--method", "ar"]
            with patch("sys.argv", argv):
                model, _, _, _ = parse_config_and_cli(parser)

        # The old dispatcher removed --method before parsing: it routed to AR but persisted
        # method=mlm in run_config.json. The override must reach both decisions.
        self.assertEqual(model.method, "ar")


class MethodRegistryTests(unittest.TestCase):
    """A method is only usable if every component the dispatchers reach for actually exists."""

    COMMAND_PROTOCOLS = {
        "train_cli": ("main",),
        "eval_cli": ("main", "prepare_evaluation", "evaluation_options", "evaluate"),
        "gen_cli": ("main", "prepare_generation", "generation_state", "generate_case"),
    }

    def test_every_registered_method_resolves_its_components(self):
        from cdrseqgen.methods.registry import METHODS, resolve
        for name, method in METHODS.items():
            with self.subTest(method=name):
                for component in ("arguments", "backbones", "training", "evaluation",
                                  *self.COMMAND_PROTOCOLS):
                    self.assertTrue(hasattr(method.module(component), "__name__"))
                self.assertTrue(callable(method.module("eval_cli").evaluate))
                self.assertTrue(callable(method.module("train_cli").main))
                self.assertTrue(callable(method.module("training").get_datasets))
                self.assertIs(resolve(name), method)

    def test_every_method_exposes_the_identical_command_protocols(self):
        """All three root commands must find the functions they call in every method."""
        from cdrseqgen.methods.registry import METHODS
        for name, method in METHODS.items():
            for component, functions in self.COMMAND_PROTOCOLS.items():
                command = method.module(component)
                for function in functions:
                    with self.subTest(method=name, command=component, function=function):
                        self.assertTrue(callable(getattr(command, function, None)))

    def test_post_training_evaluation_forwards_to_the_command_interface(self):
        """Preserve arguments and return values through the shared evaluation route."""
        from cdrseqgen.methods.registry import METHODS
        args = (object(), object(), object())
        options = {"length_mode": "inferred", "anchor_trim": 3, "seed": 43}
        for name, method in METHODS.items():
            with self.subTest(method=name):
                command = method.module("eval_cli")
                with patch.object(command, "evaluate", return_value={"records": []}) as evaluate:
                    self.assertEqual(method.evaluate(*args, **options), {"records": []})
                evaluate.assert_called_once_with(*args, **options)

    def test_a_checkpoint_cannot_be_reinterpreted_by_another_method(self):
        from cdrseqgen.methods.registry import checkpoint_method
        self.assertEqual(checkpoint_method({"method": "s2s"}).name, "s2s")
        with self.assertRaisesRegex(ValueError, "different method"):
            checkpoint_method({"method": "s2s"}, "ar")

    def test_decoding_choices_cover_every_method(self):
        """generate.py and evaluate.py hard-coded this list, so a method added later was rejected
        by argparse before its own cli could validate the request. Deriving it from the registry
        means an enumeration that cannot fall behind."""
        from cdrseqgen.methods.registry import METHODS, decoding_choices
        choices = decoding_choices()
        self.assertEqual(len(choices), len(set(choices)), "duplicate decoder name across methods")
        for name, method in METHODS.items():
            with self.subTest(method=name):
                self.assertTrue(method.decoders, f"{name} declares no decoder for --decoding")
                for decoder in method.decoders:
                    self.assertIn(decoder, choices)

    def test_the_dispatcher_routes_the_seq2seq_method(self):
        method, _ = resolve_method(["train.py", "--method", "s2s"])
        self.assertEqual(method, "s2s")


class CommonArgumentTests(unittest.TestCase):
    def _args(self):
        return (
            BaseModelArguments(),
            BaseDataArguments(),
            BaseGenerationArguments(),
        )

    def test_both_methods_expose_the_same_generation_seed(self):
        self.assertEqual(ARGenerationArguments().gen_seed, 42)
        self.assertEqual(MLMGenerationArguments().gen_seed, 42)

    def test_shared_generation_ranges_are_rejected_before_run_creation(self):
        invalid = [
            ("gen_num_examples", -2),
            ("gen_num_samples", 0),
            ("gen_temperature", -0.1),
            ("gen_top_k", -1),
            ("gen_top_p", 0.0),
            ("gen_top_p", 1.1),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                model, data, generation = self._args()
                setattr(generation, field, value)
                with self.assertRaises(ValueError):
                    validate_common_args(
                        model, data, generation, expected_method="mlm")

    def test_entrypoint_method_mismatch_is_rejected(self):
        model, data, generation = self._args()
        with self.assertRaisesRegex(ValueError, "entrypoint received"):
            validate_common_args(model, data, generation, expected_method="ar")

    def test_json_null_does_not_break_an_unrelated_cli_override(self):
        from dataclasses import dataclass

        @dataclass
        class Args:
            optional_count: int = None
            count: int = 1

        parser = HfArgumentParser(Args)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump({"optional_count": None, "count": 2}, handle)
            handle.flush()
            argv = ["prog", handle.name, "--count", "3"]
            with patch("sys.argv", argv):
                (parsed,) = parse_config_and_cli(parser)
        self.assertIsNone(parsed.optional_count)
        self.assertEqual(parsed.count, 3)

    def test_legacy_warmup_ratio_preserves_schedule_and_explicit_steps_take_precedence(self):
        parser = HfArgumentParser(TrainingArguments)
        base = dict(output_dir='/tmp/unused', use_cpu=True, report_to='none', warmup_ratio=.06)
        (training,) = parse_config_dict(parser, {**base, 'warmup_steps': 0})
        self.assertEqual(training.get_warmup_steps(1000), 60)
        (training,) = parse_config_dict(parser, {**base, 'warmup_steps': 12})
        self.assertEqual(training.get_warmup_steps(1000), 12)
        (training,) = parse_config_dict(parser, {**base, 'warmup_ratio': None})
        self.assertEqual(training.get_warmup_steps(1000), 0)

    def test_warmup_conversion_survives_an_unrelated_cli_override(self):
        parser = HfArgumentParser(TrainingArguments)
        with tempfile.NamedTemporaryFile('w', suffix='.json') as handle:
            json.dump(dict(output_dir='/tmp/unused', use_cpu=True, report_to='none',
                           warmup_ratio=.06, warmup_steps=0), handle)
            handle.flush()
            for tail in ([], ['--learning_rate', '0.0001']):
                with self.subTest(tail=tail), patch('sys.argv', ['train.py', handle.name, *tail]):
                    (training,) = parse_config_and_cli(parser)
                self.assertEqual(training.get_warmup_steps(1000), 60)

    def test_warmup_ratio_remains_native_for_a_parser_supporting_both_fields(self):
        from dataclasses import dataclass
        @dataclass
        class LegacyArgs:
            warmup_steps: int = 0
            warmup_ratio: float = 0
        (parsed,) = parse_config_dict(HfArgumentParser(LegacyArgs), dict(warmup_ratio=.06))
        self.assertEqual(parsed.warmup_ratio, .06)
        self.assertEqual(parsed.warmup_steps, 0)


class SharedTrainingLifecycleTests(unittest.TestCase):
    @patch("cdrseqgen.methods.common.training.plot_training_curves")
    @patch("cdrseqgen.methods.common.training.save_training_csv")
    def test_train_and_eval_metrics_and_tokenizer_are_persisted(self, save_csv, plot_curves):
        trainer = Mock()
        trainer.train.return_value = SimpleNamespace(metrics={"train_loss": 1.0})
        trainer.evaluate.return_value = {"eval_loss": 0.5}
        tokenizer = Mock()
        args = SimpleNamespace(do_train=True, do_eval=True, output_dir="/tmp/example")

        run_training_pipeline(trainer, args, tokenizer=tokenizer)

        trainer.save_model.assert_called_once_with(args.output_dir)
        trainer.save_metrics.assert_any_call("train", {"train_loss": 1.0})
        trainer.save_metrics.assert_any_call("eval", {"eval_loss": 0.5})
        tokenizer.save_pretrained.assert_called_once_with(args.output_dir)
        save_csv.assert_called_once_with(trainer, args.output_dir)
        plot_curves.assert_called_once_with(args.output_dir)


class Seq2SeqConfigTests(unittest.TestCase):
    def test_decode_cap_covers_the_decoder_budget(self):
        from cdrseqgen.datasets.dataset_s2s import MAX_LOOP_RESIDUES, S2S_DECODE_STEPS
        from cdrseqgen.methods.s2s.arguments import (DataArguments, GenerationArguments,
                                                     ModelArguments, validate_args)
        generation = GenerationArguments()
        self.assertEqual(generation.gen_max_new_tokens, MAX_LOOP_RESIDUES + 1)
        self.assertEqual(generation.gen_max_new_tokens, S2S_DECODE_STEPS)
        generation.gen_max_new_tokens += 1
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            validate_args(ModelArguments(), DataArguments(), generation)

    def test_stack_depths_must_be_positive_integers_or_unset(self):
        from cdrseqgen.methods.s2s.arguments import (DataArguments, GenerationArguments,
                                                     ModelArguments, validate_args)
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value):
                model = ModelArguments(num_encoder_layers=value)
                with self.assertRaisesRegex(ValueError, "num_encoder_layers"):
                    validate_args(model, DataArguments(), GenerationArguments())
        validate_args(ModelArguments(num_encoder_layers=2, num_decoder_layers=2),
                      DataArguments(), GenerationArguments())

    def test_the_shipped_configs_pin_both_stack_depths(self):
        """The comparison configs must not inherit num_layers on both stacks: that is 1.86M
        parameters against the causal arm's 800k, and the batch would confound architecture with
        2.3x capacity while every label still read 'scratch'."""
        import json
        from pathlib import Path
        for name in ("h3_s2s_scratch.json", "h3_s2s_scratch_lc.json"):
            with self.subTest(config=name):
                config = json.loads((Path("configs/s2s") / name).read_text())
                self.assertEqual(config["method"], "s2s")
                self.assertIn("num_encoder_layers", config)
                self.assertIn("num_decoder_layers", config)


class CausalConfigTests(unittest.TestCase):
    def test_decode_cap_covers_the_reserve_but_cannot_overrun_the_prompt_budget(self):
        generation = ARGenerationArguments()
        self.assertEqual(generation.gen_max_new_tokens, LOOP_LENGTH_RESERVE + 1)
        self.assertEqual(generation.gen_max_new_tokens, CAUSAL_DECODE_STEPS)
        generation.gen_max_new_tokens += 1
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            validate_ar_args(ARModelArguments(), ARDataArguments(), generation)

    def test_a_second_oas_corpus_gets_its_own_dataset_tag(self):
        """Two donor sets have two test splits, so they must not share a run-name tag."""
        from cdrseqgen.methods.common.run_setup import dataset_tag
        self.assertEqual(dataset_tag("data/processed/processed_oas_tuning.json"), "OAS")
        self.assertEqual(dataset_tag("data/processed/processed_oas_multi.json"), "OAS4")
        self.assertEqual(dataset_tag("data/processed/processed_sabdab2_data.json"), "SD2")


if __name__ == "__main__":
    unittest.main()
