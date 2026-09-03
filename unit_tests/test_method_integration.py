"""Cross-method contracts introduced by the dispatching training stack."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.data.dataset_ar import CAUSAL_DECODE_STEPS, LOOP_LENGTH_RESERVE
from cdrseqgen.methods.ar.arguments import (DataArguments as ARDataArguments,
                                            GenerationArguments as ARGenerationArguments,
                                            ModelArguments as ARModelArguments,
                                            validate_args as validate_ar_args)
from cdrseqgen.methods.common.arguments import (
    BaseDataArguments,
    BaseGenerationArguments,
    BaseModelArguments,
    parse_config_and_cli,
    validate_common_args,
)
from cdrseqgen.methods.common.training import run_training_pipeline
from cdrseqgen.methods.mlm.arguments import GenerationArguments as MLMGenerationArguments
from train import resolve_method


class DispatcherTests(unittest.TestCase):
    def test_cli_method_override_remains_for_the_method_parser(self):
        argv = ["train.py", "configs/h3_scratch.json", "--method", "ar"]
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


class CausalConfigTests(unittest.TestCase):
    def test_decode_cap_covers_the_reserve_but_cannot_overrun_the_prompt_budget(self):
        generation = ARGenerationArguments()
        self.assertEqual(generation.gen_max_new_tokens, LOOP_LENGTH_RESERVE + 1)
        self.assertEqual(generation.gen_max_new_tokens, CAUSAL_DECODE_STEPS)
        generation.gen_max_new_tokens += 1
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            validate_ar_args(ARModelArguments(), ARDataArguments(), generation)

    def test_all_causal_configs_select_on_core_recovery(self):
        configs = sorted(Path("configs").glob("*_ar_*.json"))
        self.assertEqual(len(configs), 6)
        for path in configs:
            with self.subTest(path=path.name):
                config = json.loads(path.read_text())
                self.assertEqual(config["metric_for_best_model"], "core_accuracy")

    def test_compare_suite_matches_every_budget_across_the_two_methods(self):
        """A head-to-head is only a comparison if the arms differ in method and nothing else.

        Three settings diverge between h3_scratch.json and h3_ar_scratch.json, and all three have
        to be forced onto the masked arm: the length policy (the masked side would otherwise be
        handed the native mask count), max_len (128 truncates the flanks of 12% of OAS heavy
        chains, where the causal arm's 256 truncates none), and gen_top_p (1.0 against 0.9 is a
        different sampling policy).
        """
        script = Path("scripts/run.sh").read_text()
        self.assertIn('"configs/h3_scratch.json"', script)
        self.assertIn('"configs/h3_scratch_lc.json"', script)
        for flag in ("--gen_length_mode inferred", "--max_len 256", "--gen_top_p 0.9"):
            with self.subTest(flag=flag):
                self.assertIn(flag, script)

    def test_the_compare_overrides_match_what_the_causal_configs_declare(self):
        """The forced values are read off the AR configs, so a config edit cannot silently unmatch
        the pair without this failing."""
        script = Path("scripts/run.sh").read_text()
        for name in ("h3_ar_scratch.json", "h3_ar_scratch_lc.json"):
            config = json.loads((Path("configs") / name).read_text())
            with self.subTest(config=name):
                self.assertIn(f"--max_len {config['max_len']}", script)
                self.assertIn(f"--gen_top_p {config['gen_top_p']}", script)

    def test_a_corpus_override_survives_the_compare_overrides(self):
        """`RUN_OVERRIDES=` in the compare branch would silently drop --data_path; it must append.

        A compare batch on a second corpus is exactly when both are needed at once, and the failure
        would be invisible: the run would train on the config's declared corpus and be named after
        it, so the batch table would look consistent while measuring the wrong data.
        """
        script = Path("scripts/run.sh").read_text()
        self.assertIn('RUN_OVERRIDES+=(--data_path "$DATA_PATH")', script)
        self.assertIn("RUN_OVERRIDES+=(--gen_length_mode inferred", script)
        self.assertNotIn("RUN_OVERRIDES=(--gen_length_mode", script)

    def test_a_second_oas_corpus_gets_its_own_dataset_tag(self):
        """Two donor sets have two test splits, so they must not share a run-name tag."""
        from cdrseqgen.train.logging import dataset_tag
        self.assertEqual(dataset_tag("dataset/data/processed/processed_oas_tuning.json"), "OAS")
        self.assertEqual(dataset_tag("dataset/data/processed/processed_oas_multi.json"), "OAS4")
        self.assertEqual(dataset_tag("dataset/data/processed/processed_sabdab2_data.json"), "SD2")

    def test_batch_and_sweep_runners_fail_after_reporting_failed_trials(self):
        batch_script = Path("scripts/run.sh").read_text()
        sweep_script = Path("scripts/sweep.sh").read_text()
        self.assertIn("[[ $FAILED_COUNT -eq 0 ]] || exit 1", batch_script)
        self.assertIn("if [[ $ROUND_FAILURES -gt 0 ]]; then", sweep_script)
        self.assertIn('-f "$RUNDIR/generation_eval.json"', sweep_script)
        self.assertIn('-f "$COMPLETED_RUN/generation_eval.json"', sweep_script)
        self.assertIn("exit 1", sweep_script)



if __name__ == "__main__":
    unittest.main()
