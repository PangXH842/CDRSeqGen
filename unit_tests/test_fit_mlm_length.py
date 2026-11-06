import json
from contextlib import contextmanager
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, sentinel

import fit_mlm_length as command
from fit_mlm_length import resolve_config_path


class ResolveMLMLengthConfigTests(unittest.TestCase):
    def test_discovers_canonical_run_config(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            config = checkpoint / "run_config.json"
            config.write_text(json.dumps({"model_type": "scratch"}))

            self.assertEqual(resolve_config_path(str(checkpoint)), config.resolve())

    def test_explicit_config_overrides_checkpoint_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "run_config.json").write_text("{}")
            explicit = root / "training.json"
            explicit.write_text("{}")

            self.assertEqual(
                resolve_config_path(str(checkpoint), str(explicit)), explicit.resolve()
            )

    def test_missing_config_fails_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "pass --config explicitly"):
                resolve_config_path(directory)


class FitLengthCommandTests(unittest.TestCase):
    @contextmanager
    def harness(self, *flags, method="mlm", saved_method="mlm"):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "run_config.json").write_text(json.dumps({"method": saved_method}))
            config = checkpoint / "original.json"
            config.write_text(json.dumps({"method": method}))
            args = command.parse_args(["--model_dir", directory, "--config", str(config),
                                       "--device", "cpu", *flags])
            parsed = (SimpleNamespace(model_type="scratch"),
                      SimpleNamespace(length_predictor_epochs=7,
                                      length_predictor_learning_rate=0.02,
                                      length_predictor_batch_size=4),
                      SimpleNamespace(), SimpleNamespace(seed=43))
            backbone = Mock()
            with patch.object(command, "configure_logging"), \
                    patch.object(command, "HfArgumentParser"), \
                    patch.object(command, "parse_config_dict", return_value=parsed), \
                    patch.object(command, "resolve", return_value=backbone) as resolve, \
                    patch.object(command, "get_datasets", return_value=(sentinel.train, sentinel.eval,
                                                                         sentinel.tokenizer)) as datasets, \
                    patch.object(command, "fit_length_predictor", return_value=sentinel.head) as fit:
                yield args, backbone, resolve, datasets, fit

    def test_saved_tokenizer_and_config_defaults_reach_fitter(self):
        with self.harness() as (args, backbone, resolve, datasets, fit):
            self.assertIs(command.main(args), sentinel.head)
            self.assertEqual(datasets.call_args.kwargs["checkpoint_path"], args.model_dir)
            backbone.load_pretrained_model.assert_called_once_with(args.model_dir)
            options = fit.call_args.kwargs
            self.assertEqual((options["epochs"], options["learning_rate"], options["batch_size"]),
                             (7, 0.02, 4))
            self.assertEqual(options["seed"], 43)
            self.assertEqual(options["output_dir"], args.model_dir)

    def test_explicit_fit_overrides_are_forwarded(self):
        with self.harness("--epochs", "3", "--learning_rate", "0.001", "--batch_size", "2") as setup:
            args, _, _, _, fit = setup
            command.main(args)
            self.assertEqual((fit.call_args.kwargs["epochs"], fit.call_args.kwargs["learning_rate"],
                              fit.call_args.kwargs["batch_size"]), (3, 0.001, 2))

    def test_invalid_overrides_fail_before_data_or_weights_load(self):
        for flag, value in (("--epochs", "0"), ("--epochs", "-1"), ("--batch_size", "0"),
                            ("--learning_rate", "0"), ("--learning_rate", "nan"),
                            ("--learning_rate", "inf")):
            with self.subTest(flag=flag, value=value), self.harness(flag, value) as setup:
                args, backbone, resolve, datasets, fit = setup
                with self.assertRaisesRegex(ValueError, flag[2:]):
                    command.main(args)
                resolve.assert_not_called()
                datasets.assert_not_called()
                backbone.load_pretrained_model.assert_not_called()
                fit.assert_not_called()

    def test_both_explicit_and_saved_configs_must_be_mlm(self):
        for supplied, saved in (("ar", "mlm"), ("mlm", "s2s")):
            with self.subTest(supplied=supplied, saved=saved), \
                    self.harness(method=supplied, saved_method=saved) as setup:
                args, _, resolve, datasets, fit = setup
                with self.assertRaisesRegex(ValueError, "method"):
                    command.main(args)
                resolve.assert_not_called()
                datasets.assert_not_called()
                fit.assert_not_called()


class RootEvaluationConfigTests(unittest.TestCase):
    def test_nonobject_config_fails_before_checkpoint_routing(self):
        import evaluate
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text("[]")
            args = evaluate.parse_args(["--config", str(config), "--model_dir", directory])
            with patch.object(evaluate, "read_run_config") as read, \
                    self.assertRaisesRegex(ValueError, "JSON object"):
                evaluate.main(args)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
