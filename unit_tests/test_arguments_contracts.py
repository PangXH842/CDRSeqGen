"""Contracts of the argument modules: a bad config fails before a run directory exists.

  * A config the dataset or model would reject later (unknown loop, unknown backbone, heads that
    cannot split the width, early stopping that can never engage) is rejected by setup_args() in
    all three methods BEFORE initialize_run() creates the run directory.
  * A valid config still reaches initialize_run(), and a pretrained backbone keeps its own width
    and heads instead of being held to the scratch-model rules.
  * Adding a trailing CLI override does not change how a config's list and dict values are read.
  * Every shipped training config and every experiment-spec trial passes its method's setup_args().
  * The argument dataclasses stay documented, and a field that defaults to None is declared Optional.
"""
import contextlib
import dataclasses
import io
import json
import tempfile
import typing
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import HfArgumentParser, TrainingArguments

import evaluate
from cdrseqgen.datasets.dataset_common import AntibodyFrameworkDataset
from cdrseqgen.experiments import spec_trials
from cdrseqgen.methods.common.arguments import (
    PARTNER_CONTEXTS, BaseDataArguments, BaseGenerationArguments, BaseModelArguments,
    parse_config_and_cli, parse_config_dict, validate_common_args,
)
from cdrseqgen.methods.mlm import arguments as mlm_arguments
from cdrseqgen.methods.registry import DEFAULT_METHOD, METHODS, resolve

ROOT = Path(__file__).resolve().parent.parent
PRETRAINED = "facebook/esm2_t6_8M_UR50D"
EARLY_STOPPING = dict(early_stopping_patience=3, load_best_model_at_end=True,
                      eval_strategy="epoch", save_strategy="epoch")


def _parser(module):
    return HfArgumentParser((module.ModelArguments, module.DataArguments,
                             module.GenerationArguments, TrainingArguments))


@contextlib.contextmanager
def _setup_args(config):
    """The method's setup_args() on a config dict, with initialize_run replaced by a mock."""
    module = resolve(config.get("method", DEFAULT_METHOD)).module("arguments")
    groups = parse_config_dict(_parser(module), {**config, "use_cpu": True, "report_to": "none"})
    with patch.object(module, "parse_config_and_cli", return_value=groups), \
            patch.object(module, "initialize_run") as initialise:
        yield module.setup_args, initialise


def _shipped_training_configs():
    """(label, config) for each plain config under configs/ and each trial of each experiment spec."""
    cases = []
    with contextlib.chdir(ROOT):  # a spec names its base configs relative to the repository root
        for path in sorted((ROOT / "configs").rglob("*.json")):
            label = str(path.relative_to(ROOT))
            data = json.loads(path.read_text())
            if "trials" in data:
                trials, _ = spec_trials(label, {})
                cases += [(f"{label}:{trial['suffix']}", trial["config"]) for trial in trials]
            elif "target_region" in data:
                cases.append((label, data))
    return cases


class RejectedBeforeTheRunDirectoryTests(unittest.TestCase):
    def assert_rejected(self, config, pattern):
        with _setup_args(config) as (setup_args, initialise):
            with self.assertRaisesRegex(ValueError, pattern):
                setup_args()
        initialise.assert_not_called()

    def assert_accepted(self, config):
        with _setup_args(config) as (setup_args, initialise):
            setup_args()
        initialise.assert_called_once()

    def test_what_the_dataset_or_model_would_reject_later_fails_first(self):
        problems = {
            "unknown loop": (dict(target_region="h9"), "Unsupported target_region"),
            "unknown backbone": (dict(model_type="nope"), "No .*backbone registered"),
            "heads do not divide the width": (dict(hidden_size=130, num_attention_heads=8),
                                              "must be divisible by num_attention_heads=8"),
            "no heads": (dict(num_attention_heads=0), "must be divisible by num_attention_heads=0"),
            "zero width": (dict(hidden_size=0), "hidden_size must be positive"),
            "negative width": (dict(hidden_size=-128), "hidden_size must be positive"),
            "early stopping without best-model loading": (dict(early_stopping_patience=3),
                                                          "requires load_best_model_at_end"),
        }
        for method in METHODS:
            for problem, (overrides, pattern) in problems.items():
                with self.subTest(method=method, problem=problem):
                    self.assert_rejected(dict(method=method, **overrides), pattern)

    def test_rotary_models_need_an_even_head_width_and_the_masked_model_does_not(self):
        odd_heads = dict(hidden_size=130, num_attention_heads=2)  # heads of width 65
        for method in ("ar", "s2s"):
            with self.subTest(method=method):
                self.assert_rejected(dict(method=method, **odd_heads), "even head width")
        self.assert_accepted(dict(method="mlm", **odd_heads))

    def test_valid_configs_still_reach_the_run_directory(self):
        accepted = {
            "defaults": {},
            "early stopping with best-model loading": EARLY_STOPPING,
            "patience 0 needs no best-model loading": dict(early_stopping_patience=0),
            "heads of even width": dict(hidden_size=130, num_attention_heads=5),
            "loop spelled as the dataset spells it": dict(target_region="H3"),
            "legacy light alias": dict(target_region="lcdr1"),
        }
        for method in METHODS:
            for label, overrides in accepted.items():
                with self.subTest(method=method, config=label):
                    self.assert_accepted(dict(method=method, **overrides))

    def test_a_pretrained_backbone_keeps_its_own_width_and_heads(self):
        scratch_only = dict(model_type=PRETRAINED, hidden_size=130, num_attention_heads=8)
        for method in ("mlm", "ar"):
            with self.subTest(method=method):
                self.assert_accepted(dict(method=method, **scratch_only))
        # S2S builds its own residue decoder around a pretrained encoder, so its rules still apply.
        self.assert_rejected(dict(method="s2s", **scratch_only), "must be divisible")
        self.assert_rejected(dict(method="s2s", model_type=PRETRAINED, hidden_size=130,
                                  num_attention_heads=2), "even head width")
        self.assert_accepted(dict(method="s2s", model_type=PRETRAINED, hidden_size=130,
                                  num_attention_heads=5))

    def test_label_smoothing_belongs_to_the_masked_method(self):
        for method in ("ar", "s2s"):
            with self.subTest(method=method):
                self.assert_rejected(dict(method=method, label_smoothing_factor=0.1),
                                     "label_smoothing_factor is only supported by the masked method")
        self.assert_accepted(dict(method="mlm", label_smoothing_factor=0.1))

    def test_the_loop_error_is_the_datasets_own_message(self):
        with self.assertRaises(ValueError) as dataset_error:
            AntibodyFrameworkDataset._resolve_target("h9")
        groups = (mlm_arguments.ModelArguments(), mlm_arguments.DataArguments(target_region="h9"),
                  mlm_arguments.GenerationArguments())
        with self.assertRaises(ValueError) as early_error:
            validate_common_args(*groups, expected_method="mlm")
        self.assertEqual(str(early_error.exception), str(dataset_error.exception))

    def test_the_judge_context_is_checked_from_a_config_and_from_the_command_line(self):
        self.assert_rejected(dict(gen_judge_partner_context="both"), "gen_judge_partner_context")
        parser = _parser(mlm_arguments)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args_into_dataclasses(["--gen_judge_partner_context", "both"])


class ConfigWithOverridesTests(unittest.TestCase):
    def parse(self, config, *overrides):
        parser = _parser(mlm_arguments)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with patch("sys.argv", ["train.py", str(path), *overrides]):
                return parse_config_and_cli(parser)

    def test_an_override_does_not_change_how_list_and_dict_values_are_read(self):
        values = {
            "list": dict(label_names=["labels", "weights"]),
            "single-item list": dict(report_to=["none"]),
            "dict": dict(lr_scheduler_type="cosine_with_restarts",
                         lr_scheduler_kwargs={"num_cycles": 2}),
        }
        for kind, config in values.items():
            with self.subTest(kind=kind):
                alone = self.parse(config)[3]
                overridden = self.parse(config, "--seed", "7")[3]
                for name in config:
                    self.assertEqual(getattr(overridden, name), getattr(alone, name))
                self.assertEqual(overridden.seed, 7)

    def test_an_empty_list_cannot_be_expressed_as_an_override(self):
        self.parse(dict(label_names=[]))  # fine on its own
        with self.assertRaisesRegex(ValueError, "label_names: an empty list"):
            self.parse(dict(label_names=[]), "--seed", "7")

    def test_a_null_keeps_the_dataclass_default_and_the_override_still_applies(self):
        model, _, gen, training = self.parse(dict(gen_judge_path=None, intermediate_size=None,
                                                  hidden_size=64), "--seed", "7")
        self.assertIsNone(gen.gen_judge_path)
        self.assertIsNone(model.intermediate_size)
        self.assertEqual((model.hidden_size, training.seed), (64, 7))


class ShippedConfigTests(unittest.TestCase):
    def test_every_shipped_training_config_passes_its_methods_setup(self):
        """A config setup_args() rejects would waste the queue slot it was written for."""
        cases = _shipped_training_configs()
        self.assertTrue(set(METHODS) <= {config.get("method", DEFAULT_METHOD) for _, config in cases})
        for label, config in cases:
            with self.subTest(config=label):
                with _setup_args(config) as (setup_args, initialise):
                    setup_args()
                initialise.assert_called_once()


def _argument_classes():
    classes = [BaseModelArguments, BaseDataArguments, BaseGenerationArguments]
    for name in METHODS:
        module = resolve(name).module("arguments")
        classes += [module.ModelArguments, module.DataArguments, module.GenerationArguments]
    return classes


class ArgumentDeclarationTests(unittest.TestCase):
    def test_every_field_has_help_text(self):
        for cls in _argument_classes():
            for field in dataclasses.fields(cls):
                with self.subTest(field=f"{cls.__module__}.{cls.__name__}.{field.name}"):
                    self.assertTrue(field.metadata.get("help"))

    def test_a_field_that_defaults_to_none_is_declared_optional(self):
        for cls in _argument_classes():
            hints = typing.get_type_hints(cls)
            for field in dataclasses.fields(cls):
                if field.default is None:
                    with self.subTest(field=f"{cls.__module__}.{cls.__name__}.{field.name}"):
                        self.assertIn(type(None), typing.get_args(hints[field.name]))

    def test_the_partner_context_choices_are_defined_once(self):
        field = {f.name: f for f in dataclasses.fields(BaseGenerationArguments)}["gen_judge_partner_context"]
        self.assertEqual(field.metadata["choices"], PARTNER_CONTEXTS)
        self.assertIn(field.default, PARTNER_CONTEXTS)
        flags = {action.dest: action for action in evaluate.build_parser()._actions}
        for dest in ("partner_context", "judge_partner_context"):
            with self.subTest(flag=dest):
                self.assertEqual(tuple(flags[dest].choices), PARTNER_CONTEXTS)


if __name__ == "__main__":
    unittest.main()
