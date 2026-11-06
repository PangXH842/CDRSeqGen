"""Reject invalid inference options and contradictory EOS records before use."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import evaluate
import generate
from cdrseqgen.methods.common import arguments as shared_arguments
from cdrseqgen.methods.common import evaluation as shared_evaluation
from cdrseqgen.methods.common import generation as shared_generation
from cdrseqgen.methods.common.eos_decoding import _partition_attempts


class SamplingValidationTests(unittest.TestCase):
    def test_valid_boundaries_work_for_command_and_training_names(self):
        values = dict(num_samples=1, temperature=0.0, top_k=0, top_p=1.0,
                      length_temperature=0.0, num_examples=-1)
        for prefix in ("", "gen_"):
            with self.subTest(prefix=prefix):
                args = SimpleNamespace(**{prefix + key: value for key, value in values.items()})
                shared_arguments.validate_sampling_args(args, prefix=prefix)
        # The command parser's absent max_len means use the checkpoint's cap.
        shared_arguments.validate_sampling_args(SimpleNamespace(max_len=None, show_samples=0))
        shared_arguments.validate_sampling_args(SimpleNamespace())

    def test_invalid_numeric_values_name_the_rejected_field(self):
        invalid = {
            "num_samples": (0, -1, True, 1.5, None),
            "temperature": (-0.1, float("nan"), float("inf"), None),
            "length_temperature": (-0.1, float("nan"), float("inf"), None),
            "top_k": (-1, True, 1.5, None),
            "top_p": (0, -0.1, 1.1, float("nan"), float("inf"), None),
            "num_examples": (-2, True, 1.5, None),
        }
        for prefix in ("", "gen_"):
            for field, values in invalid.items():
                for value in values:
                    key = prefix + field
                    with self.subTest(field=key, value=value):
                        with self.assertRaisesRegex(ValueError, field):
                            shared_arguments.validate_sampling_args(
                                SimpleNamespace(**{key: value}), prefix=prefix)

    def test_invalid_command_caps_and_display_limits_are_rejected(self):
        for field, values in (("max_len", (0, -1, True, 2.5)),
                              ("show_samples", (-1, True, 2.5, None))):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        shared_arguments.validate_sampling_args(SimpleNamespace(**{field: value}))

    def test_generation_preflight_rejects_bad_parser_values_before_loading(self):
        settings = (("--temperature", "nan"), ("--top_p", "inf"),
                    ("--num_samples", "0"), ("--top_k", "-1"),
                    ("--length_temperature", "nan"), ("--max_len", "0"),
                    ("--show_samples", "-1"))
        for flag, value in settings:
            with self.subTest(flag=flag, value=value):
                args = generate.parse_args(["--model_dir", "/tmp/checkpoint", "--output", "none",
                                            flag, value])
                backbone = Mock()
                method = SimpleNamespace(name="mlm", module=Mock(return_value=SimpleNamespace(
                    resolve=Mock(return_value=backbone))))
                with patch.object(shared_generation, "resolve", return_value=method), \
                        patch.object(shared_generation, "load_test_cases", return_value=[{}]), \
                        patch.object(shared_generation, "check_compatibility", return_value=([], [])), \
                        patch("builtins.print"):
                    with self.assertRaises((ValueError, SystemExit)) as caught:
                        shared_generation.run_generation_pipeline(
                            "mlm", args, {"source": "run_config", "model_type": "scratch", "max_len": 64},
                            prepare_generation=Mock(), generation_state=Mock(), generate_case=Mock())
                self.assertIn(flag[2:], str(caught.exception))
                backbone.build_tokenizer.assert_not_called()
                backbone.load_pretrained_model.assert_not_called()

    def test_evaluation_preflight_rejects_bad_parser_values_before_resources(self):
        settings = (("--temperature", "inf"), ("--top_p", "nan"),
                    ("--num_examples", "-2"), ("--num_samples", "0"),
                    ("--length_temperature", "nan"))
        for flag, value in settings:
            with self.subTest(flag=flag, value=value):
                args = evaluate.parse_args(["--model_dir", "/tmp/checkpoint", "--config", "/tmp/config.json",
                                            "--output", "none", flag, value])
                method = SimpleNamespace(name="mlm", module=Mock(side_effect=AssertionError(
                    "method resources must not be requested before numeric preflight")))
                with patch.object(shared_evaluation, "resolve", return_value=method):
                    with self.assertRaises((ValueError, SystemExit)) as caught:
                        shared_evaluation.run_evaluation_pipeline(
                            "mlm", args, {"method": "mlm"}, None,
                            prepare_evaluation=Mock(), evaluation_options=Mock(), evaluate=Mock())
                self.assertIn(flag[2:], str(caught.exception))
                method.module.assert_not_called()


class EosDrawContractTests(unittest.TestCase):
    def test_legacy_and_native_records_keep_disjoint_nonnegative_counts(self):
        results = [dict(loop="AC", complete=True), dict(loop="A", complete=False),
                   dict(loop="", complete=True),
                   dict(loop="", complete=False, valid=False, terminated=True),
                   dict(loop="AC", complete=True, valid=True, terminated=True)]
        kept, counts = _partition_attempts(results)
        self.assertEqual([record["loop"] for record in kept], ["AC", "AC"])
        self.assertEqual(counts, dict(attempts=5, complete_nonempty=2, complete_empty=1,
                                     incomplete=1, invalid=1))
        self.assertEqual(sum(counts[key] for key in ("complete_nonempty", "complete_empty", "incomplete", "invalid")), 5)
        self.assertTrue(all(value >= 0 for value in counts.values()))

    def test_contradictory_or_malformed_sampler_records_raise_clear_errors(self):
        invalid = [dict(loop="AC", complete=True, valid=False, terminated=True),
                   dict(loop="AC", complete=False, valid=True),
                   dict(loop="AC", complete=True, terminated=False),
                   dict(loop="AC", complete="yes"), dict(loop="AC", complete=True, valid=1),
                   dict(loop=["A", "C"], complete=True), dict(loop="AC"), dict(complete=True)]
        for record in invalid:
            with self.subTest(record=record):
                with self.assertRaises(ValueError) as caught:
                    _partition_attempts([record])
                self.assertTrue(str(caught.exception))


if __name__ == "__main__":
    unittest.main()
