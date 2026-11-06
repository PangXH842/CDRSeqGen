"""EOS command contracts shared by AR/S2S, without models or inference."""
import importlib
from types import SimpleNamespace
import unittest

from cdrseqgen.datasets.dataset_ar import CAUSAL_DECODE_STEPS
from cdrseqgen.datasets.dataset_s2s import S2S_DECODE_STEPS
from cdrseqgen.methods.ar.models.iglm import resolve_conditioning


METHODS = (("ar", "causal", CAUSAL_DECODE_STEPS),
           ("s2s", "seq2seq", S2S_DECODE_STEPS))


def arguments(**overrides):
    values = dict(decoding=None, length_mode=None, min_length=None, max_length=None,
                  target_length=None, use_native_length=None, length_temperature=1.0,
                  max_new_tokens=None, target_loop=None)
    return SimpleNamespace(**(values | overrides))


def module(method, kind):
    return importlib.import_module(f"cdrseqgen.methods.{method}.{kind}")


def prepare(method, kind, args, config, cases=None):
    if kind == "generation":
        return module(method, kind).prepare_generation(args, [{}] if cases is None else cases, config)
    return module(method, kind).prepare_evaluation(args, config)


class EosCommandSetupTests(unittest.TestCase):
    def test_modes_and_decode_cap_precedence_through_both_commands(self):
        settings = ((7, {"gen_max_new_tokens": 9}, 7),
                    (None, {"gen_max_new_tokens": 9}, 9), (None, {}, None))
        for method, decoder, default in METHODS:
            for kind in ("generation", "evaluation"):
                for requested, config, expected in settings:
                    with self.subTest(method=method, kind=kind, requested=requested, config=config):
                        args = arguments(max_new_tokens=requested)
                        self.assertIsNone(prepare(method, kind, args, config))
                        self.assertEqual(args.max_new_tokens, default if expected is None else expected)
                        self.assertEqual(args.decoding, decoder)
                        if kind == "evaluation":
                            self.assertEqual(args.length_mode, "inferred")
                            self.assertEqual(module(method, kind).evaluation_options(args, "cpu"),
                                             {"max_new_tokens": args.max_new_tokens})

    def test_invalid_explicit_saved_and_null_caps_still_fail(self):
        for method, _, default in METHODS:
            for kind in ("generation", "evaluation"):
                for value in (0, -1, True, 1.5, default + 1, None):
                    configs = [(None, {"gen_max_new_tokens": value})]
                    if value is not None:
                        configs.append((value, {"gen_max_new_tokens": 8}))
                    for requested, config in configs:
                        with self.subTest(method=method, kind=kind, value=value, requested=requested):
                            args = arguments(max_new_tokens=requested)
                            with self.assertRaisesRegex(ValueError, "max_new_tokens"):
                                prepare(method, kind, args, config)

    def test_generation_accepts_missing_decode_cap_attribute(self):
        for method, _, default in METHODS:
            args = arguments()
            del args.max_new_tokens
            prepare(method, "generation", args, {})
            self.assertEqual(args.max_new_tokens, default)

    def test_case_length_controls_reject_zero_and_report_the_exact_key(self):
        for method, _, _ in METHODS:
            for key in ("min_length", "max_length", "target_length", "use_native_length"):
                for value in (0, 8, True):
                    with self.subTest(method=method, key=key, value=value):
                        with self.assertRaises(ValueError) as caught:
                            prepare(method, "generation", arguments(), {}, [{key: value}])
                        self.assertEqual(str(caught.exception),
                                         f"{method.upper()} chooses length via EOS; {key} is unsupported")

    def test_case_none_or_false_overrides_cli_target_length_controls(self):
        for method, _, _ in METHODS:
            for key in ("target_length", "use_native_length"):
                for value in (None, False):
                    with self.subTest(method=method, key=key, value=value):
                        args = arguments(**{key: 8})
                        prepare(method, "generation", args, {}, [{key: value}])
                        with self.assertRaisesRegex(ValueError, key):
                            prepare(method, "generation", arguments(**{key: 8}), {}, [{}])

    def test_evaluation_rejects_native_length_and_incompatible_decoder(self):
        for method, _, _ in METHODS:
            for overrides in ({"length_mode": "fixed_native"}, {"decoding": "independent"},
                              {"min_length": 8}, {"length_temperature": 2.0}):
                with self.subTest(method=method, overrides=overrides):
                    with self.assertRaises(ValueError):
                        prepare(method, "evaluation", arguments(**overrides), {})

    def test_ar_target_region_still_controls_native_iglm_chain(self):
        for requested, expected, chain in ((None, "h3", "[HEAVY]"), ("l3", "l3", "[LIGHT]")):
            args = arguments(target_loop=requested)
            prepare("ar", "generation", args, {"target_region": "h3"})
            self.assertEqual(args.target_region, expected)
            self.assertEqual(resolve_conditioning({"target_region": args.target_region})["chain_token"], chain)

    def test_self_scorer_keeps_legacy_import_and_shared_command_ownership(self):
        for method, _, _ in METHODS:
            sampling = module(method, "sampling")
            self.assertIs(module(method, "evaluation").perplexities, sampling.perplexities)
            self.assertIs(module(method, "generation").perplexities, sampling.perplexities)
            self.assertEqual(sampling.perplexities.__module__, sampling.__name__)


if __name__ == "__main__":
    unittest.main()
