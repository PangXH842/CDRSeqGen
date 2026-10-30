"""Shared backbone dispatch must preserve method isolation and tokenizer provenance."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cdrseqgen.methods.common.backbones import (
    Backbone, BackboneRegistry, bert_tokenizer, esm_tokenizer,
)
from cdrseqgen.methods.registry import resolve as resolve_method


def descriptor(name, *, matches=None, fallback=False):
    return Backbone(
        id=name,
        matches=matches or (lambda model_type: model_type == name),
        position_type="relative",
        uses_special_tokens=True,
        load_tokenizer=Mock(),
        load_training_model=Mock(),
        load_pretrained_model=Mock(),
        fallback=fallback,
    )


class RegistryTests(unittest.TestCase):
    def test_first_matching_family_wins_without_loading_resources(self):
        registry = BackboneRegistry("test", require_fallback=True)
        named = descriptor("named", matches=lambda name: name == "org/model")
        fallback = descriptor("generic", matches=lambda name: "/" in name, fallback=True)
        self.assertIs(registry.register(named), named)
        registry.register(fallback)
        registry.validate()
        self.assertIs(registry.resolve("org/model"), named)
        self.assertIs(registry.resolve("other/model"), fallback)
        for entry in (named, fallback):
            entry.load_tokenizer.assert_not_called()
            entry.load_training_model.assert_not_called()
            entry.load_pretrained_model.assert_not_called()

    def test_registries_keep_their_own_scratch_definitions(self):
        first, second = BackboneRegistry("first"), BackboneRegistry("second")
        first_scratch, second_scratch = descriptor("scratch"), descriptor("scratch")
        first.register(first_scratch)
        second.register(second_scratch)
        first.register(descriptor("only_first"))
        self.assertIs(first.resolve("scratch"), first_scratch)
        self.assertIs(second.resolve("scratch"), second_scratch)
        with self.assertRaisesRegex(ValueError, "second backbone registered"):
            second.resolve("only_first")

    def test_unknown_models_report_the_method_supported_entries_and_hint(self):
        registry = BackboneRegistry("s2s", unsupported_hint="Requires its own adapter.")
        registry.register(descriptor("scratch"))
        with self.assertRaises(ValueError) as error:
            registry.resolve("unsupported/model")
        message = str(error.exception)
        for text in ("s2s", "unsupported/model", "scratch", "own adapter"):
            self.assertIn(text, message)

    def test_missing_fallback_is_allowed_only_when_optional(self):
        for required in (False, True):
            with self.subTest(required=required):
                registry = BackboneRegistry("test", require_fallback=required)
                registry.register(descriptor("scratch"))
                if required:
                    with self.assertRaisesRegex(ValueError, "one generic fallback"):
                        registry.validate()
                else:
                    registry.validate()

    def test_duplicate_ids_are_rejected_even_when_matching_different_models(self):
        for require_fallback in (False, True):
            with self.subTest(require_fallback=require_fallback):
                registry = BackboneRegistry("test", require_fallback=require_fallback)
                registry.register(descriptor("family", matches=lambda name: name == "first"))
                registry.register(descriptor("family", matches=lambda name: name == "second",
                                             fallback=require_fallback))
                with self.assertRaisesRegex(ValueError, "test: duplicate backbone id 'family'"):
                    registry.validate()

    def test_misplaced_and_multiple_fallbacks_are_rejected(self):
        for required in (False, True):
            for flags in ((True, False), (True, True), (False, True, True)):
                with self.subTest(required=required, flags=flags):
                    registry = BackboneRegistry("test", require_fallback=required)
                    for i, fallback in enumerate(flags):
                        registry.register(descriptor(str(i), fallback=fallback))
                    with self.assertRaisesRegex(ValueError, "registered last"):
                        registry.validate()

    def test_each_method_retains_its_dispatch_and_fallback_policy(self):
        cases = {
            "mlm": [("scratch", "scratch"), ("facebook/esm2_t6", "esm2"),
                    ("Exscientia/IgBert", "protein_bert"), ("other/model", "hf_encoder")],
            "ar": [("scratch", "scratch"), ("other/model", "hf_decoder")],
            "s2s": [("scratch", "scratch"), ("facebook/esm2_t6", "esm2"),
                    ("Exscientia/IgBert", "protein_bert")],
        }
        scratch_entries = []
        for method, models in cases.items():
            module = resolve_method(method).module("backbones")
            scratch_entries.append(module.resolve("scratch"))
            for model_type, expected in models:
                with self.subTest(method=method, model_type=model_type):
                    self.assertEqual(module.resolve(model_type).id, expected)
        self.assertEqual(len({id(entry) for entry in scratch_entries}), 3)
        with self.assertRaisesRegex(ValueError, "own adapter"):
            resolve_method("s2s").module("backbones").resolve("other/model")


class TokenizerSourceTests(unittest.TestCase):
    def test_saved_vocabulary_is_used_with_the_registered_layout_conventions(self):
        backbone = descriptor("family")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "tokenizer_config.json").write_text("{}")
            result = backbone.build_tokenizer("original/model", checkpoint_path=checkpoint)
        backbone.load_tokenizer.assert_called_once_with(
            str(checkpoint), "relative", True
        )
        self.assertIs(result, backbone.load_tokenizer.return_value)

    def test_legacy_checkpoint_without_tokenizer_files_uses_the_original_source(self):
        backbone = descriptor("family")
        with tempfile.TemporaryDirectory() as directory:
            backbone.build_tokenizer("original/model", checkpoint_path=directory)
        backbone.load_tokenizer.assert_called_once_with("original/model", "relative", True)


class ProteinTokenizerTests(unittest.TestCase):
    def test_helpers_forward_named_layout_options_and_bert_residue_spacing(self):
        for helper, loader, options in (
            (esm_tokenizer, "EsmTokenizer", {}),
            (bert_tokenizer, "AutoTokenizer", {"space_residues": True}),
        ):
            with self.subTest(helper=helper.__name__), \
                    patch(f"transformers.{loader}.from_pretrained") as load, \
                    patch("cdrseqgen.methods.common.backbones.TokenizerAdapter.from_hf") as adapt:
                result = helper(model_type="local/model", position_type="relative",
                                uses_special_tokens=False)
                load.assert_called_once_with(
                    "local/model", **({"use_fast": False} if helper is bert_tokenizer else {})
                )
                adapt.assert_called_once_with(
                    load.return_value, position_type="relative", uses_special_tokens=False,
                    **options,
                )
                self.assertIs(result, adapt.return_value)

    def test_auto_tokenizer_compatibility_errors_use_the_bert_fallback(self):
        for error in (ValueError("missing metadata"), KeyError("model_type"),
                      ImportError("optional tokenizer dependency")):
            with self.subTest(error=type(error).__name__), \
                    patch("transformers.AutoTokenizer.from_pretrained", side_effect=error), \
                    patch("transformers.BertTokenizer.from_pretrained") as fallback, \
                    patch("cdrseqgen.methods.common.backbones.TokenizerAdapter.from_hf") as adapt:
                result = bert_tokenizer("legacy/model", "absolute", True)
                fallback.assert_called_once_with("legacy/model")
                adapt.assert_called_once_with(
                    fallback.return_value, position_type="absolute",
                    uses_special_tokens=True, space_residues=True,
                )
                self.assertIs(result, adapt.return_value)

    def test_io_and_unexpected_errors_propagate_without_retrying_another_tokenizer(self):
        for error in (PermissionError("denied"), FileNotFoundError("missing"),
                      OSError("download failed"), RuntimeError("unexpected failure")):
            with self.subTest(error=type(error).__name__), \
                    patch("transformers.AutoTokenizer.from_pretrained", side_effect=error), \
                    patch("transformers.BertTokenizer.from_pretrained") as fallback:
                with self.assertRaises(type(error)) as raised:
                    bert_tokenizer("broken/model", "absolute", True)
                self.assertIs(raised.exception, error)
                fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
