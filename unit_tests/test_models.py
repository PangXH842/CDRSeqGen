"""Tests merged from: test_custom_model_validation.py, test_model_roundtrip.py, test_common_backbones.py, test_shared_contracts.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import unittest
import torch
from cdrseqgen.methods.ar.models.transformer import (
    CausalAntibodyConfig,
    CausalAntibodyModel,
    build_model as build_ar,
)
from cdrseqgen.methods.mlm.models.transformer import (
    CustomAntibodyConfig,
    CustomAntibodyModel,
    build_model as build_mlm,
)
from cdrseqgen.methods.s2s.models.transformer import (
    Seq2SeqAntibodyConfig,
    Seq2SeqAntibodyModel,
    build_model as build_s2s,
)
import json
from pathlib import Path
import tempfile
from cdrseqgen.datasets.dataset_ar import CausalInfillingDataset, causal_collate_fn
from cdrseqgen.datasets.dataset_mlm import MaskedInfillingDataset, mlm_collate_fn
from cdrseqgen.datasets.dataset_s2s import Seq2SeqInfillingDataset, s2s_collate_fn
from cdrseqgen.methods.ar.backbones import resolve as resolve_ar
from cdrseqgen.methods.mlm.backbones import resolve as resolve_mlm
from cdrseqgen.methods.s2s.backbones import resolve as resolve_s2s
from unittest.mock import Mock, patch
from cdrseqgen.methods.common.backbones import (
    Backbone,
    BackboneRegistry,
    bert_tokenizer,
    esm_tokenizer,
)
from cdrseqgen.methods.registry import resolve as resolve_method
import contextlib
import dataclasses
import io
from types import SimpleNamespace
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


# ====================================================================================================
# from test_custom_model_validation.py
# Custom-model builders reject invalid architectures before allocating or running a model.
#
# The direct Python builders are public entry points too: CLI validation alone does
# not protect callers building a custom model or loading a saved architecture.
# ====================================================================================================

BUILDERS = {"mlm": build_mlm, "ar": build_ar, "s2s": build_s2s}
CONSTRUCTORS = {
    "mlm": (CustomAntibodyConfig, CustomAntibodyModel),
    "ar": (CausalAntibodyConfig, CausalAntibodyModel),
    "s2s": (Seq2SeqAntibodyConfig, Seq2SeqAntibodyModel),
}
SMALL_MODEL = dict(hidden_size=32, num_attention_heads=4, num_layers=1, max_len=32)
SMALL_CONFIG = dict(hidden_size=32, num_attention_heads=4, num_hidden_layers=1,
                    intermediate_size=64, max_position_embeddings=32)

class DirectBuilderValidationTests(unittest.TestCase):
    def test_architecture_counts_are_positive_integers_in_every_method(self):
        fields = ("vocab_size", "hidden_size", "num_layers", "num_attention_heads",
                  "intermediate_size", "max_len")
        for method, build in BUILDERS.items():
            for field in fields:
                for invalid in (0, -1, True, 1.5):
                    with self.subTest(method=method, field=field, value=invalid):
                        options = {**SMALL_MODEL, field: invalid}
                        with self.assertRaises(ValueError):
                            build(**options)

    def test_seq2seq_encoder_and_decoder_overrides_are_positive_integers(self):
        for field in ("num_encoder_layers", "num_decoder_layers"):
            for invalid in (0, -1, True, 1.5):
                with self.subTest(field=field, value=invalid):
                    with self.assertRaises(ValueError):
                        build_s2s(**SMALL_MODEL, **{field: invalid})

    def test_attention_heads_must_divide_the_width_in_every_method(self):
        for method, build in BUILDERS.items():
            with self.subTest(method=method), self.assertRaises(ValueError):
                build(**{**SMALL_MODEL, "hidden_size": 30, "num_attention_heads": 4})

    def test_rotary_models_reject_odd_head_width_before_the_forward_pass(self):
        # The old AR builder accepted heads of width 15 and crashed when RoPE
        # produced 16 channels on the first forward call.
        for method in ("ar", "s2s"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                BUILDERS[method](**{**SMALL_MODEL, "hidden_size": 30,
                                    "num_attention_heads": 2})

    def test_mlm_allows_odd_head_width_because_its_positions_are_absolute(self):
        with torch.random.fork_rng(devices=[]):
            model = build_mlm(**{**SMALL_MODEL, "hidden_size": 30,
                                 "num_attention_heads": 2}).eval()
            with torch.no_grad():
                outputs = model(torch.tensor([[5, 6, 7]]))
        self.assertEqual(tuple(outputs["logits"].shape), (1, 3, model.config.vocab_size))
        self.assertTrue(torch.isfinite(outputs["logits"]).all())

    def test_ffn_default_and_explicit_override_are_preserved(self):
        for method, build in BUILDERS.items():
            with self.subTest(method=method), torch.random.fork_rng(devices=[]):
                default = build(**SMALL_MODEL)
                explicit = build(**SMALL_MODEL, intermediate_size=48)
            self.assertEqual(default.config.intermediate_size, 4 * SMALL_MODEL["hidden_size"])
            self.assertEqual(explicit.config.intermediate_size, 48)


class DirectModelValidationTests(unittest.TestCase):
    def test_model_construction_rejects_invalid_dimensions_in_supplied_configs(self):
        # Checkpoint loading and direct Python users instantiate Model(Config(...))
        # without going through build_model or the CLI. Keep that boundary safe too.
        fields = ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                  "intermediate_size", "max_position_embeddings")
        for method, (config_class, model_class) in CONSTRUCTORS.items():
            for field in fields:
                for invalid in (0, -1, True, 1.5):
                    with self.subTest(method=method, field=field, value=invalid):
                        config = config_class(**{**SMALL_CONFIG, field: invalid})
                        with self.assertRaises(ValueError):
                            model_class(config)

    def test_direct_seq2seq_construction_checks_decoder_depth(self):
        for invalid in (0, -1, True, 1.5):
            with self.subTest(value=invalid):
                config = Seq2SeqAntibodyConfig(**SMALL_CONFIG, num_decoder_layers=invalid)
                with self.assertRaises(ValueError):
                    Seq2SeqAntibodyModel(config)

    def test_direct_rotary_construction_rejects_odd_head_width(self):
        for method in ("ar", "s2s"):
            config_class, model_class = CONSTRUCTORS[method]
            with self.subTest(method=method):
                config = config_class(**{**SMALL_CONFIG, "hidden_size": 30,
                                         "num_attention_heads": 2})
                with self.assertRaises(ValueError):
                    model_class(config)


# ====================================================================================================
# from test_model_roundtrip.py
# Real CPU models: processed data -> padded batch -> update -> checkpoint -> identical logits.
#
# No pretrained downloads or mocked model/optimizer/serializer. This is a plumbing check, not evidence
# that a one-step model learns antibody biology. Full training and external PLMs need separate runs.
# ====================================================================================================

SINGLE_STREAM = ('input_ids', 'attention_mask', 'labels')
TWO_STREAM = SINGLE_STREAM + ('decoder_input_ids', 'decoder_attention_mask')

class CheckpointRoundTripTests(unittest.TestCase):
    def check_roundtrip(self, resolve, dataset_class, collate, batch_keys=SINGLE_STREAM,
                        pad_check_key='attention_mask'):
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            root = Path(directory)
            records = []
            for loop in ('GGG', 'GSGGG'):
                chain = dict(fwr1='AAAA', cdr1='CC', fwr2='DDDD', cdr2='EE',
                             fwr3='FFFF', cdr3=loop, fwr4='HHHH')
                chain['full_aa'] = ''.join(chain.values())
                records.append({'heavy': chain, 'light': chain, 'split': 'train'})
            path = root / 'data.json'
            path.write_text(json.dumps(records))
            backbone = resolve('scratch')
            tokenizer = backbone.build_tokenizer('scratch')
            model = backbone.load_training_model('scratch', vocab_size=tokenizer.vocab_size,
                                                hidden_size=32, num_layers=1, num_attention_heads=4)
            dataset = dataset_class(str(path), tokenizer, target_region='h3')
            collated = collate([dataset[i] for i in range(len(dataset))], pad_token_id=tokenizer.pad_id)
            batch = {k: collated[k] for k in batch_keys}
            # Which stream pads is itself a structural fact. The fixture's two records share a
            # framework and differ only in loop length, so a single-stream method pads its one
            # sequence while the seq2seq encoder — which never carries the loop — pads nothing and
            # its decoder does. Asserting the wrong key would silently stop exercising padding.
            self.assertTrue((batch[pad_check_key] == 0).any(), 'fixture must exercise padding')
            initial = {k: v.detach().clone() for k, v in model.named_parameters()}
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            model.train()
            loss = model(**batch)['loss']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
            self.assertTrue(any(not torch.equal(initial[k], v) for k, v in model.named_parameters()),
                            'an optimizer step must change learned weights')
            model.eval()
            with torch.no_grad():
                expected = model(**batch)['logits']
            checkpoint = root / 'checkpoint'
            model.save_pretrained(checkpoint)
            # A fresh RNG state must not influence a correctly restored checkpoint.
            torch.manual_seed(999)
            restored = backbone.load_pretrained_model(str(checkpoint))
            with torch.no_grad():
                actual = restored(**batch)['logits']
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_masked_model_updates_and_reloads(self):
        self.check_roundtrip(resolve_mlm, MaskedInfillingDataset, mlm_collate_fn)

    def test_causal_model_updates_and_reloads(self):
        self.check_roundtrip(resolve_ar, CausalInfillingDataset, causal_collate_fn)

    def test_seq2seq_model_updates_and_reloads(self):
        # Two streams, so the batch carries decoder inputs the single-stream methods do not have.
        self.check_roundtrip(resolve_s2s, Seq2SeqInfillingDataset, s2s_collate_fn,
                             batch_keys=TWO_STREAM, pad_check_key='decoder_attention_mask')


# ====================================================================================================
# from test_common_backbones.py
# Shared backbone dispatch must preserve method isolation and tokenizer provenance.
# ====================================================================================================

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


# ====================================================================================================
# from test_shared_contracts.py
# Contracts of the small shared helpers behind the MLM/common presentation cleanup.
#
#   * MLM decoder names are validated wherever a run can name one (training config, standalone
#     evaluation, standalone generation, the sampler itself) instead of falling back to iterative.
#   * Both MLM decoders run without autograd.
#   * anchor_trim validation lives in one place.
#   * The context-regime helpers behind the generative-evaluation payload keep their key order.
#   * evaluate.py and generate.py define their shared flags once, those flags mirror the
#     training-time Generation fields, and each command's public flag set cannot change silently.
# ====================================================================================================

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
