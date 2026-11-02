"""Custom-model builders reject invalid architectures before allocating or running a model.

The direct Python builders are public entry points too: CLI validation alone does
not protect callers building a custom model or loading a saved architecture.
"""
import unittest

import torch

from cdrseqgen.methods.ar.models.transformer import (
    CausalAntibodyConfig, CausalAntibodyModel, build_model as build_ar,
)
from cdrseqgen.methods.mlm.models.transformer import (
    CustomAntibodyConfig, CustomAntibodyModel, build_model as build_mlm,
)
from cdrseqgen.methods.s2s.models.transformer import (
    Seq2SeqAntibodyConfig, Seq2SeqAntibodyModel, build_model as build_s2s,
)


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


if __name__ == "__main__":
    unittest.main()
