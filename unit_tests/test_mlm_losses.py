import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from transformers import TrainingArguments

from cdrseqgen.methods.mlm import losses
from cdrseqgen.methods.mlm.losses import (
    LossConfig,
    MLMLossTrainer,
    calculate_mlm_loss,
    normalize_loss_type,
)


class MLMLossTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [[[3.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]]]
        )
        self.labels = torch.tensor([[0, 1, 1, 0, 0, 1]])
        self.offsets = torch.arange(6).unsqueeze(0)
        self.lengths = torch.tensor([6])

    def test_cross_entropy_reproduces_torch_default(self):
        config = LossConfig("cross_entropy")
        got = calculate_mlm_loss(self.logits, self.labels, config=config)
        expected = F.cross_entropy(self.logits.reshape(-1, 2), self.labels.reshape(-1))
        self.assertTrue(torch.allclose(got, expected))

    def test_unit_weighted_loss_reproduces_cross_entropy(self):
        config = LossConfig("weighted_cross_entropy", core_weight=1.0, anchor_weight=1.0)
        got = calculate_mlm_loss(
            self.logits,
            self.labels,
            config=config,
            target_offsets=self.offsets,
            target_lengths=self.lengths,
        )
        expected = F.cross_entropy(self.logits.reshape(-1, 2), self.labels.reshape(-1))
        self.assertTrue(torch.allclose(got, expected))

    def test_core_and_anchor_weights_are_applied_by_original_target_offset(self):
        config = LossConfig("weighted_cross_entropy", core_weight=3.0, anchor_weight=1.0)
        got = calculate_mlm_loss(
            self.logits,
            self.labels,
            config=config,
            target_offsets=self.offsets,
            target_lengths=self.lengths,
        )
        token_losses = F.cross_entropy(
            self.logits.reshape(-1, 2), self.labels.reshape(-1), reduction="none"
        )
        # trim=2 on a six-residue loop: offsets 2 and 3 are core.
        weights = torch.tensor([1.0, 1.0, 3.0, 3.0, 1.0, 1.0])
        expected = (token_losses * weights).sum() / weights.sum()
        self.assertTrue(torch.allclose(got, expected))

    def test_mixed_masking_uses_offsets_not_supervised_position_order(self):
        labels = self.labels.clone()
        labels[0, [0, 3, 5]] = -100
        config = LossConfig("weighted_cross_entropy", core_weight=4.0, anchor_weight=1.0)
        got = calculate_mlm_loss(
            self.logits,
            labels,
            config=config,
            target_offsets=self.offsets,
            target_lengths=self.lengths,
        )
        token_losses = F.cross_entropy(
            self.logits.reshape(-1, 2), labels.reshape(-1), reduction="none", ignore_index=-100
        )
        # Active offsets are 1 (anchor), 2 (core), and 4 (anchor).
        expected = (token_losses * torch.tensor([0.0, 1.0, 4.0, 0.0, 1.0, 0.0])).sum() / 6
        self.assertTrue(torch.allclose(got, expected))

    def test_short_loops_are_all_anchor(self):
        logits = self.logits[:, :4]
        labels = self.labels[:, :4]
        config = LossConfig("weighted_cross_entropy", core_weight=9.0, anchor_weight=1.0)
        got = calculate_mlm_loss(
            logits,
            labels,
            config=config,
            target_offsets=self.offsets[:, :4],
            target_lengths=torch.tensor([4]),
        )
        expected = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        self.assertTrue(torch.allclose(got, expected))

    def _subset_cross_entropy(self, positions):
        """Cross-entropy over a chosen subset of the fixture's six positions."""
        return F.cross_entropy(
            self.logits[0, positions], self.labels[0, positions]
        )

    def test_zero_anchor_weight_supervises_the_core_only(self):
        # offsets 0..5 at trim 2 => anchors {0, 1, 4, 5}, core {2, 3}.
        config = LossConfig("weighted_cross_entropy", core_weight=1.0, anchor_weight=0.0)
        got = calculate_mlm_loss(
            self.logits,
            self.labels,
            config=config,
            target_offsets=self.offsets,
            target_lengths=self.lengths,
        )
        self.assertTrue(torch.allclose(got, self._subset_cross_entropy([2, 3])))

    def test_zero_core_weight_supervises_the_anchors_only(self):
        config = LossConfig("weighted_cross_entropy", core_weight=0.0, anchor_weight=1.0)
        got = calculate_mlm_loss(
            self.logits,
            self.labels,
            config=config,
            target_offsets=self.offsets,
            target_lengths=self.lengths,
        )
        self.assertTrue(torch.allclose(got, self._subset_cross_entropy([0, 1, 4, 5])))

    def test_batch_with_no_supervised_weight_raises_instead_of_returning_nan(self):
        # A length-4 loop is all anchor, so core-only weighting leaves nothing to supervise.
        # Dividing by a zero total weight would emit a NaN loss that trains silently to garbage.
        config = LossConfig("weighted_cross_entropy", core_weight=1.0, anchor_weight=0.0)
        with self.assertRaisesRegex(ValueError, "has weight zero"):
            calculate_mlm_loss(
                self.logits[:, :4],
                self.labels[:, :4],
                config=config,
                target_offsets=self.offsets[:, :4],
                target_lengths=torch.tensor([4]),
            )

    def test_aliases_and_invalid_configuration(self):
        self.assertEqual(normalize_loss_type("weighted"), "weighted_cross_entropy")
        with self.assertRaisesRegex(ValueError, "loss_type"):
            normalize_loss_type("dice")
        with self.assertRaisesRegex(ValueError, "cannot both be zero"):
            LossConfig("weighted_cross_entropy", core_weight=0.0, anchor_weight=0.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            LossConfig("weighted_cross_entropy", core_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            LossConfig(anchor_trim=1.5)

    def test_trainer_removes_loss_metadata_before_model_forward(self):
        class MetadataRejectingModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = logits
                self.received = None

            def forward(self, input_ids, attention_mask):
                self.received = (input_ids, attention_mask)
                return {"logits": self.logits}

        config = LossConfig("weighted_cross_entropy", core_weight=3.0, anchor_weight=1.0)
        trainer = object.__new__(MLMLossTrainer)
        trainer.loss_config = config
        model = MetadataRejectingModel(self.logits)
        inputs = {
            "input_ids": torch.zeros_like(self.labels),
            "attention_mask": torch.ones_like(self.labels),
            "labels": self.labels,
            "target_offsets": self.offsets,
            "target_lengths": self.lengths,
        }
        got, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
        expected = calculate_mlm_loss(
            self.logits,
            self.labels,
            config=config,
            target_offsets=self.offsets,
            target_lengths=self.lengths,
        )
        self.assertTrue(torch.allclose(got, expected))
        self.assertIs(outputs["logits"], self.logits)
        self.assertIsNotNone(model.received)

    def test_label_smoothing_ignores_padding_and_preserves_gradients(self):
        labels = self.labels.clone()
        labels[0, [0, 3, 5]] = -100
        logits = self.logits.clone().requires_grad_()
        expected_logits = self.logits.clone().requires_grad_()
        config = LossConfig("weighted", core_weight=4, label_smoothing=0.2)
        actual = calculate_mlm_loss(
            logits, labels, config=config,
            target_offsets=self.offsets, target_lengths=self.lengths,
        )
        # Selected loop offsets are anchor 1, core 2, and anchor 4.
        expected_losses = F.cross_entropy(
            expected_logits[0, [1, 2, 4]], labels[0, [1, 2, 4]],
            label_smoothing=0.2, reduction="none",
        )
        expected = (expected_losses * torch.tensor([1., 4., 1.])).sum() / 6
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(logits.grad, expected_logits.grad)
        self.assertEqual(torch.count_nonzero(logits.grad[0, [0, 3, 5]]).item(), 0)

    def test_mixed_mask_can_have_zero_weight_even_when_loop_has_a_core(self):
        labels = self.labels.clone()
        labels[0, [0, 1, 4, 5]] = -100  # only the core is supervised
        with self.assertRaisesRegex(ValueError, "has weight zero") as error:
            calculate_mlm_loss(
                self.logits, labels,
                config=LossConfig("weighted", core_weight=0, anchor_weight=1),
                target_offsets=self.offsets, target_lengths=self.lengths,
            )
        self.assertNotIn("there is no core", str(error.exception))

    def test_nonfinite_weights_are_rejected(self):
        for field in ("core_weight", "anchor_weight"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        LossConfig(**{field: value})

    def test_invalid_batch_or_target_metadata_is_rejected(self):
        config = LossConfig("weighted")
        cases = [
            ({"logits": self.logits[0]}, "expected logits"),
            ({"labels": torch.full_like(self.labels, -100)}, "no supervised tokens"),
            ({"target_offsets": None}, "requires target_offsets"),
            ({"target_offsets": self.offsets[:, :3]}, "target_offsets shape"),
            ({"target_lengths": self.lengths[:, None]}, "one length per batch row"),
            ({"target_offsets": torch.full_like(self.offsets, -1)}, "invalid target offsets"),
            ({"target_offsets": torch.full_like(self.offsets, 6)}, "invalid target offsets"),
        ]
        for changes, message in cases:
            with self.subTest(changes=tuple(changes)):
                kwargs = dict(logits=self.logits, labels=self.labels, config=config,
                              target_offsets=self.offsets, target_lengths=self.lengths)
                kwargs.update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    calculate_mlm_loss(**kwargs)

    def test_trainer_preserves_output_formats_and_does_not_mutate_batch(self):
        trainer = object.__new__(MLMLossTrainer)
        trainer.loss_config = LossConfig()
        inputs = dict(input_ids=torch.zeros_like(self.labels), labels=self.labels,
                      target_offsets=self.offsets, target_lengths=self.lengths)
        before = {key: value.clone() for key, value in inputs.items()}
        expected = calculate_mlm_loss(self.logits, self.labels, config=LossConfig())
        for outputs in ({"logits": self.logits}, SimpleNamespace(logits=self.logits),
                        (self.logits,)):
            with self.subTest(output_type=type(outputs).__name__):
                model = Mock(return_value=outputs)
                actual, returned = trainer.compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=999,
                )
                torch.testing.assert_close(actual, expected)
                self.assertIs(returned, outputs)
                self.assertEqual(set(model.call_args.kwargs), {"input_ids"})
                self.assertEqual(set(inputs), set(before))
                for key in inputs:
                    torch.testing.assert_close(inputs[key], before[key])

    def test_real_trainer_constructor_and_prediction_step_keep_loss_contract(self):
        class Model(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.predictions = torch.nn.Parameter(logits.clone())

            def forward(self, input_ids, labels=None):
                if labels is not None:
                    raise AssertionError("loss labels must not reach the model")
                return {"logits": self.predictions}

        with tempfile.TemporaryDirectory() as directory:
            model = Model(self.logits)
            args = TrainingArguments(output_dir=directory, use_cpu=True,
                                     report_to=[], remove_unused_columns=False)
            trainer = MLMLossTrainer(model=model, args=args)
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            self.assertEqual(trainer.loss_config, LossConfig())
            inputs = dict(input_ids=torch.zeros_like(self.labels), labels=self.labels,
                          target_offsets=self.offsets, target_lengths=self.lengths)
            loss, predictions, labels = trainer.prediction_step(
                model, inputs, prediction_loss_only=False)
            expected = calculate_mlm_loss(self.logits, self.labels, config=LossConfig())
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(predictions, self.logits)
            torch.testing.assert_close(labels, self.labels)


class LossRegistrationTests(unittest.TestCase):
    def setUp(self):
        # Each test can extend the registry without leaking a new loss into other tests.
        self.enterContext(patch.dict(losses.LOSS_REGISTRY))
        self.enterContext(patch.dict(losses._LOSS_ALIASES))
        self.enterContext(patch.object(
            losses, "_LOSSES_REQUIRING_METADATA", set(losses._LOSSES_REQUIRING_METADATA)))
        self.enterContext(patch.object(losses, "LOSS_TYPES", losses.LOSS_TYPES))

    def test_new_loss_registration_supplies_names_metadata_and_dispatch(self):
        result = torch.tensor(2., requires_grad=True)
        calculation = Mock(return_value=result)
        losses.register_loss("custom_loss", calculation, aliases=("custom",),
                             requires_target_metadata=True)
        config = LossConfig(" CUSTOM ")
        self.assertEqual(config.loss_type, "custom_loss")
        self.assertTrue(config.requires_target_metadata)
        self.assertIn("custom_loss", losses.LOSS_TYPES)
        self.assertNotIn("custom", losses.LOSS_TYPES)
        self.assertIs(losses.LOSS_REGISTRY["custom_loss"], calculation)

        logits = torch.zeros(1, 2, 3)
        labels = torch.tensor([[0, 1]])
        with self.assertRaisesRegex(ValueError, "custom_loss requires"):
            calculate_mlm_loss(logits, labels, config=config)
        calculation.assert_not_called()

        offsets, lengths = torch.tensor([[0, 1]]), torch.tensor([2])
        actual = calculate_mlm_loss(logits, labels, config=config,
                                    target_offsets=offsets, target_lengths=lengths)
        self.assertIs(actual, result)
        calculation.assert_called_once_with(logits, labels, offsets, lengths, config)

    def test_duplicate_names_and_alias_collisions_do_not_modify_registry(self):
        original = dict(losses.LOSS_REGISTRY)
        for name, aliases in (("cross_entropy", ()), ("ce", ()),
                              ("custom", ("weighted",)), ("custom", ("same", "same"))):
            with self.subTest(name=name, aliases=aliases):
                with self.assertRaisesRegex(ValueError, "already registered"):
                    losses.register_loss(name, Mock(), aliases=aliases)
                self.assertEqual(losses.LOSS_REGISTRY, original)
                self.assertEqual(losses.LOSS_TYPES, tuple(original))


if __name__ == "__main__":
    unittest.main()
