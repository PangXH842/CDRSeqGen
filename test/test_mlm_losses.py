import unittest

import torch
import torch.nn.functional as F

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


if __name__ == "__main__":
    unittest.main()
