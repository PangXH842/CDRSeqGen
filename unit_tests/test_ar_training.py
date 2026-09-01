"""The causal training pipeline: split selection, and the metric's alignment.

The alignment test is the one that matters. `causal_compute_metrics` receives predictions and
labels UNSHIFTED and must redo the next-token shift itself; an off-by-one there reports a different
quantity than the masked side's `compute_metrics` while looking perfectly healthy, which would make
every AR-vs-MLM comparison wrong in a way no other test would catch.
"""
import json
import os
import tempfile
import unittest

import numpy as np

from cdrseqgen.data.dataset_common import select_split_indices
from cdrseqgen.methods.ar.backbones import resolve
from cdrseqgen.methods.ar.training import causal_compute_metrics, get_datasets


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


CHAIN = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
         "fwr3": "FFFF", "cdr3": "GGGGGG", "fwr4": "HHHH"}


def _data(splits):
    records = [{"heavy": dict(CHAIN, full_aa="x"), "light": dict(CHAIN, full_aa="x"), "split": s}
               for s in splits]
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(records, handle)
    handle.close()
    return handle.name


# labels as the model saw them: 3 prompt positions, a 6-residue loop (10..15), then <eos>=26.
LABELS = np.array([[-100, -100, -100, 10, 11, 12, 13, 14, 15, 26]])
PERFECT = np.array([[0, 0, 10, 11, 12, 13, 14, 15, 26, 0]])   # preds[t] == labels[t+1]


class MetricAlignmentTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self):
        m = causal_compute_metrics((PERFECT, LABELS))
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["core_accuracy"], 1.0)
        self.assertEqual(m["anchor_accuracy"], 1.0)

    def test_a_shift_that_is_off_by_one_scores_zero(self):
        # The failure this test exists for: silently measuring position t against residue t+1.
        misaligned = np.array([[0, 10, 11, 12, 13, 14, 15, 26, 0, 0]])
        self.assertEqual(causal_compute_metrics((misaligned, LABELS))["accuracy"], 0.0)

    def test_core_and_anchor_split_at_the_same_boundary_as_the_masked_side(self):
        # Loop length 6, trim 2 -> core is the middle two residues.
        broken = PERFECT.copy()
        broken[0, 4] = 99
        broken[0, 5] = 99
        m = causal_compute_metrics((broken, LABELS))
        self.assertEqual(m["core_accuracy"], 0.0)
        self.assertEqual(m["anchor_accuracy"], 1.0)

    def test_eos_is_excluded_from_recovery(self):
        # Recovery must measure residue prediction, comparable with a masked model that has no eos.
        eos_wrong = PERFECT.copy()
        eos_wrong[0, 8] = 99
        self.assertEqual(causal_compute_metrics((eos_wrong, LABELS))["accuracy"], 1.0)

    def test_logits_are_accepted_as_well_as_argmaxed_ids(self):
        logits = np.zeros((1, 10, 30), dtype=np.float32)
        for t, token in enumerate(PERFECT[0]):
            logits[0, t, token] = 1.0
        self.assertEqual(causal_compute_metrics((logits, LABELS))["accuracy"], 1.0)

    def test_a_row_with_no_supervision_is_skipped_not_counted_as_zero(self):
        labels = np.array([[-100] * 10])
        self.assertEqual(causal_compute_metrics((PERFECT, labels))["accuracy"], 0.0)


class SplitTests(unittest.TestCase):
    """The causal pipeline must inherit the guard, not reimplement it."""

    def _datasets(self, splits, eval_split):
        path = _data(splits)
        backbone = resolve("scratch")
        model_args = Args(model_type="scratch")
        data_args = Args(data_path=path, max_len=128, target_region="h3",
                         include_partner_chain=False, include_antigen=False,
                         eval_split=eval_split, context_dropout=0.0, test_size=0.1)
        try:
            return get_datasets(model_args, data_args, Args(seed=42), backbone)
        finally:
            os.unlink(path)

    def test_an_unset_eval_split_is_refused_when_a_test_partition_exists(self):
        with self.assertRaises(ValueError) as ctx:
            self._datasets(["train", "train", "validation", "test"], "")
        self.assertIn("test", str(ctx.exception))

    def test_validation_is_selected_without_touching_test(self):
        train, evaluation, _ = self._datasets(
            ["train", "train", "validation", "validation", "test"], "validation")
        self.assertEqual((len(train), len(evaluation)), (2, 2))

    def test_the_guard_is_the_shared_one(self):
        # If this ever diverges from the masked side, one method's results can be contaminated
        # without the other's tests noticing.
        with self.assertRaises(ValueError):
            select_split_indices([{"split": "train"}, {"split": "test"}], "",
                                 data_path="x.json", seed=0)


class NoMaskingViewTests(unittest.TestCase):
    def test_train_and_eval_share_one_dataset(self):
        # A causal example never reveals the answer, so there is no full_span view to build.
        path = _data(["train", "train", "validation"])
        backbone = resolve("scratch")
        train, evaluation, _ = get_datasets(
            Args(model_type="scratch"),
            Args(data_path=path, max_len=128, target_region="h3", include_partner_chain=False,
                 include_antigen=False, eval_split="validation", context_dropout=0.0, test_size=0.1),
            Args(seed=42), backbone)
        self.assertIs(train.dataset, evaluation.dataset)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
