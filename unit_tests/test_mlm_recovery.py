"""Trainer recovery uses shared region boundaries and retains token pooling."""
import unittest
from unittest.mock import call, patch

import numpy as np

from cdrseqgen.eval.metrics import core_anchor_indices
from cdrseqgen.methods.mlm import recovery


class MLMRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(recovery, "_warned_low_core_coverage", False))

    def test_shared_loop_offsets_map_to_input_columns_and_preserve_short_rows(self):
        selected = np.zeros((3, 10), dtype=bool)
        selected[0, 2:8] = True   # loop length 6, after two context positions
        selected[1, 1:5] = True   # loop length 4: all anchor at trim 2
        with patch.object(recovery, "core_anchor_indices", wraps=core_anchor_indices) as split:
            core, anchor = recovery._core_anchor_masks(selected, trim=2)
        self.assertEqual(split.call_args_list, [call(6, 2), call(4, 2), call(0, 2)])
        self.assertEqual(np.flatnonzero(core[0]).tolist(), [4, 5])
        self.assertEqual(np.flatnonzero(anchor[0]).tolist(), [2, 3, 6, 7])
        self.assertFalse(core[1:].any())
        self.assertEqual(np.flatnonzero(anchor[1]).tolist(), [1, 2, 3, 4])
        self.assertFalse(anchor[2].any())
        np.testing.assert_array_equal(core | anchor, selected)
        self.assertFalse((core & anchor).any())

    def test_zero_trim_assigns_every_supervised_position_to_core(self):
        selected = np.array([[False, True, True, True, False]])
        core, anchor = recovery._core_anchor_masks(selected, trim=0)
        np.testing.assert_array_equal(core, selected)
        self.assertFalse(anchor.any())

    def test_accuracy_pools_tokens_while_coverage_counts_rows(self):
        labels = np.full((2, 10), -100)
        labels[0, 2:8] = [1, 2, 3, 4, 5, 6]
        labels[1, 1:4] = [1, 2, 3]
        predicted = np.full((2, 10), 7)
        predicted[0, 2:8] = labels[0, 2:8]
        logits = np.eye(8)[predicted]
        # Row 0 is entirely correct; row 1 is entirely wrong and has no core.
        # Token pooling gives 6/9 rather than the per-row mean of 1/2.
        scores = recovery.compute_metrics(((logits, np.zeros((2, 1))), labels), trim=2)
        self.assertEqual(scores, {
            "accuracy": 6/9, "core_accuracy": 1.,
            "anchor_accuracy": 4/7, "core_coverage": 1/2,
        })

    def test_no_core_is_distinguished_from_no_correct_tokens_by_coverage(self):
        labels = np.array([[0, 1, 2, 3]])
        logits = np.eye(4)[labels]
        with patch.object(recovery.logger, "warning") as warning:
            scores = recovery.compute_metrics((logits, labels), trim=2)
        self.assertEqual(scores, {
            "accuracy": 1., "core_accuracy": 0.,
            "anchor_accuracy": 1., "core_coverage": 0.,
        })
        warning.assert_called_once()

    def test_entirely_unsupervised_input_keeps_existing_return_keys(self):
        labels = np.full((1, 4), -100)
        scores = recovery.compute_metrics((np.zeros((1, 4, 5)), labels), trim=2)
        self.assertEqual(scores, dict(accuracy=0., core_accuracy=0., anchor_accuracy=0.))

    def test_low_coverage_warns_once_and_describes_population_without_dismissing_score(self):
        with patch.object(recovery.logger, "warning") as warning:
            recovery._warn_if_low_core_coverage(0.5, 2, 4, trim=2)
            warning.assert_not_called()
            recovery._warn_if_low_core_coverage(0.25, 1, 4, trim=2)
            recovery._warn_if_low_core_coverage(0.25, 1, 4, trim=2)
        warning.assert_called_once()
        message = warning.call_args.args[0]
        self.assertIn("1/4", message)
        self.assertIn("25.0%", message)
        self.assertIn("non-empty core", message)
        self.assertIn("whole-span accuracy", message)
        self.assertNotIn("noise", message)


if __name__ == "__main__":
    unittest.main()
