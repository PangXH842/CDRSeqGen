import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.methods.mlm.training import get_datasets


def _chain(cdr1="CDEF"):
    regions = {
        "fwr1": "AA",
        "cdr1": cdr1,
        "fwr2": "GG",
        "cdr2": "HI",
        "fwr3": "KK",
        "cdr3": "LMN",
        "fwr4": "PP",
    }
    regions["full_aa"] = "".join(regions.values())
    return regions


class _Backbone:
    def build_tokenizer(self, _model_type, checkpoint_path=None):
        return TokenizerAdapter.from_custom(AntibodyTokenizer())


class EvalSplitSelectionTests(unittest.TestCase):
    """The eval split drives early stopping and load_best_model_at_end, so leaving it implicit
    silently selects checkpoints on whatever non-train partition the dataset happens to carry."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _data_args(self, splits, eval_split):
        path = self.dir / f"{'_'.join(sorted(set(splits)))}.json"
        path.write_text(json.dumps([
            {"heavy": _chain(), "light": _chain("QRST"), "split": split} for split in splits
        ]))
        return SimpleNamespace(
            data_path=str(path),
            max_len=64,
            test_size=0.5,
            eval_split=eval_split,
            target_region="h1",
            include_partner_chain=False,
            include_antigen=False,
            context_dropout=0.0,
            masking_policy="full_span",
            mixed_full_span_probability=0.5,
        )

    def _call(self, splits, eval_split):
        return get_datasets(
            SimpleNamespace(model_type="scratch"),
            self._data_args(splits, eval_split),
            SimpleNamespace(seed=42),
            _Backbone(),
        )

    def test_unset_eval_split_is_refused_when_a_test_partition_exists(self):
        with self.assertRaisesRegex(ValueError, "eval_split is unset"):
            self._call(["train", "train", "test"], "")

    def test_unset_eval_split_is_refused_on_a_three_way_split(self):
        with self.assertRaisesRegex(ValueError, "eval_split is unset"):
            self._call(["train", "validation", "test"], "")

    def test_validation_is_selected_without_touching_test(self):
        _, evaluation, _ = self._call(["train", "validation", "test"], "validation")
        self.assertEqual(len(evaluation), 1)
        self.assertEqual(evaluation.dataset.data[evaluation.indices[0]]["split"], "validation")

    def test_test_split_remains_available_once_declared(self):
        _, evaluation, _ = self._call(["train", "validation", "test"], "test")
        self.assertEqual(evaluation.dataset.data[evaluation.indices[0]]["split"], "test")

    def test_fallback_survives_for_datasets_with_no_test_partition(self):
        train, evaluation, _ = self._call(["train", "train", "validation"], "")
        self.assertEqual(len(train), 2)
        self.assertEqual(len(evaluation), 1)


if __name__ == "__main__":
    unittest.main()
