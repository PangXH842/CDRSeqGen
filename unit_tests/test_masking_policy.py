import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cdrseqgen.datasets.dataset_mlm import (
    MaskedInfillingDataset,
    AntibodyTokenizer,
    TokenizerAdapter,
    mlm_collate_fn,
    normalize_masking_policy,
)
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


def _records():
    return [
        {"heavy": _chain(), "light": _chain("QRST"), "split": split}
        for split in ("train", "validation")
    ]


class _Backbone:
    def build_tokenizer(self, _model_type, checkpoint_path=None):
        return TokenizerAdapter.from_custom(AntibodyTokenizer())


class MaskingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "processed.json"
        self.path.write_text(json.dumps(_records()))
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    def tearDown(self):
        self.temp.cleanup()

    def _dataset(self, **kwargs):
        return MaskedInfillingDataset(
            str(self.path), self.tokenizer, target_region="h1", **kwargs
        )

    def test_full_span_remains_the_default(self):
        item = self._dataset()[0]
        masked = item["labels"] != -100
        self.assertEqual(item["masked_count"], item["target_len"])
        self.assertTrue(torch.all(item["input_ids"][masked] == self.tokenizer.mask_id))

    def test_mixed_policy_reveals_and_unsupervises_native_residues(self):
        torch.manual_seed(7)
        item = self._dataset(
            masking_policy="mixed", mixed_full_span_probability=0.0
        )[0]
        supervised = item["labels"] != -100
        span_start = len(self.tokenizer.encode("AA"))
        span = item["input_ids"][span_start:span_start + item["target_len"]]
        target = torch.tensor(self.tokenizer.encode("CDEF"))
        span_labels = item["labels"][span_start:span_start + item["target_len"]]
        visible = span != self.tokenizer.mask_id

        self.assertGreater(item["masked_count"], 0)
        self.assertLess(item["masked_count"], item["target_len"])
        self.assertEqual(int(supervised.sum()), item["masked_count"])
        self.assertEqual(int((span == self.tokenizer.mask_id).sum()), item["masked_count"])
        self.assertTrue(torch.equal(span[visible], target[visible]))
        self.assertTrue(torch.all(span_labels[visible] == -100))
        offsets = item["target_offsets"][span_start:span_start + item["target_len"]]
        self.assertTrue(torch.equal(offsets, torch.arange(item["target_len"])))

    def test_collate_preserves_padded_loss_metadata(self):
        first = self._dataset()[0]
        second = self._dataset()[1]
        batch = mlm_collate_fn([first, second], pad_token_id=self.tokenizer.pad_id)
        self.assertEqual(batch["target_offsets"].shape, batch["labels"].shape)
        self.assertEqual(batch["target_lengths"].tolist(), [4, 4])

    def test_pipeline_uses_distinct_shared_data_views_and_full_span_eval(self):
        data_args = SimpleNamespace(
            data_path=str(self.path),
            max_len=64,
            test_size=0.5,
            eval_split="validation",
            target_region="h1",
            include_partner_chain=False,
            include_antigen=False,
            context_dropout=0.0,
            masking_policy="mixed",
            mixed_full_span_probability=0.0,
        )
        model_args = SimpleNamespace(model_type="scratch")
        training_args = SimpleNamespace(seed=42)

        train, evaluation, _ = get_datasets(
            model_args, data_args, training_args, _Backbone()
        )

        self.assertIsNot(train.dataset, evaluation.dataset)
        self.assertIs(train.dataset.data, evaluation.dataset.data)
        self.assertEqual(train.dataset.masking_policy, "mixed")
        self.assertEqual(evaluation.dataset.masking_policy, "full_span")
        self.assertEqual(evaluation[0]["masked_count"], evaluation[0]["target_len"])

    def test_policy_alias_and_invalid_probability_are_validated(self):
        self.assertEqual(normalize_masking_policy("fullspan"), "full_span")
        with self.assertRaisesRegex(ValueError, "masking_policy"):
            normalize_masking_policy("single_token")
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self._dataset(masking_policy="mixed", mixed_full_span_probability=1.1)


if __name__ == "__main__":
    unittest.main()
