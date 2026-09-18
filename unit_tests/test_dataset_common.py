"""The method-neutral data seam: what a non-masked method can build on.

`dataset_common.py` was split out of `dataset_mlm.py` so an autoregressive or encoder-decoder method can
reuse framework/target splitting, context injection and length capping without inheriting masking.
These tests pin the two properties that make that seam real:

  * the neutral layer never requires a mask token — that requirement was the concrete coupling, and
    a causal tokenizer has no mask token to give it;
  * the masked-LM view is a *subclass*, so a fix to framework splitting or context dropout reaches
    every method rather than only the one that happened to be edited.
"""
import json
import os
import tempfile
import unittest

import torch

from cdrseqgen.datasets.dataset_mlm import MaskedInfillingDataset
from cdrseqgen.datasets.dataset_common import (AntibodyFrameworkDataset, AntibodyTokenizer,
                                       TokenizerAdapter, framework_for_dataset, inject_context,
                                            select_split_indices)

CHAIN = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
         "fwr3": "FFFF", "cdr3": "GGG", "fwr4": "HHHH"}
RECORD = {"heavy": dict(CHAIN, full_aa="AAAACCDDDDEEFFFFGGGHHHH"),
          "light": dict(CHAIN, full_aa="AAAACCDDDDEEFFFFGGGHHHH"),
          "antigen": {"full_aa": "WWWW"}, "split": "train"}


def _dataset_file(n=4):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump([RECORD] * n, handle)
    handle.close()
    return handle.name


def _causal_adapter():
    """An adapter with NO mask token — what an autoregressive method would supply."""
    base = AntibodyTokenizer()
    return TokenizerAdapter(
        encode=base.encode, decode=base.decode, vocab_size=len(base.vocab), pad_id=base.pad_id,
        cls_id=None, end_id=None, chain_sep_id=base.tok_to_id["<sep>"],
        special_ids={base.tok_to_id[t] for t in ("<pad>", "<unk>", "<cls>", "<sep>")},
        uses_special_tokens=False, position_type="absolute")


class NeutralLayerTests(unittest.TestCase):
    def test_the_adapter_does_not_require_a_mask_token(self):
        self.assertIsNone(_causal_adapter().mask_id)

    def test_context_injection_works_without_a_mask_token(self):
        t = _causal_adapter()
        pt1, pt2 = inject_context(t, t.encode("AAAA"), t.encode("HHHH"), 3,
                                  partner_seq="KKKK", max_len=64)
        self.assertIn(t.chain_sep_id, pt1)       # partner chain prepended
        self.assertEqual(pt2, t.encode("HHHH"))

    def test_framework_parts_needs_no_masking_configuration(self):
        path = _dataset_file()
        ds = AntibodyFrameworkDataset(path, _causal_adapter(), target_region="h3")
        parts = ds.framework_parts(0)
        self.assertEqual(parts["target"], "GGG")
        self.assertEqual(parts["pt1"], "AAAACCDDDDEEFFFF")
        self.assertEqual(parts["pt2"], "HHHH")
        self.assertEqual(parts["target_length"], 3)
        os.unlink(path)

    def test_a_causal_subclass_can_lay_out_its_own_inputs(self):
        class Causal(AntibodyFrameworkDataset):
            def __getitem__(self, idx):
                f = self.framework_parts(idx)
                t = self.tokenizer
                pt1, pt2 = inject_context(t, t.encode(f["pt1"]), t.encode(f["pt2"]),
                                          f["target_length"], max_len=self.max_len)
                return torch.tensor(pt1 + t.encode(f["target"]) + pt2)
        path = _dataset_file()
        ds = Causal(path, _causal_adapter(), target_region="h3")
        self.assertEqual(len(ds), 4)
        self.assertEqual(len(ds[0]), 23)          # the whole chain, nothing masked out
        os.unlink(path)


class SharedWithTheMaskedViewTests(unittest.TestCase):
    def test_the_masked_dataset_is_a_subclass_of_the_neutral_one(self):
        # Not a copy: a fix to splitting or context dropout must reach every method.
        self.assertTrue(issubclass(MaskedInfillingDataset, AntibodyFrameworkDataset))

    def test_both_views_resolve_identical_framework_parts(self):
        path = _dataset_file()
        kw = dict(target_region="h3", include_partner_chain=True, context_dropout=0.5,
                  context_dropout_seed=3)
        neutral = AntibodyFrameworkDataset(path, _causal_adapter(), **kw)
        masked = MaskedInfillingDataset(path, TokenizerAdapter.from_custom(AntibodyTokenizer()), **kw)
        for i in range(4):
            self.assertEqual(neutral.framework_parts(i), masked.framework_parts(i))
        os.unlink(path)

    def test_framework_for_dataset_unwraps_a_subset(self):
        path = _dataset_file()
        ds = AntibodyFrameworkDataset(path, _causal_adapter(), target_region="h3")
        subset = torch.utils.data.Subset(ds, [2, 3])
        self.assertEqual(framework_for_dataset(subset, 0), ds.framework_parts(2))
        os.unlink(path)

    def test_context_dropout_is_deterministic_across_methods(self):
        path = _dataset_file(20)
        kw = dict(target_region="h3", include_partner_chain=True, context_dropout=0.5,
                  context_dropout_seed=11)
        a = AntibodyFrameworkDataset(path, _causal_adapter(), **kw)
        b = MaskedInfillingDataset(path, TokenizerAdapter.from_custom(AntibodyTokenizer()), **kw)
        kept_a = [a.framework_parts(i)["partner_chain"] is not None for i in range(20)]
        kept_b = [b.framework_parts(i)["partner_chain"] is not None for i in range(20)]
        self.assertEqual(kept_a, kept_b)
        self.assertIn(False, kept_a)              # dropout actually dropped something
        os.unlink(path)

class HFAdapterTests(unittest.TestCase):
    def test_runtime_added_tokens_count_toward_vocab_and_tokenizer_is_saveable(self):
        class FakeTokenizer:
            vocab_size = 2
            pad_token_id = 0
            mask_token_id = 3
            cls_token_id = None
            bos_token_id = 1
            eos_token_id = 2
            sep_token_id = None
            all_special_ids = [0, 1, 2, 3]

            def __len__(self):
                return 4

            def encode(self, seq, add_special_tokens=False):
                return [1] * len(seq)

            def decode(self, ids):
                return "A" * len(ids)

            def save_pretrained(self, path):
                return path

        tokenizer = FakeTokenizer()
        adapter = TokenizerAdapter.from_hf(
            tokenizer, position_type="absolute", uses_special_tokens=True)

        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.vocab_size, 4)
        self.assertIs(adapter.backend_tokenizer, tokenizer)
        self.assertTrue(callable(adapter.save_pretrained))



class DatasetContractTests(unittest.TestCase):
    def select(self, records, eval_split=""):
        return select_split_indices(records, eval_split, data_path="probe.json", seed=42)

    def test_missing_and_null_splits_use_the_same_seeded_random_partition(self):
        missing = [{} for _ in range(20)]
        expected = self.select(missing)
        for records in ([{"split": None} for _ in missing],
                        [{"split": None} if i % 2 else {} for i in range(20)]):
            self.assertEqual(self.select(records), expected)
            with self.assertRaisesRegex(ValueError, "named split"):
                self.select(records, "validation")
        train, evaluation, label = expected
        self.assertEqual((len(train), len(evaluation)), (18, 2))
        self.assertFalse(set(train) & set(evaluation))
        self.assertEqual(label, "random split")

    def test_partially_labelled_corpora_never_repartition_the_holdout(self):
        for unlabelled in ({}, {"split": None}):
            for selected in ("", "test"):
                with self.subTest(unlabelled=unlabelled, selected=selected):
                    with self.assertRaisesRegex(ValueError, "partially labelled"):
                        self.select([{"split": "test"}] + [unlabelled] * 9, selected)

    def test_named_splits_keep_membership_and_explicit_test_selection(self):
        records = [{"split": name} for name in ("train", "test", "validation", "train")]
        self.assertEqual(self.select(records, "validation"), ([0, 3], [2], "validation"))
        self.assertEqual(self.select(records, "test"), ([0, 3], [1], "test"))
        with self.assertRaisesRegex(ValueError, "eval_split is unset"):
            self.select(records)

    def test_invalid_split_types_are_rejected_before_sorting(self):
        for label in ("", " ", 0, False, [], {}):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "Invalid split label"):
                self.select([{"split": "train"}, {"split": label}])

    def test_all_supported_targets_and_legacy_aliases_select_the_right_chain_and_loop(self):
        path = _dataset_file()
        self.addCleanup(os.unlink, path)
        for chain, prefixes in (("heavy", ("h", "cdr")), ("light", ("l", "lcdr"))):
            for number in (1, 2, 3):
                for prefix in prefixes:
                    name = f"{prefix}{number}"
                    with self.subTest(target=name):
                        self.assertEqual(AntibodyFrameworkDataset._resolve_target(name.upper()),
                                         (chain, f"cdr{number}"))
                        ds = AntibodyFrameworkDataset(path, _causal_adapter(), target_region=name)
                        self.assertEqual(ds.framework_parts(0)["target"], CHAIN[f"cdr{number}"])

    def test_invalid_targets_fail_at_construction_in_all_methods(self):
        from cdrseqgen.datasets.dataset_ar import CausalInfillingDataset
        from cdrseqgen.datasets.dataset_s2s import Seq2SeqInfillingDataset
        for dataset_class in (AntibodyFrameworkDataset, MaskedInfillingDataset,
                              CausalInfillingDataset, Seq2SeqInfillingDataset):
            for target in ("h4", "typo3", "", None, 3):
                with self.subTest(method=dataset_class.__name__, target=target):
                    with self.assertRaisesRegex(ValueError, "Unsupported target_region"):
                        dataset_class("does-not-exist.json", _causal_adapter(), target_region=target)
        with self.assertRaisesRegex(ValueError, "Unsupported CDR region"):
            AntibodyFrameworkDataset._split_regions(CHAIN, "cdr4")

    def test_mlm_length_budget_includes_wrappers_and_preserves_the_full_target(self):
        path = _dataset_file()
        self.addCleanup(os.unlink, path)
        for wrapped in (False, True):
            tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
            tokenizer.uses_special_tokens = wrapped
            tokenizer.cls_id, tokenizer.end_id = 2, 3
            minimum = len(CHAIN["cdr3"]) + (2 if wrapped else 0)
            for cap in (minimum - 1, minimum, minimum + 2):
                with self.subTest(wrapped=wrapped, max_len=cap):
                    ds = MaskedInfillingDataset(path, tokenizer, max_len=cap,
                                                include_partner_chain=True, include_antigen=True)
                    if cap < minimum:
                        with self.assertRaisesRegex(ValueError, "cannot hold.*target"):
                            ds[0]
                    else:
                        item = ds[0]
                        self.assertLessEqual(len(item["input_ids"]), cap)
                        labelled = item["labels"][item["labels"] != -100].tolist()
                        self.assertEqual(labelled, tokenizer.encode(CHAIN["cdr3"]))
                        if wrapped:
                            self.assertEqual(item["input_ids"][[0, -1]].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
