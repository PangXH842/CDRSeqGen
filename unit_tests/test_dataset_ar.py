"""Causal span-infilling data: the layout properties an AR method depends on.

The science-critical one is the reordering. A causal model reads left to right, so writing the
sequence naturally — pt1, loop, pt2 — leaves the right flank unread at the moment the loop is
generated. FR4 carries the J-gene motif that terminates CDR-H3, and Findings §11 measured the cost
of losing it: 2% of loops ended in that motif when IgLM was forced past its stopping point against
20% when it chose freely. So `pt2 must appear in the prompt` is not a style preference, and it gets
a test.
"""
import json
import os
import tempfile
import unittest

import torch

from cdrseqgen.data.dataset_common import AntibodyFrameworkDataset, AntibodyTokenizer
from cdrseqgen.data.dataset_ar import (CausalInfillingDataset, CausalTokenizer, build_causal_prompt,
                                   causal_adapter, causal_collate_fn)

CHAIN = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
         "fwr3": "FFFF", "cdr3": "GGGGG", "fwr4": "HHHH"}
RECORD = {"heavy": dict(CHAIN, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
          "light": dict(CHAIN, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
          "antigen": {"full_aa": "WWWW"}, "split": "train"}


def _data(n=4):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump([RECORD] * n, handle)
    handle.close()
    return handle.name


class VocabTests(unittest.TestCase):
    def test_the_shared_vocab_is_a_prefix_of_the_causal_one(self):
        # Renumbering would silently mismatch every existing 25-row MLM embedding table.
        shared, causal = AntibodyTokenizer(), CausalTokenizer()
        self.assertEqual(causal.vocab[:len(shared.vocab)], shared.vocab)
        self.assertEqual(len(shared.vocab), 25)
        self.assertEqual((causal.bos_id, causal.eos_id), (25, 26))

    def test_the_adapter_exposes_an_end_token(self):
        # A masked LM has no EOS, which is why §10 needed a length classifier at all.
        self.assertIsNotNone(causal_adapter().eos_id)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.path = _data()
        self.tok = CausalTokenizer()
        self.ds = CausalInfillingDataset(self.path, causal_adapter(self.tok), target_region="h3")

    def tearDown(self):
        os.unlink(self.path)

    def test_the_right_flank_is_inside_the_prompt(self):
        item = self.ds[0]
        ids, prompt_len = item["input_ids"].tolist(), item["prompt_len"]
        adapter = self.ds.tokenizer
        between = ids[ids.index(adapter.mask_id) + 1: prompt_len - 1]
        self.assertEqual(self.tok.decode(between), "HHHH")     # pt2, before generation starts

    def test_the_answer_is_the_native_loop_then_eos(self):
        item = self.ds[0]
        ids, prompt_len = item["input_ids"].tolist(), item["prompt_len"]
        self.assertEqual(self.tok.decode(ids[prompt_len:-1]), "GGGGG")
        self.assertEqual(ids[-1], self.tok.eos_id)

    def test_the_prompt_is_never_supervised(self):
        # Scoring given framework residues would inflate any token-level number.
        item = self.ds[0]
        labels = item["labels"].tolist()
        self.assertEqual(set(labels[:item["prompt_len"]]), {-100})

    def test_eos_is_supervised_so_length_is_learned(self):
        # This is what replaces the gap-sentinel length head of Findings §10.
        self.assertEqual(self.ds[0]["labels"].tolist()[-1], self.tok.eos_id)

    def test_supervised_positions_are_exactly_the_loop_plus_eos(self):
        item = self.ds[0]
        supervised = (item["labels"] != -100).sum().item()
        self.assertEqual(supervised, item["target_len"] + 1)


class CapTests(unittest.TestCase):
    def setUp(self):
        self.path = _data()
        self.adapter = causal_adapter()

    def tearDown(self):
        os.unlink(self.path)

    def test_capping_truncates_context_not_the_answer(self):
        # max_len must clear LOOP_LENGTH_RESERVE + the three layout tokens; below that the reserve
        # itself cannot fit and the example is refused rather than silently truncated.
        for max_len in (128, 64, 44):
            item = CausalInfillingDataset(self.path, self.adapter, target_region="h3",
                                          max_len=max_len, include_partner_chain=True)[0]
            self.assertLessEqual(len(item["input_ids"]), max_len)
            self.assertEqual(item["target_len"], 5)            # loop survives intact

    def test_a_max_len_too_small_for_the_reserve_is_refused(self):
        # Honouring it would mean cutting the training target; fail where the cause is named.
        ds = CausalInfillingDataset(self.path, self.adapter, target_region="h3", max_len=7)
        with self.assertRaises(ValueError) as ctx:
            ds[0]
        self.assertIn("max_len", str(ctx.exception))

    def test_the_generation_prompt_is_byte_identical_to_the_training_one(self):
        """The bug this fix exists for.

        The reserve used to be the native loop length at training and `max_new_tokens` at
        generation, so `inject_context` trimmed the flanks differently in the two paths. With
        max_len=128 that produced an 87-token prompt at sampling against 110 at training: the model
        met a framework 23 tokens shorter than anything it had trained on, and stopped emitting
        <eos> entirely (P(eos) ~ 1e-4 free-running against 0.91 teacher-forced). The prompt must be
        a function of (framework, max_len) alone, because generation cannot know the length it is
        trying to predict.
        """
        for max_len in (64, 128, 256):
            ds = CausalInfillingDataset(self.path, self.adapter, target_region="h3",
                                        max_len=max_len, include_partner_chain=True)
            item = ds[0]
            trained_prompt = item["input_ids"].tolist()[:item["prompt_len"]]
            generated_prompt, _ = build_causal_prompt(self.adapter, ds.framework_parts(0))
            self.assertEqual(trained_prompt, generated_prompt, f"diverged at max_len={max_len}")

    def test_a_loop_longer_than_the_reserve_is_refused(self):
        # Otherwise the assembled example could overflow max_len without anyone noticing.
        from cdrseqgen.data.dataset_ar import LOOP_LENGTH_RESERVE
        framework = {"pt1": "AAAA", "pt2": "HHHH", "target": "G" * (LOOP_LENGTH_RESERVE + 1),
                     "target_length": LOOP_LENGTH_RESERVE + 1, "partner_chain": None,
                     "antigen": None, "max_len": 256}
        with self.assertRaises(ValueError):
            build_causal_prompt(self.adapter, framework)

    def test_generation_can_reserve_a_length_it_does_not_know(self):
        # At sampling time the loop length is exactly what is unknown, so reserve is explicit.
        ds = AntibodyFrameworkDataset(self.path, self.adapter, target_region="h3", max_len=64)
        # reserve=50 leaves a budget of 64-53=11 for 20 residues of flank, so context must give way.
        short, _ = build_causal_prompt(self.adapter, ds.framework_parts(0), reserve=5)
        long, _ = build_causal_prompt(self.adapter, ds.framework_parts(0), reserve=50)
        self.assertGreater(len(short), len(long))              # a longer reserve caps context harder


class SharedWithTheMaskedMethodTests(unittest.TestCase):
    def test_the_causal_dataset_reuses_the_neutral_base(self):
        self.assertTrue(issubclass(CausalInfillingDataset, AntibodyFrameworkDataset))

    def test_context_dropout_is_inherited_not_reimplemented(self):
        path = _data(20)
        kw = dict(target_region="h3", include_partner_chain=True, context_dropout=0.5,
                  context_dropout_seed=5)
        causal = CausalInfillingDataset(path, causal_adapter(), **kw)
        neutral = AntibodyFrameworkDataset(path, causal_adapter(), **kw)
        kept_c = [causal.framework_parts(i)["partner_chain"] is not None for i in range(20)]
        kept_n = [neutral.framework_parts(i)["partner_chain"] is not None for i in range(20)]
        self.assertEqual(kept_c, kept_n)
        self.assertIn(False, kept_c)
        os.unlink(path)


class CollateTests(unittest.TestCase):
    def test_padding_is_masked_and_labels_stay_ignored(self):
        path = _data()
        adapter = causal_adapter()
        ds = CausalInfillingDataset(path, adapter, target_region="h3")
        items = [ds[0], ds[1]]
        items[1] = dict(items[1])
        items[1]["input_ids"] = items[1]["input_ids"][:-3]      # force a ragged batch
        items[1]["labels"] = items[1]["labels"][:-3]
        batch = causal_collate_fn(items, pad_token_id=adapter.pad_id)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertEqual(batch["attention_mask"][1].sum().item(), len(items[1]["input_ids"]))
        self.assertTrue(torch.all(batch["labels"][1, len(items[1]["labels"]):] == -100))
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
