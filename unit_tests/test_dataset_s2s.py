"""Encoder-decoder span-infilling data: the two-stream properties an s2s method depends on.

The science-critical ones are leakage and the shift. The encoder must carry no residue and no
length of the loop it is asking for — otherwise "length-free" generation is reading the answer —
and the decoder pair must be offset by exactly one, since a double shift still trains and produces
a checkpoint that looks merely bad rather than broken.
"""
import json
import os
import tempfile
import unittest

import torch

from cdrseqgen.datasets.dataset_common import AntibodyTokenizer
from cdrseqgen.datasets.dataset_ar import CausalTokenizer, build_causal_prompt, causal_adapter
from cdrseqgen.datasets.dataset_s2s import (
    ENCODER_LAYOUT_TOKENS,
    MAX_LOOP_RESIDUES,
    S2S_DECODE_STEPS,
    Seq2SeqInfillingDataset,
    Seq2SeqTokenizer,
    build_decoder_pair,
    build_encoder_input,
    s2s_adapter,
    s2s_collate_fn,
    validate_s2s_decode_steps,
)

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
    def test_the_shared_vocab_is_a_prefix_of_the_seq2seq_one(self):
        shared, s2s = AntibodyTokenizer(), Seq2SeqTokenizer()
        self.assertEqual(s2s.vocab[:len(shared.vocab)], shared.vocab)
        self.assertEqual(len(shared.vocab), 25)
        self.assertEqual((s2s.bos_id, s2s.eos_id), (25, 26))

    def test_the_seq2seq_and_causal_vocabularies_agree_token_for_token(self):
        # dataset_s2s duplicates the <bos>/<eos> extension rather than importing it from the causal
        # module, so the two methods stay independently testable. Drift is the failure that buys,
        # and it would surface as a checkpoint decoding to the wrong residues, so it is asserted.
        self.assertEqual(Seq2SeqTokenizer().vocab, CausalTokenizer().vocab)


class EncoderStreamTests(unittest.TestCase):
    def setUp(self):
        self.path = _data()
        self.tok = Seq2SeqTokenizer()
        self.adapter = s2s_adapter(self.tok)
        self.ds = Seq2SeqInfillingDataset(self.path, self.adapter, target_region="h3")

    def tearDown(self):
        os.unlink(self.path)

    def test_both_flanks_are_present_and_separated_by_one_gap_marker(self):
        encoder_ids = self.ds[0]["input_ids"].tolist()
        self.assertEqual(encoder_ids.count(self.adapter.mask_id), ENCODER_LAYOUT_TOKENS)
        gap = encoder_ids.index(self.adapter.mask_id)
        self.assertEqual(self.tok.decode(encoder_ids[:gap]), "AAAACCDDDDEEFFFF")
        self.assertEqual(self.tok.decode(encoder_ids[gap + 1:]), "HHHH")

    def test_the_encoder_never_carries_the_loop(self):
        # "GGGGG" is the native H3. If it reaches the encoder the task is copying, not design.
        encoder_ids = self.ds[0]["input_ids"].tolist()
        self.assertNotIn("G", self.tok.decode(encoder_ids))

    def test_the_encoder_input_does_not_depend_on_the_loop_length(self):
        # The gap marker is ONE token whether the loop is 2 residues or 30. A marker that scaled
        # with the answer would hand the model the length it is supposed to predict — the §10
        # length-head problem, reintroduced through the back door.
        base = dict(pt1="AAAACCDDDDEEFFFF", pt2="HHHH", partner_chain=None, antigen=None,
                    max_len=128, target="GG", target_length=2)
        short, _ = build_encoder_input(self.adapter, base)
        long, _ = build_encoder_input(self.adapter, dict(base, target="G" * 30, target_length=30))
        self.assertEqual(short, long)

    def test_context_is_capped_against_the_whole_budget(self):
        # The reserve fix, asserted rather than described: at one max_len the causal prompt must
        # hold back LOOP_LENGTH_RESERVE positions for an answer in the same stream, while the
        # encoder spends one token on the gap. So s2s retains strictly more context here.
        framework = dict(pt1="A" * 200, pt2="H" * 40, partner_chain="P" * 80, antigen=None,
                         max_len=128, target="GGGGG", target_length=5)
        encoder_ids, _ = build_encoder_input(self.adapter, framework)
        causal_prompt, _ = build_causal_prompt(causal_adapter(CausalTokenizer()), dict(framework))
        self.assertGreater(len(encoder_ids), len(causal_prompt))
        self.assertLessEqual(len(encoder_ids), framework["max_len"])


class DecoderStreamTests(unittest.TestCase):
    def setUp(self):
        self.path = _data()
        self.tok = Seq2SeqTokenizer()
        self.adapter = s2s_adapter(self.tok)
        self.ds = Seq2SeqInfillingDataset(self.path, self.adapter, target_region="h3")

    def tearDown(self):
        os.unlink(self.path)

    def test_the_pair_is_offset_by_exactly_one(self):
        target = self.adapter.encode("GGGGG")
        decoder_input, labels = build_decoder_pair(self.adapter, target)
        self.assertEqual(len(decoder_input), len(labels))
        self.assertEqual(decoder_input[0], self.adapter.bos_id)
        self.assertEqual(labels[-1], self.adapter.eos_id)
        # position i of the input predicts position i of the labels, which is input position i+1
        self.assertEqual(decoder_input[1:], labels[:-1])

    def test_the_loop_is_supervised_in_full_with_no_ignore_index(self):
        # Unlike the causal layout there is no prompt in this stream, so answer-only supervision is
        # structural. A -100 appearing here would mean framework tokens leaked into the decoder.
        item = self.ds[0]
        self.assertNotIn(-100, item["labels"].tolist())
        self.assertEqual(self.tok.decode(item["labels"].tolist()[:-1]), "GGGGG")
        self.assertEqual(item["target_len"], 5)

    def test_a_loop_beyond_the_decoder_budget_is_rejected_at_construction(self):
        oversized = dict(RECORD)
        oversized["heavy"] = dict(CHAIN, cdr3="G" * (MAX_LOOP_RESIDUES + 1))
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump([oversized], handle)
        handle.close()
        try:
            ds = Seq2SeqInfillingDataset(handle.name, self.adapter, target_region="h3")
            with self.assertRaisesRegex(ValueError, "decoder budget"):
                ds[0]
        finally:
            os.unlink(handle.name)

    def test_the_decode_cap_leaves_a_step_for_the_terminator(self):
        self.assertEqual(S2S_DECODE_STEPS, MAX_LOOP_RESIDUES + 1)
        self.assertEqual(validate_s2s_decode_steps(S2S_DECODE_STEPS), S2S_DECODE_STEPS)
        for bad in (0, -1, S2S_DECODE_STEPS + 1, True, 3.0):
            with self.assertRaises(ValueError):
                validate_s2s_decode_steps(bad)


class CollatorTests(unittest.TestCase):
    def setUp(self):
        self.adapter = s2s_adapter()

    def _item(self, encoder_len, loop):
        target = self.adapter.encode(loop)
        decoder_input, labels = build_decoder_pair(self.adapter, target)
        return {
            "input_ids": torch.arange(1, encoder_len + 1, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_len": len(target),
            "pt1_len": 0,
        }

    def test_the_streams_are_padded_independently(self):
        # A long framework with a short loop must not stretch the decoder to the encoder's length.
        batch = s2s_collate_fn([self._item(30, "GG"), self._item(8, "GGGGGGGG")])
        self.assertEqual(batch["input_ids"].shape[1], 30)
        self.assertEqual(batch["decoder_input_ids"].shape[1], 9)     # 8 residues + <bos>
        self.assertEqual(batch["labels"].shape[1], 9)

    def test_padding_is_masked_on_both_sides_and_ignored_by_the_loss(self):
        batch = s2s_collate_fn([self._item(30, "GG"), self._item(8, "GGGGGGGG")])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1] * 8 + [0] * 22)
        self.assertEqual(batch["decoder_attention_mask"][0].tolist(), [1] * 3 + [0] * 6)
        self.assertEqual(batch["labels"][0].tolist()[3:], [-100] * 6)


if __name__ == "__main__":
    unittest.main()
