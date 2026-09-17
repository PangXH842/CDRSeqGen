"""Seq2seq decoding: the sampler contract, and its agreement with training.

The failure that matters most is silent: a sampler that reads a different encoder input than
training built produces a model meeting frameworks it never saw, and nothing crashes — that is the
2026-09-02 prompt-divergence bug, which cost a batch and was invisible until <eos> stopped arriving.
"""
import unittest
from types import SimpleNamespace

import torch

from cdrseqgen.datasets.dataset_s2s import (S2S_DECODE_STEPS, Seq2SeqInfillingDataset,
                                        build_encoder_input, s2s_adapter)
from cdrseqgen.methods.s2s.sampling import sample_loops

FRAMEWORK = dict(pt1="ACDEFGHIK", pt2="LMNPQRST", partner_chain=None, antigen=None,
                 max_len=64, target="WYVWY", target_length=5)


class StubModel:
    """A model with scripted next-token behaviour, so decode control flow is testable exactly."""

    def __init__(self, script, vocab_size=27, hidden=4):
        self.script = list(script)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.calls = 0
        self.encode_calls = 0

    def eval(self):
        return self

    def encode(self, input_ids, attention_mask=None):
        self.encode_calls += 1
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], self.hidden)

    def __call__(self, *, decoder_input_ids, encoder_hidden_states=None, attention_mask=None,
                 **_kw):
        step = decoder_input_ids.shape[1] - 1
        rows = decoder_input_ids.shape[0]
        logits = torch.full((rows, decoder_input_ids.shape[1], self.vocab_size), -20.0)
        token = self.script[min(step, len(self.script) - 1)]
        logits[:, -1, token] = 20.0
        self.calls += 1
        return {"logits": logits, "loss": None}


class DecodeControlFlowTests(unittest.TestCase):
    def setUp(self):
        self.tok = s2s_adapter()
        self.encoder_ids, _ = build_encoder_input(self.tok, dict(FRAMEWORK))

    def test_decoding_stops_at_eos_and_the_terminator_is_not_a_residue(self):
        residues = self.tok.encode("WYV")
        model = StubModel(residues + [self.tok.eos_id])
        (draw,) = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                               temperature=0)
        self.assertEqual(draw["loop"], "WYV")
        self.assertEqual(draw["length"], 3)
        self.assertTrue(draw["complete"])

    def test_a_draw_that_never_terminates_is_reported_incomplete_not_truncated(self):
        # A cut-off loop is not a length the model chose, and scoring it as one would make an
        # unhealthy <eos> mechanism look like a merely short-loop model.
        model = StubModel(self.tok.encode("W"))
        (draw,) = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                               temperature=0, max_new_tokens=5)
        self.assertFalse(draw["complete"])
        self.assertEqual(draw["length"], 5)

    def test_greedy_returns_exactly_one_candidate(self):
        # Matches the masked and causal samplers: --temperature 0 means one deterministic answer,
        # so a table cannot silently average 20 identical greedy draws as if they were a sample.
        model = StubModel(self.tok.encode("WY") + [self.tok.eos_id])
        draws = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                             temperature=0, num_samples=20)
        self.assertEqual(len(draws), 1)

    def test_sampling_returns_the_requested_number_of_candidates(self):
        model = StubModel(self.tok.encode("WY") + [self.tok.eos_id])
        draws = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                             temperature=1.0, num_samples=7,
                             generator=torch.Generator().manual_seed(0))
        self.assertEqual(len(draws), 7)

    def test_the_encoder_runs_once_for_all_candidates(self):
        # Every candidate must condition on byte-identical framework states, so between-candidate
        # variation is sampling noise rather than a re-encoded prompt. It is also the speed win.
        model = StubModel(self.tok.encode("WY") + [self.tok.eos_id])
        sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                     temperature=1.0, num_samples=12,
                     generator=torch.Generator().manual_seed(0))
        self.assertEqual(model.encode_calls, 1)

    def test_non_terminator_special_tokens_cannot_appear_in_a_loop(self):
        # Left unbanned the model may emit <mask>/<sep>/<pad> mid-loop and have it scored as a
        # residue. Scripted to always want <mask>, the sampler must never produce it.
        model = StubModel([self.tok.mask_id])
        (draw,) = sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                              temperature=0, max_new_tokens=4)
        self.assertNotIn("<mask>", draw["loop"])
        for token in ("<pad>", "<sep>", "<bos>", "<eos>"):
            self.assertNotIn(token, draw["loop"])

    def test_the_decode_budget_is_validated_before_the_model_is_touched(self):
        model = StubModel([self.tok.eos_id])
        with self.assertRaises(ValueError):
            sample_loops(model, self.tok, self.encoder_ids, device=torch.device("cpu"),
                         max_new_tokens=S2S_DECODE_STEPS + 1)
        self.assertEqual(model.calls, 0)


class TrainingGenerationAgreementTests(unittest.TestCase):
    """The encoder input the sampler reads must be the one training built. Byte-identical."""

    def test_generation_rebuilds_the_training_encoder_input_exactly(self):
        import json
        import os
        import tempfile

        chain = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
                 "fwr3": "FFFF", "cdr3": "GGGGG", "fwr4": "HHHH"}
        record = {"heavy": dict(chain, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
                  "light": dict(chain, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
                  "antigen": {"full_aa": "WWWW"}, "split": "train"}
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump([record], handle)
        handle.close()
        try:
            adapter = s2s_adapter()
            dataset = Seq2SeqInfillingDataset(handle.name, adapter, max_len=64, target_region="h3")
            trained = dataset[0]["input_ids"].tolist()
            # Generation path: the framework only, with no access to the native loop or its length.
            framework = dataset.framework_parts(0)
            generated, _ = build_encoder_input(adapter, framework)
            self.assertEqual(trained, generated)
        finally:
            os.unlink(handle.name)


if __name__ == "__main__":
    unittest.main()
