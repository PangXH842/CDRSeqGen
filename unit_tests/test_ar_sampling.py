"""Causal decoding: the properties that make a sampled loop trustworthy.

Uses a stub model returning controlled logits, so each behaviour is asserted exactly rather than
inferred from a real model's noise.
"""
import unittest

import torch

from cdrseqgen.datasets.dataset_ar import (
    LOOP_LENGTH_RESERVE,
    CausalTokenizer,
    causal_adapter,
)
from cdrseqgen.methods.ar.sampling import _left_pad, sample_loops


class StubModel:
    """Emits a fixed token sequence, then <eos> forever. `forced` is a list of token ids."""

    def __init__(self, vocab_size, forced, eos_id):
        self.vocab_size, self.forced, self.eos_id = vocab_size, list(forced), eos_id
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, **_):
        batch = input_ids.shape[0]
        token = self.forced[self.calls] if self.calls < len(self.forced) else self.eos_id
        self.calls += 1
        logits = torch.full((batch, input_ids.shape[1], self.vocab_size), -10.0)
        logits[:, -1, token] = 10.0
        return {"logits": logits}


class NeverStops:
    """Always predicts the same residue — used to exercise the max_new_tokens cutoff."""

    def __init__(self, vocab_size, token):
        self.vocab_size, self.token = vocab_size, token

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, **_):
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.vocab_size), -10.0)
        logits[:, -1, self.token] = 10.0
        return {"logits": logits}


class LeftPadTests(unittest.TestCase):
    def test_every_rows_last_token_lands_at_index_minus_one(self):
        # Right padding would put the newest token at a different index per row, so a single
        # logits[:, -1] read would be wrong for all but the longest prompt.
        ids, mask = _left_pad([[1, 2, 3], [4, 5], [6]], pad_id=0, device="cpu")
        self.assertEqual(ids.shape, (3, 3))
        self.assertEqual(ids[:, -1].tolist(), [3, 5, 6])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [0, 1, 1], [0, 0, 1]])


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.tok = CausalTokenizer()
        self.adapter = causal_adapter(self.tok)
        self.aa = [self.tok.tok_to_id[c] for c in "ACDE"]

    def test_the_loop_is_the_tokens_before_eos(self):
        model = StubModel(len(self.tok.vocab), self.aa, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2, 3], device="cpu", temperature=0,
                           max_new_tokens=20)
        self.assertEqual(out[0]["loop"], "ACDE")
        self.assertEqual(out[0]["length"], 4)
        self.assertTrue(out[0]["complete"])

    def test_default_budget_can_finish_the_full_reserved_loop(self):
        residue = self.tok.tok_to_id["A"]
        model = StubModel(
            len(self.tok.vocab), [residue] * LOOP_LENGTH_RESERVE, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", temperature=0)
        self.assertTrue(out[0]["complete"])
        self.assertEqual(out[0]["length"], LOOP_LENGTH_RESERVE)

    def test_greedy_returns_exactly_one_candidate(self):
        # Matches the masked sampler's convention so --temperature 0 behaves the same either side.
        model = StubModel(len(self.tok.vocab), self.aa, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", num_samples=8,
                           temperature=0, max_new_tokens=20)
        self.assertEqual(len(out), 1)

    def test_sampling_returns_the_requested_number_of_candidates(self):
        model = StubModel(len(self.tok.vocab), self.aa, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", num_samples=5,
                           temperature=1.0, max_new_tokens=20,
                           generator=torch.Generator().manual_seed(0))
        self.assertEqual(len(out), 5)

    def test_a_draw_that_never_emits_eos_is_marked_incomplete(self):
        # A truncated loop is not a length the model chose, so it must not pass as a short loop.
        model = NeverStops(len(self.tok.vocab), self.tok.tok_to_id["A"])
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", temperature=0,
                           max_new_tokens=6)
        self.assertFalse(out[0]["complete"])
        self.assertEqual(out[0]["length"], 6)

    def test_special_tokens_cannot_appear_inside_a_loop(self):
        # The model is pushed hard toward <mask>; banning must override it, leaving only residues.
        model = StubModel(len(self.tok.vocab), [self.tok.mask_id] * 3, self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", temperature=0,
                           max_new_tokens=10)
        residues = set("ACDEFGHIKLMNPQRSTVWY")
        self.assertTrue(set(out[0]["loop"]) <= residues, out[0]["loop"])

    def test_eos_stays_reachable_despite_the_ban(self):
        # Banning every special would make termination impossible and every draw incomplete.
        model = StubModel(len(self.tok.vocab), self.aa[:2], self.adapter.eos_id)
        out = sample_loops(model, self.adapter, [1, 2], device="cpu", temperature=0,
                           max_new_tokens=10)
        self.assertTrue(out[0]["complete"])
        self.assertEqual(out[0]["length"], 2)


if __name__ == "__main__":
    unittest.main()
