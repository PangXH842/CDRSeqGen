"""Sampler/output contracts exercised with deterministic CPU models, without weights."""
from types import SimpleNamespace
import unittest

import torch

from cdrseqgen.datasets.dataset_ar import causal_adapter
from cdrseqgen.datasets.dataset_s2s import s2s_adapter
from cdrseqgen.methods.ar.sampling import perplexities as ar_perplexities
from cdrseqgen.methods.ar.sampling import sample_loops as ar_sample
from cdrseqgen.methods.common.sampling import warp_logits
from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood, sample_masked_span
from cdrseqgen.methods.s2s.sampling import perplexities as s2s_perplexities
from cdrseqgen.methods.s2s.sampling import sample_loops as s2s_sample


def output(logits, form):
    if form == "dict":
        return {"logits": logits}
    if form == "object":
        return SimpleNamespace(logits=logits)
    return (logits, "unused cached state")


class MaskedModel(torch.nn.Module):
    def __init__(self, form):
        super().__init__()
        self.form = form

    def forward(self, input_ids):
        batch, width = input_ids.shape
        logits = torch.arange(9, dtype=torch.float32).expand(batch, width, 9).clone()
        return output(logits, self.form)


class EosModel(torch.nn.Module):
    """Emit A, C, EOS for decoding; also supply deterministic self-scoring logits."""

    def __init__(self, form, tokenizer, prompt_length):
        super().__init__()
        self.form = form
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.script = tokenizer.encode("AC") + [tokenizer.eos_id]

    def encode(self, input_ids, attention_mask=None):
        return torch.zeros(*input_ids.shape, 4)

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        ids = input_ids if decoder_input_ids is None else decoder_input_ids
        step = ids.shape[1] - (self.prompt_length if decoder_input_ids is None else 1)
        token = self.script[min(max(step, 0), len(self.script) - 1)]
        logits = torch.zeros(*ids.shape, self.tokenizer.vocab_size)
        logits[:, -1, token] = 20.0
        return output(logits, self.form)


class StaggeredModel(EosModel):
    """Rows stop after zero, one and two residues, keeping completed rows in the batch."""

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        ids = input_ids if decoder_input_ids is None else decoder_input_ids
        step = ids.shape[1] - (self.prompt_length if decoder_input_ids is None else 1)
        logits = torch.zeros(*ids.shape, self.tokenizer.vocab_size)
        for row in range(ids.shape[0]):
            token = self.tokenizer.eos_id if step >= row else self.script[0]
            logits[row, -1, token] = 20.0
        return {"logits": logits}


class ExtremeModel(EosModel):
    """Make candidate residues extremely unlikely or produce non-finite loss."""

    def __init__(self, tokenizer, peak):
        super().__init__("dict", tokenizer, 4)
        self.peak = peak

    def forward(self, input_ids=None, decoder_input_ids=None, **kwargs):
        ids = input_ids if decoder_input_ids is None else decoder_input_ids
        logits = torch.zeros(*ids.shape, self.tokenizer.vocab_size)
        logits[..., self.tokenizer.encode("Y")[0]] = self.peak
        return {"logits": logits}


class OutputFormatTests(unittest.TestCase):
    def test_both_mlm_decoders_preserve_draws_and_rng_with_tuple_outputs(self):
        tokenizer = SimpleNamespace(special_ids={0, 1}, mask_id=1)
        for decoder in ("independent", "iterative"):
            expected = None
            for form in ("dict", "object", "tuple"):
                with self.subTest(decoder=decoder, form=form):
                    generator = torch.Generator().manual_seed(73)
                    samples = sample_masked_span(
                        MaskedModel(form), tokenizer, "cpu", generator,
                        [2, 1, 1, 3], [1, 2], decoding=decoder, num_samples=4,
                        temperature=0.8, top_k=3, top_p=0.9)
                    actual = (samples, generator.get_state())
                    if expected is None:
                        expected = actual
                    self.assertEqual(samples, expected[0])
                    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_eos_decoders_preserve_completion_and_rng_with_tuple_outputs(self):
        for tokenizer, sampler in ((causal_adapter(), ar_sample), (s2s_adapter(), s2s_sample)):
            for temperature in (0.0, 1.0):
                expected = None
                for form in ("dict", "object", "tuple"):
                    with self.subTest(sampler=sampler.__module__, temperature=temperature, form=form):
                        generator = torch.Generator().manual_seed(73)
                        draws = sampler(
                            EosModel(form, tokenizer, 4), tokenizer, [5, 4, 6, 3],
                            device="cpu", num_samples=3, temperature=temperature,
                            max_new_tokens=5, generator=generator)
                        actual = (draws, generator.get_state())
                        if expected is None:
                            expected = actual
                        self.assertEqual(draws, expected[0])
                        self.assertEqual(len(draws), 1 if temperature == 0 else 3)
                        self.assertTrue(all(d["complete"] and d["loop"] == "AC" for d in draws))
                        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_self_scores_are_identical_for_all_supported_output_formats(self):
        for tokenizer, scorer in ((causal_adapter(), ar_perplexities), (s2s_adapter(), s2s_perplexities)):
            expected = None
            for form in ("dict", "object", "tuple"):
                with self.subTest(scorer=scorer.__module__, form=form):
                    spans = [([5, 4, 6, 3], tokenizer.encode(loop)) for loop in ("AC", "DE")]
                    scores = scorer(EosModel(form, tokenizer, 4), tokenizer, "cpu", spans)
                    if expected is None:
                        expected = scores
                    self.assertEqual(scores, expected)
                    self.assertTrue(all(score is not None for score in scores))
        expected = None
        for form in ("dict", "object", "tuple"):
            with self.subTest(scorer="MLM", form=form):
                scores = pseudo_log_likelihood(MaskedModel(form), [2, 1, 1, 3], [1, 2],
                                               [5, 6], {0, 1}, 1, "cpu")
                if expected is None:
                    expected = scores
                self.assertEqual(scores, expected)


class BoundaryTests(unittest.TestCase):
    def test_empty_mlm_spans_skip_the_model_and_preserve_rng(self):
        def forbidden_model(**kwargs):
            self.fail("an empty span must not invoke the sequence model")

        tokenizer = SimpleNamespace(special_ids={0, 1}, mask_id=1)
        for decoder in ("independent", "iterative"):
            for temperature in (0.0, 1.0):
                with self.subTest(decoder=decoder, temperature=temperature):
                    generator = torch.Generator().manual_seed(73)
                    before = generator.get_state().clone()
                    samples = sample_masked_span(
                        forbidden_model, tokenizer, "cpu", generator, [2, 3], [],
                        decoding=decoder, num_samples=4, temperature=temperature,
                        top_k=0, top_p=1.0)
                    self.assertEqual(samples, [[]] * (1 if temperature == 0 else 4))
                    torch.testing.assert_close(generator.get_state(), before, rtol=0, atol=0)

    def test_pll_rejects_missing_or_extra_candidate_tokens_before_scoring(self):
        def forbidden_model(**kwargs):
            self.fail("a malformed candidate must fail before model execution")

        for positions, tokens in (([1, 2], [5]), ([1], [5, 6]), ([], [5])):
            with self.subTest(positions=positions, tokens=tokens), self.assertRaisesRegex(
                    ValueError, "token_ids must contain one token per mask position"):
                pseudo_log_likelihood(forbidden_model, [2, 1, 1, 3], positions,
                                      tokens, {0, 1}, 1, "cpu")

    def test_eos_rows_stop_independently_without_changing_rng_consumption(self):
        for tokenizer, sampler in ((causal_adapter(), ar_sample), (s2s_adapter(), s2s_sample)):
            with self.subTest(sampler=sampler.__module__):
                generator = torch.Generator().manual_seed(73)
                draws = sampler(
                    StaggeredModel("dict", tokenizer, 4), tokenizer, [5, 4, 6, 3],
                    device="cpu", num_samples=3, temperature=1.0, top_k=1,
                    max_new_tokens=5, generator=generator)
                self.assertEqual([d["loop"] for d in draws], ["", "A", "AA"])
                self.assertTrue(all(d["complete"] for d in draws))
                expected_rng = torch.Generator().manual_seed(73)
                for step in range(3):
                    probabilities = torch.zeros(3, tokenizer.vocab_size)
                    for row in range(3):
                        token = tokenizer.eos_id if step >= row else tokenizer.encode("A")[0]
                        probabilities[row, token] = 1.0
                    torch.multinomial(probabilities, 1, generator=expected_rng)
                torch.testing.assert_close(generator.get_state(), expected_rng.get_state(),
                                           rtol=0, atol=0)

    def test_eos_self_scorers_handle_overflow_and_nonfinite_losses(self):
        for tokenizer, scorer in ((causal_adapter(), ar_perplexities), (s2s_adapter(), s2s_perplexities)):
            for peak in (1000.0, float("nan"), float("inf")):
                with self.subTest(scorer=scorer.__module__, peak=peak):
                    scores = scorer(ExtremeModel(tokenizer, peak), tokenizer, "cpu",
                                    [([5, 4, 6, 3], tokenizer.encode("AC"))])
                    self.assertEqual(scores, [None])

    def test_nucleus_keeps_the_crossing_token_and_top_k_keeps_cutoff_ties(self):
        nucleus = warp_logits(torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05])), top_p=0.6)
        self.assertEqual(torch.isfinite(nucleus).nonzero().flatten().tolist(), [0, 1])
        tied = warp_logits(torch.tensor([4.0, 3.0, 3.0, 1.0]), top_k=2)
        self.assertEqual(torch.isfinite(tied).nonzero().flatten().tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
