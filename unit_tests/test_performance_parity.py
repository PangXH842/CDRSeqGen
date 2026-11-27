"""Numerical parity tests for batched judge scoring, KV-cached AR decoding, and batched perplexities."""

import unittest
from unittest.mock import Mock

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from benchmark.scorer import Judge
from cdrseqgen.datasets.dataset_ar import build_causal_prompt, causal_adapter
from cdrseqgen.datasets.dataset_common import TokenizerAdapter
from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer
from cdrseqgen.datasets.dataset_s2s import build_encoder_input, s2s_adapter
from cdrseqgen.methods.ar import sampling as ar_sampling
from cdrseqgen.methods.ar.models import transformer as ar_model_mod
from cdrseqgen.methods.mlm.models import transformer as mlm_model_mod
from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood, pseudo_log_likelihood_batch
from cdrseqgen.methods.s2s import sampling as s2s_sampling
from cdrseqgen.methods.s2s.models import transformer as s2s_model_mod


class BatchedJudgeParityTests(unittest.TestCase):
    def test_pseudo_log_likelihood_batch_matches_single_candidate_across_lengths_and_chunks(self):
        torch.manual_seed(0)
        tok = TokenizerAdapter.from_custom(AntibodyTokenizer(), position_type="absolute")
        model = mlm_model_mod.build_model(
            vocab_size=tok.vocab_size, hidden_size=32, num_layers=2, num_attention_heads=4, max_len=64
        )
        model.eval()
        device = torch.device("cpu")

        items = [
            (tok.encode("EVQL") + [tok.mask_id] * 4 + tok.encode("WGQG"), [4, 5, 6, 7], tok.encode("ARDY")),
            (tok.encode("EVQL") + [tok.mask_id] * 6 + tok.encode("WGQG"), [4, 5, 6, 7, 8, 9], tok.encode("ARGGDY")),
            (tok.encode("EVQL") + [tok.mask_id] * 4 + tok.encode("WGQG"), [4, 5, 6, 7], tok.encode("VKFW")),
            (tok.encode("EVQL") + tok.encode("WGQG"), [], []),
        ]
        expected = [
            pseudo_log_likelihood(model, ids, pos, tids, tok.special_ids, tok.mask_id, device)
            for ids, pos, tids in items
        ]
        batched = pseudo_log_likelihood_batch(
            model, items, tok.special_ids, tok.mask_id, device, batch_size=3
        )
        self.assertEqual(len(batched), len(expected))
        for (e_pll, e_ppl), (b_pll, b_ppl) in zip(expected, batched):
            self.assertAlmostEqual(b_pll, e_pll, places=4)
            self.assertAlmostEqual(b_ppl, e_ppl, places=4)

    def test_judge_deduplicates_within_and_across_calls(self):
        judge = Judge.__new__(Judge)
        judge.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer(), position_type="absolute")
        judge.max_len = 64
        judge.batch_size = 16
        judge.device = torch.device("cpu")
        judge.model = object()
        judge._cache = {}
        from cdrseqgen.datasets.dataset_mlm import assemble_masked_input, inject_context
        judge.inject_context = inject_context
        judge.assemble_masked_input = assemble_masked_input

        calls = []

        def fake_batch(model, items, banned_ids, mask_id, device, *, batch_size=256):
            calls.append(len(items))
            return [(-1.0 * len(tids), float(len(tids) + 1)) for _, _, tids in items]

        judge.pseudo_log_likelihood_batch = fake_batch
        rec = {"example_id": "ex1", "pt1": "EVQL", "pt2": "WGQG", "native_length": 4}
        first = judge.score(rec, ["ARDY", "ARDY", "ARGGDY"])
        second = judge.score(rec, ["ARDY", "ARGGDY", "VKFW"])
        self.assertEqual(calls, [2, 1])  # only 2 unique in first call, 1 new in second
        self.assertEqual(first, [5.0, 5.0, 7.0])
        self.assertEqual(second, [5.0, 7.0, 5.0])


class ARAndS2SOptimizationParityTests(unittest.TestCase):
    def test_ar_kv_cache_matches_uncached_hf_gpt2(self):
        torch.manual_seed(7)
        tok = causal_adapter()
        cfg_cached = GPT2Config(
            vocab_size=tok.vocab_size, n_positions=128, n_ctx=128, n_embd=32, n_layer=2, n_head=4,
            use_cache=True, bos_token_id=tok.bos_id, eos_token_id=tok.eos_id,
        )
        model_cached = GPT2LMHeadModel(cfg_cached).eval()
        cfg_uncached = GPT2Config(**{**cfg_cached.to_dict(), "use_cache": False})
        model_uncached = GPT2LMHeadModel(cfg_uncached).eval()
        model_uncached.load_state_dict(model_cached.state_dict())

        prompt, _ = build_causal_prompt(
            tok, dict(pt1="EVQL", pt2="WGQG", target_length=0, max_len=128, partner_chain=None, antigen=None)
        )
        device = torch.device("cpu")

        g1 = torch.Generator(device=device).manual_seed(123)
        g2 = torch.Generator(device=device).manual_seed(123)
        out_cached = ar_sampling.sample_loops(
            model_cached, tok, prompt, device=device, num_samples=4, temperature=1.0, top_p=0.9,
            max_new_tokens=8, generator=g1
        )
        out_uncached = ar_sampling.sample_loops(
            model_uncached, tok, prompt, device=device, num_samples=4, temperature=1.0, top_p=0.9,
            max_new_tokens=8, generator=g2
        )
        self.assertEqual(out_cached, out_uncached)

    def test_ar_batched_perplexities_match_single_item(self):
        torch.manual_seed(11)
        tok = causal_adapter()
        model = ar_model_mod.build_model(
            vocab_size=tok.vocab_size, hidden_size=32, num_layers=2, num_attention_heads=4, max_len=128
        ).eval()
        device = torch.device("cpu")
        p1, _ = build_causal_prompt(
            tok, dict(pt1="EVQL", pt2="WGQG", target_length=0, max_len=128, partner_chain=None, antigen=None)
        )
        p2, _ = build_causal_prompt(
            tok, dict(pt1="QVQLQ", pt2="WGQ", target_length=0, max_len=128, partner_chain=None, antigen=None)
        )
        spans = [(p1, tok.encode("ARDY")), (p1, tok.encode("ARGGFDY")), (p2, tok.encode("VK"))]
        singles = [ar_sampling.perplexities(model, tok, device, [s], batch_size=1)[0] for s in spans]
        batched = ar_sampling.perplexities(model, tok, device, spans, batch_size=2)
        for a, b in zip(singles, batched):
            self.assertAlmostEqual(a, b, places=4)

    def test_s2s_batched_perplexities_match_single_item(self):
        torch.manual_seed(13)
        tok = s2s_adapter()
        model = s2s_model_mod.build_model(
            vocab_size=tok.vocab_size, hidden_size=32, num_layers=2, num_attention_heads=4, max_len=64
        ).eval()
        device = torch.device("cpu")
        e1, _ = build_encoder_input(
            tok, dict(pt1="EVQL", pt2="WGQG", target_length=0, max_len=64, partner_chain=None, antigen=None)
        )
        e2, _ = build_encoder_input(
            tok, dict(pt1="QVQLQ", pt2="WGQ", target_length=0, max_len=64, partner_chain=None, antigen=None)
        )
        spans = [(e1, tok.encode("ARDY")), (e1, tok.encode("ARGGFDY")), (e2, tok.encode("VK"))]
        singles = [s2s_sampling.perplexities(model, tok, device, [s], batch_size=1)[0] for s in spans]
        batched = s2s_sampling.perplexities(model, tok, device, spans, batch_size=2)
        for a, b in zip(singles, batched):
            self.assertAlmostEqual(a, b, places=4)


if __name__ == "__main__":
    unittest.main()
