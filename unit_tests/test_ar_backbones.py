"""The causal backbone registry: dispatch, vocab agreement, and the pretrained-sentinel problem.

Two failure modes are silent and expensive, so both are pinned:

  * **the fallback shadowing a named backbone.** `hf_decoder` matches any string containing "/", so
    if it were registered before a named entry every hub id would resolve to the generic loader and
    look like it worked.
  * **vocab disagreement between tokenizer and model.** The causal vocab is 27 (the shared 25 plus
    <bos>/<eos>); building a model sized 25 makes <eos> unrepresentable, which would silently
    disable the very mechanism that gives an autoregressive model its length.
"""
import unittest

import torch

from cdrseqgen.data.dataset_ar import (CausalInfillingDataset, CausalTokenizer,
                                       causal_collate_fn)
from cdrseqgen.methods.ar import backbones
from cdrseqgen.methods.ar.backbones import REQUIRED_SENTINELS, missing_sentinels, resolve


class DispatchTests(unittest.TestCase):
    def test_scratch_resolves_to_the_custom_decoder(self):
        self.assertEqual(resolve("scratch").id, "scratch")

    def test_a_hub_id_falls_back_to_the_generic_causal_loader(self):
        for name in ("nferruz/ProtGPT2", "hugohrban/progen2-small", "./local_run"):
            self.assertEqual(resolve(name).id, "hf_decoder")

    def test_an_unknown_bare_name_is_refused_with_the_registry_listed(self):
        with self.assertRaises(ValueError) as ctx:
            resolve("not_a_backbone")
        self.assertIn("scratch", str(ctx.exception))

    def test_the_generic_fallback_is_registered_last(self):
        # Registered earlier, it would shadow every named backbone whose id contains a slash.
        flags = [b.fallback for b in backbones.REGISTRY]
        self.assertEqual(flags.count(True), 1)
        self.assertTrue(flags[-1])


class VocabAgreementTests(unittest.TestCase):
    def test_the_scratch_model_is_sized_for_the_causal_vocab_not_the_masked_one(self):
        backbone = resolve("scratch")
        adapter = backbone.build_tokenizer("scratch")
        model = backbone.load_training_model("scratch", hidden_size=32, num_layers=1,
                                             num_attention_heads=4)
        self.assertEqual(adapter.vocab_size, len(CausalTokenizer().vocab))
        self.assertEqual(model.config.vocab_size, adapter.vocab_size)

    def test_eos_is_inside_the_model_vocabulary(self):
        # Sizing to the masked vocab (25) would put <eos> (26) out of range and silently remove the
        # model's ability to end a loop at all.
        backbone = resolve("scratch")
        adapter = backbone.build_tokenizer("scratch")
        model = backbone.load_training_model("scratch", hidden_size=32, num_layers=1,
                                             num_attention_heads=4)
        self.assertLess(adapter.eos_id, model.config.vocab_size)

    def test_the_scratch_adapter_uses_relative_positions(self):
        # Rotary; Findings §8 is why absolute positions are the wrong default for varying prompts.
        self.assertEqual(resolve("scratch").position_type, "relative")
        self.assertEqual(resolve("scratch").build_tokenizer("scratch").position_type, "relative")


class SentinelTests(unittest.TestCase):
    def test_a_plain_lm_vocabulary_is_reported_as_missing_every_sentinel(self):
        class FakeTok:
            def get_vocab(self):
                return {"A": 0, "C": 1, "<|endoftext|>": 2}
        self.assertEqual(missing_sentinels(FakeTok()), list(REQUIRED_SENTINELS))

    def test_a_vocabulary_that_already_has_them_needs_nothing_added(self):
        class FakeTok:
            def get_vocab(self):
                return {t: i for i, t in enumerate(REQUIRED_SENTINELS)}
        self.assertEqual(missing_sentinels(FakeTok()), [])

    def test_the_causal_tokenizer_supplies_all_three_sentinels(self):
        vocab = CausalTokenizer().tok_to_id
        for token in REQUIRED_SENTINELS:
            self.assertIn(token, vocab)


class RoundTripTests(unittest.TestCase):
    def test_a_registry_built_pair_trains_and_reloads(self):
        import json, os, tempfile
        chain = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
                 "fwr3": "FFFF", "cdr3": "GGGGG", "fwr4": "HHHH"}
        rec = {"heavy": dict(chain, full_aa="x"), "light": dict(chain, full_aa="x"), "split": "train"}
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump([rec] * 4, handle)
        handle.close()

        backbone = resolve("scratch")
        adapter = backbone.build_tokenizer("scratch")
        model = backbone.load_training_model("scratch", hidden_size=32, num_layers=1,
                                             num_attention_heads=4)
        ds = CausalInfillingDataset(handle.name, adapter, target_region="h3")
        batch = causal_collate_fn([ds[i] for i in range(4)], pad_token_id=adapter.pad_id)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        self.assertTrue(torch.isfinite(out["loss"]))

        directory = tempfile.mkdtemp()
        model.save_pretrained(directory)
        reloaded = backbone.load_pretrained_model(directory)
        with torch.no_grad():
            before = model(input_ids=batch["input_ids"][:1])["logits"]
            after = reloaded(input_ids=batch["input_ids"][:1])["logits"]
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        os.unlink(handle.name)


if __name__ == "__main__":
    unittest.main()
