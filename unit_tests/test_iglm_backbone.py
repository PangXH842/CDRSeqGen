"""Native IgLM integration, using tiny local GPT-2 weights and controlled draws only."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from cdrseqgen.datasets.dataset_ar import (
    CausalInfillingDataset,
    build_causal_prompt,
    causal_adapter,
    span_layout_metadata,
)
from cdrseqgen.methods.ar.backbones import resolve
from cdrseqgen.methods.ar.evaluation import perplexities
from cdrseqgen.methods.ar.models import iglm
from cdrseqgen.methods.ar.sampling import sample_loops


VOCAB = (
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    *"ACDEFGHIKLMNPQRSTVWY",
    "[CAMEL]", "[HUMAN]", "[MOUSE]", "[RABBIT]", "[RAT]", "[RHESUS]",
    "[HEAVY]", "[LIGHT]",
)
FRAMEWORK = dict(pt1="AC", pt2="FG", target="DE", target_length=2,
                 target_region="h3", partner_chain=None, antigen=None, max_len=192)


class ControlledGenerator:
    """Supply known raw completions while retaining the native generate() arguments."""

    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = []
        self.config = SimpleNamespace(n_positions=192, max_position_embeddings=192)

    def eval(self):
        return self

    def generate(self, input_ids=None, **kwargs):
        self.calls.append(kwargs["generation_config"].to_dict())
        completion = self.completions[(len(self.calls) - 1) % len(self.completions)]
        if completion is None:
            # Exercise the adapter's explicit-generator bridge, not a model's weights.
            completion = [int(torch.randint(5, 25, (1,)).item()), 2]
        suffix = torch.tensor([completion], device=input_ids.device, dtype=input_ids.dtype)
        return torch.cat((input_ids, suffix), dim=1)


class IgLMBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "tiny-iglm"
        cls.source.mkdir()
        (cls.source / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
        config = GPT2Config(
            vocab_size=33, n_positions=192, n_ctx=192, n_embd=16, n_layer=1,
            n_head=4, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
            bos_token_id=50256, eos_token_id=50256,
        )
        config.cdrseqgen_backbone = "iglm"
        model = GPT2LMHeadModel(config)
        with torch.no_grad():
            model.get_input_embeddings().weight[5].fill_(0.314)
        model.save_pretrained(cls.source)
        tokenizer = iglm.load_tokenizer(str(cls.source))
        config.span_layout = span_layout_metadata(tokenizer)
        config.save_pretrained(cls.source)
        tokenizer.save_pretrained(cls.source)
        (cls.source / "span_layout.json").write_text(json.dumps(config.span_layout))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def tokenizer(self):
        return resolve(str(self.source)).build_tokenizer(str(self.source))

    def test_aliases_and_explicit_local_sources_resolve_before_hf_fallback(self):
        for name in ("IgLM", "iglm", "IgLM-S", "iglm-s", str(self.source),
                     "iglm:" + str(self.source)):
            with self.subTest(source=name):
                self.assertEqual(resolve(name).id, "iglm")
        self.assertEqual(resolve("somewhere/other-causal-model").id, "hf_decoder")
        self.assertEqual(resolve("scratch").id, "scratch")

    def test_native_ids_and_embedding_size_are_preserved_without_added_tokens(self):
        tokenizer = self.tokenizer()
        self.assertEqual(tokenizer.backend_tokenizer.get_vocab(),
                         {token: index for index, token in enumerate(VOCAB)})
        self.assertEqual(tokenizer.vocab_size, 33)
        self.assertEqual(tokenizer.encode("ACD"), [5, 6, 7])
        self.assertEqual(tokenizer.decode([5, 6, 7]), "ACD")
        self.assertEqual(tokenizer.eos_id, 2)
        self.assertEqual(tokenizer.pad_id, 0)
        self.assertFalse(tokenizer.uses_special_tokens)
        self.assertEqual(set(tokenizer.special_ids), {0, 1, 2, 3, 4, *range(25, 33)})
        with self.assertRaises(ValueError):
            tokenizer.encode("ACX")

    def test_prompt_and_answer_follow_the_native_infill_format(self):
        tokenizer = self.tokenizer()
        prompt, _ = build_causal_prompt(tokenizer, FRAMEWORK)
        self.assertEqual(prompt, [31, 26, 5, 6, 4, 9, 10, 3])
        self.assertEqual(tokenizer.span_layout.build_answer(tokenizer, [7, 8]), [7, 8, 2])
        self.assertEqual(span_layout_metadata(tokenizer)["n_layout_tokens"], 5)

    def test_chain_species_and_explicit_chain_conflicts_are_validated(self):
        tokenizer = self.tokenizer()
        light, _ = build_causal_prompt(tokenizer, dict(FRAMEWORK, target_region="l3"))
        self.assertEqual(light[:2], [32, 26])
        mouse, _ = build_causal_prompt(tokenizer, dict(FRAMEWORK, species_token="[MOUSE]"))
        self.assertEqual(mouse[:2], [31, 27])
        explicit = dict(FRAMEWORK, chain_token="[LIGHT]", target_region=None)
        self.assertEqual(build_causal_prompt(tokenizer, explicit)[0][:2], [32, 26])
        for change in (dict(target_region=None), dict(chain_token="[LIGHT]"),
                       dict(species_token="[UNKNOWN]")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_causal_prompt(tokenizer, dict(FRAMEWORK, **change))

    def test_native_target_and_target_length_do_not_change_generation_prompt(self):
        tokenizer = self.tokenizer()
        before = build_causal_prompt(tokenizer, FRAMEWORK)[0]
        after = build_causal_prompt(tokenizer, dict(FRAMEWORK, target="W" * 63,
                                                    target_length=63))[0]
        self.assertEqual(before, after)

    def test_native_context_and_capacity_are_enforced_before_generation(self):
        tokenizer = self.tokenizer()
        for change in (dict(partner_chain="AC"), dict(antigen="AC"),
                       dict(max_len=193), dict(max_len=100)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_causal_prompt(tokenizer, dict(FRAMEWORK, **change))

    def test_training_rejects_unsupported_context_before_model_construction(self):
        backbone = resolve(str(self.source))
        model_args = SimpleNamespace(model_type=str(self.source))
        for flag in ("include_partner_chain", "include_antigen"):
            data = dict(include_partner_chain=False, include_antigen=False, max_len=192)
            data[flag] = True
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                backbone.validate_training_args(model_args, SimpleNamespace(**data))
        with self.assertRaises(ValueError):
            backbone.validate_training_args(model_args, SimpleNamespace(
                include_partner_chain=False, include_antigen=False, max_len=193))

    def dataset(self, target_region="h3"):
        chain = dict(fwr1="AC", cdr1="DE", fwr2="FG", cdr2="HI", fwr3="KL",
                     cdr3="MNP", fwr4="QR", full_aa="ACDEFGHIKLMNPQR")
        corpus = self.root / "corpus.json"
        corpus.write_text(json.dumps([dict(heavy=chain, light=chain, species="human", split="train")]))
        return CausalInfillingDataset(corpus, self.tokenizer(), target_region=target_region, max_len=192)

    def test_heavy_and_light_training_labels_match_generation_layout(self):
        for target_region, chain_id in (("h3", 31), ("l3", 32)):
            with self.subTest(target_region=target_region):
                dataset = self.dataset(target_region)
                item = dataset[0]
                tokenizer = dataset.tokenizer
                prompt = build_causal_prompt(tokenizer, dataset.framework_parts(0))[0]
                self.assertEqual(prompt[:2], [chain_id, 26])
                self.assertEqual(item["input_ids"].tolist()[:item["prompt_len"]], prompt)
                self.assertEqual(item["labels"].tolist(),
                                 [-100] * len(prompt) + tokenizer.encode("MNP") + [2])

    def test_loaded_native_weights_and_shifted_answer_loss_match_the_checkpoint(self):
        backbone = resolve(str(self.source))
        tokenizer = self.tokenizer()
        model = backbone.load_training_model(str(self.source), vocab_size=33, max_len=192)
        self.assertEqual(model.get_input_embeddings().num_embeddings, 33)
        torch.testing.assert_close(model.get_input_embeddings().weight[5], torch.full((16,), 0.314))
        self.assertEqual(model.config.eos_token_id, 2)
        self.assertEqual(model.config.pad_token_id, 0)
        self.assertEqual(model.generation_config.eos_token_id, 2)
        self.assertEqual(model.generation_config.pad_token_id, 0)
        dataset = self.dataset()
        item = dataset[0]
        ids = item["input_ids"][None]
        labels = item["labels"][None]
        output = model(input_ids=ids, labels=labels)
        selected = labels[:, 1:] != -100
        expected = -output.logits[:, :-1].log_softmax(-1)[selected].gather(
            1, labels[:, 1:][selected, None]).mean()
        torch.testing.assert_close(output.loss, expected)
        output.loss.backward()
        self.assertGreater(model.get_input_embeddings().weight.grad.abs().sum().item(), 0)
        prompt = ids[0, :item["prompt_len"]].tolist()
        score = perplexities(model, tokenizer, "cpu", [(prompt, tokenizer.encode("MNP"))])[0]
        self.assertAlmostEqual(score, float(expected.detach().exp()), places=4)

    def test_saved_tokenizer_model_and_layout_round_trip(self):
        model = iglm.load_model(str(self.source))
        tokenizer = self.tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            model.config.span_layout = span_layout_metadata(tokenizer)
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            backbone = resolve(str(checkpoint))
            restored_tokenizer = backbone.build_tokenizer(str(checkpoint), checkpoint_path=checkpoint)
            restored_model = backbone.load_pretrained_model(checkpoint)
            self.assertEqual(restored_tokenizer.encode("ACDE"), tokenizer.encode("ACDE"))
            self.assertEqual(span_layout_metadata(restored_tokenizer), span_layout_metadata(tokenizer))
            for name, weights in model.state_dict().items():
                torch.testing.assert_close(restored_model.state_dict()[name], weights, rtol=0, atol=0)
            metadata = span_layout_metadata(tokenizer)
            metadata["version"] += 1
            (checkpoint / "span_layout.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "layout mismatch|Layout mismatch"):
                backbone.build_tokenizer(str(checkpoint), checkpoint_path=checkpoint)

    def test_converted_checkpoint_cannot_silently_lose_its_layout_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            model = iglm.load_model(str(self.source))
            if hasattr(model.config, "span_layout"):
                del model.config.span_layout
            model.save_pretrained(checkpoint)
            self.tokenizer().backend_tokenizer.save_pretrained(checkpoint)
            with self.assertRaisesRegex(ValueError, "span layout metadata"):
                resolve(str(checkpoint)).build_tokenizer(str(checkpoint), checkpoint_path=checkpoint)

    def draw(self, completions, **options):
        tokenizer = self.tokenizer()
        model = ControlledGenerator(completions)
        prompt = build_causal_prompt(tokenizer, FRAMEWORK)[0]
        defaults = dict(device="cpu", temperature=1.0, top_k=0, top_p=0.9,
                        num_samples=1, max_new_tokens=8,
                        generator=torch.Generator().manual_seed(7))
        defaults.update(options)
        return sample_loops(model, tokenizer, prompt, **defaults), model

    def test_native_sampling_policy_keeps_marker_permissions_and_natural_stopping(self):
        results, model = self.draw([[5, 6, 2]])
        self.assertEqual(results[0]["loop"], "AC")
        self.assertTrue(results[0]["complete"])
        controls = model.calls[0]
        self.assertEqual(controls["eos_token_id"], 2)
        self.assertEqual(controls["pad_token_id"], 0)
        self.assertIsNone(controls.get("forced_eos_token_id"))
        self.assertEqual({entry[0] for entry in controls["bad_words_ids"]},
                         {0, 1, *range(25, 33)})
        self.assertEqual(controls["max_new_tokens"], 8)
        self.assertEqual(controls["top_k"], 0)
        self.assertEqual(controls["top_p"], 0.9)

    def test_fixed_attempts_retain_duplicates_and_do_not_refill_invalid_draws(self):
        results, model = self.draw([[5, 6, 2], [4, 5, 2]], num_samples=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual([record["loop"] for record in results if record["complete"]], ["AC", "AC"])
        self.assertFalse(results[1]["complete"])
        self.assertTrue(results[1].get("failure"))

    def test_unterminated_draws_and_extra_layout_markers_are_not_valid_loops(self):
        for completion in ([5, 6], [3, 5, 2], [4, 5, 2]):
            with self.subTest(completion=completion):
                results, model = self.draw([completion])
                self.assertFalse(results[0]["complete"])
                self.assertEqual(len(model.calls), 1)

    def test_greedy_returns_one_attempt(self):
        results, model = self.draw([[5, 6, 2]], temperature=0.0, num_samples=5)
        self.assertEqual(len(results), 1)
        self.assertFalse(model.calls[0]["do_sample"])

    def test_explicit_generator_advances_without_changing_ambient_cpu_rng(self):
        generator = torch.Generator().manual_seed(19)
        before_generator = generator.get_state().clone()
        before_ambient = torch.random.get_rng_state().clone()
        first, _ = self.draw([None], generator=generator, num_samples=3)
        torch.testing.assert_close(torch.random.get_rng_state(), before_ambient, rtol=0, atol=0)
        self.assertFalse(torch.equal(before_generator, generator.get_state()))
        second, _ = self.draw([None], generator=torch.Generator().manual_seed(19), num_samples=3)
        self.assertEqual([record["loop"] for record in first], [record["loop"] for record in second])

    def test_scratch_still_uses_the_existing_vocabulary_and_layout(self):
        tokenizer = resolve("scratch").build_tokenizer("scratch")
        self.assertEqual(tokenizer.vocab_size, 27)
        self.assertEqual(span_layout_metadata(tokenizer), span_layout_metadata(causal_adapter()))


if __name__ == "__main__":
    unittest.main()
