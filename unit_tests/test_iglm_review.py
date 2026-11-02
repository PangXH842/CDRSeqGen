"""IgLM review regressions, using temporary vocabularies and tiny CPU checkpoints."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, sentinel

import torch
from safetensors.torch import load_file, save_file
from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel

import generate
from cdrseqgen.datasets.dataset_ar import (
    CDRSEQGEN_CAUSAL, CausalInfillingDataset, build_causal_prompt, causal_adapter,
    span_layout_metadata, validate_layout,
)
from cdrseqgen.methods.ar.backbones import resolve
from cdrseqgen.methods.ar.evaluation import run_generative_eval
from cdrseqgen.methods.ar.generation import generate_case, prepare_generation
from cdrseqgen.methods.ar.models import iglm


class AttemptModel:
    """Return controlled complete, malformed and unterminated native draws."""

    def __init__(self, completions):
        self.completions = iter(completions)
        self.config = SimpleNamespace(n_positions=128)

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        answer = torch.tensor([next(self.completions)], dtype=input_ids.dtype,
                              device=input_ids.device)
        return torch.cat((input_ids, answer), dim=1)


class IgLMCheckpointReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source"
        cls.source.mkdir()
        config = GPT2Config(
            vocab_size=33, n_positions=128, n_ctx=128, n_embd=8, n_layer=1, n_head=2,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
            bos_token_id=None, eos_token_id=2, pad_token_id=0,
        )
        config.cdrseqgen_backbone = "iglm"
        (cls.source / "vocab.txt").write_text("\n".join(iglm.NATIVE_VOCAB) + "\n")
        config.save_pretrained(cls.source)
        tokenizer = iglm.load_tokenizer(cls.source)
        config.span_layout = span_layout_metadata(tokenizer)
        model = GPT2LMHeadModel(config)
        model.save_pretrained(cls.source)
        (cls.source / "span_layout.json").write_text(json.dumps(config.span_layout))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def variant(self):
        directory = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(directory.cleanup)
        target = Path(directory.name) / "checkpoint"
        shutil.copytree(self.source, target)
        return target

    def test_missing_learned_parameter_is_rejected_instead_of_initialized(self):
        checkpoint = self.variant()
        weights = load_file(checkpoint / "model.safetensors")
        del weights["transformer.h.0.ln_1.weight"]
        save_file(weights, checkpoint / "model.safetensors", metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "weights.*architecture"):
            iglm.load_model(checkpoint)

    def test_unexpected_learned_parameter_is_rejected(self):
        checkpoint = self.variant()
        weights = load_file(checkpoint / "model.safetensors")
        weights["unregistered_projection.weight"] = torch.ones(8, 8)
        save_file(weights, checkpoint / "model.safetensors", metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "weights.*architecture"):
            iglm.load_model(checkpoint)

    def test_native_style_bin_may_contain_legacy_attention_buffers(self):
        checkpoint = self.variant()
        weights = load_file(checkpoint / "model.safetensors")
        expected = weights["transformer.wte.weight"].clone()
        weights["transformer.h.0.attn.bias"] = torch.tril(
            torch.ones(1, 1, 128, 128, dtype=torch.uint8))
        weights["transformer.h.0.attn.masked_bias"] = torch.tensor(-1e4)
        torch.save(weights, checkpoint / "pytorch_model.bin")
        (checkpoint / "model.safetensors").unlink()
        restored = iglm.load_model(checkpoint)
        torch.testing.assert_close(restored.get_input_embeddings().weight, expected, rtol=0, atol=0)

    def test_training_and_tokenizer_loading_reject_changed_or_stripped_layouts(self):
        for corruption in ("changed", "stripped"):
            with self.subTest(corruption=corruption):
                checkpoint = self.variant()
                config = json.loads((checkpoint / "config.json").read_text())
                if corruption == "changed":
                    layout = json.loads((checkpoint / "span_layout.json").read_text())
                    layout["version"] += 1
                    (checkpoint / "span_layout.json").write_text(json.dumps(layout))
                else:
                    config.pop("span_layout")
                    (checkpoint / "span_layout.json").unlink()
                    (checkpoint / "config.json").write_text(json.dumps(config))
                source = "iglm:" + str(checkpoint)
                backbone = resolve(source)
                with self.assertRaisesRegex(ValueError, "layout"):
                    backbone.build_tokenizer(source)
                with self.assertRaisesRegex(ValueError, "layout"):
                    backbone.load_training_model(source, vocab_size=33, max_len=128)

    def test_checkpoint_sampling_policy_cannot_be_silently_reinterpreted(self):
        checkpoint = self.variant()
        config = json.loads((checkpoint / "config.json").read_text())
        config["causal_sampling"] = dict(iglm.SAMPLING_METADATA, forced_eos=True)
        (checkpoint / "config.json").write_text(json.dumps(config))
        source = "iglm:" + str(checkpoint)
        backbone = resolve(source)
        with self.assertRaisesRegex(ValueError, "sampling policy mismatch"):
            backbone.build_tokenizer(source)
        with self.assertRaisesRegex(ValueError, "sampling policy mismatch"):
            backbone.load_training_model(source, vocab_size=33, max_len=128)

    def test_saved_generation_options_do_not_change_the_registered_sampling_protocol(self):
        model = iglm.load_model(self.source)
        tokenizer = resolve(str(self.source)).build_tokenizer(str(self.source))
        prompt, _ = build_causal_prompt(tokenizer, dict(
            pt1="AC", pt2="FG", target_region="h3", partner_chain=None, antigen=None,
            max_len=128))

        def draw():
            return iglm.sample_loops(
                model, tokenizer, prompt, device="cpu", num_samples=2,
                temperature=1.0, top_k=0, top_p=0.9, max_new_tokens=4,
                generator=torch.Generator().manual_seed(613))

        expected = draw()
        model.generation_config = GenerationConfig(
            eos_token_id=2, pad_token_id=0, min_new_tokens=100, repetition_penalty=7.0,
            suppress_tokens=list(range(33)), return_dict_in_generate=True, output_scores=True)
        actual = draw()
        self.assertEqual(actual, expected)

    def test_failed_draw_restores_model_configuration_and_ambient_rng(self):
        model = iglm.load_model(self.source)
        tokenizer = resolve(str(self.source)).build_tokenizer(str(self.source))
        prompt, _ = build_causal_prompt(tokenizer, dict(
            pt1="AC", pt2="FG", target_region="h3", partner_chain=None, antigen=None,
            max_len=128))
        original_config = model.generation_config
        ambient_rng = torch.get_rng_state().clone()
        generator = torch.Generator().manual_seed(72)
        generator_state = generator.get_state().clone()
        with patch.object(model, "generate", side_effect=RuntimeError("controlled failure")):
            with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                iglm.sample_loops(model, tokenizer, prompt, device="cpu", max_new_tokens=4,
                                  generator=generator)
        self.assertIs(model.generation_config, original_config)
        self.assertTrue(torch.equal(torch.get_rng_state(), ambient_rng))
        self.assertFalse(torch.equal(generator.get_state(), generator_state))

    def test_generation_records_policy_conditioning_and_each_attempt(self):
        tokenizer = resolve(str(self.source)).build_tokenizer(str(self.source))
        args = SimpleNamespace(
            decoding=None, min_length=None, max_length=None, length_temperature=1.0,
            max_new_tokens=4, max_len=128, partner_chain=None, antigen=None,
            num_samples=3, temperature=1.0, top_k=0, top_p=0.9, skip_perplexity=True,
            seed=73, target_loop=None)
        prepare_generation(args, [], {"target_region": "h3"})
        model = AttemptModel([[5, 2], [4, 5, 2], [5, 5]])
        record = generate_case(
            model, tokenizer, "cpu", args, torch.Generator().manual_seed(73),
            dict(pt1="AC", pt2="FG", species_token="mouse"), None, lambda *a: None)
        self.assertEqual(record["decoding"]["protocol"], iglm.SAMPLING_METADATA)
        self.assertEqual(record["conditioning"],
                         {"chain_token": "[HEAVY]", "species_token": "[MOUSE]"})
        self.assertEqual(record["generation_validity"], dict(
            attempts=3, complete_nonempty=1, complete_empty=0, invalid=1, incomplete=1))
        self.assertEqual([attempt["failure"] for attempt in record["generation_attempts"]],
                         [None, "invalid_token_layout", "missing_terminator"])

    def test_evaluation_keeps_failed_framework_attempts_outside_scored_population(self):
        tokenizer = resolve(str(self.source)).build_tokenizer(str(self.source))
        checkpoint = self.variant()
        chain = dict(fwr1="AC", cdr1="DE", fwr2="FG", cdr2="HI", fwr3="KL",
                     cdr3="MN", fwr4="PQ", full_aa="ACDEFGHIKLMNPQ")
        corpus = checkpoint / "corpus.json"
        corpus.write_text(json.dumps([
            dict(heavy=chain, species="human", split="validation") for _ in range(2)]))
        dataset = CausalInfillingDataset(corpus, tokenizer, target_region="h3", max_len=128)
        model = AttemptModel([[5, 2], [4, 5, 2], [5, 5],
                              [4, 5, 2], [4, 5, 2], [4, 5, 2]])
        payload = run_generative_eval(
            model, tokenizer, dataset, device="cpu", num_examples=-1, num_samples=3,
            max_new_tokens=4, skip_perplexity=True, output_path=None)
        self.assertEqual(payload["decoding"]["protocol"], iglm.SAMPLING_METADATA)
        self.assertEqual(payload["n_selected"], 2)
        self.assertEqual(payload["n_generated"], 1)
        validity = payload["generation_validity"]
        self.assertEqual((validity["attempts"], validity["complete_nonempty"],
                          validity["invalid"], validity["incomplete"]), (6, 1, 4, 1))
        self.assertEqual(len(payload["generation_attempts"]), 2)
        self.assertEqual([len(record["draws"]) for record in payload["generation_attempts"]], [3, 3])
        self.assertEqual(payload["records"][0]["conditioning"],
                         {"chain_token": "[HEAVY]", "species_token": "[HUMAN]"})

    def test_backend_accepts_legacy_and_current_tokenizer_constructor_signatures(self):
        for keyword in ("vocab_file", "vocab"):
            calls = []

            class Backend:
                def get_vocab(self):
                    return dict(iglm.TOKEN_IDS)

                def save_pretrained(self, directory):
                    raise AssertionError("This constructor check never saves its backend")

            if keyword == "vocab_file":
                def constructor(vocab_file=None, do_lower_case=True):
                    calls.append((vocab_file, do_lower_case))
                    return Backend()
            else:
                def constructor(vocab=None, do_lower_case=True):
                    calls.append((vocab, do_lower_case))
                    return Backend()

            with self.subTest(keyword=keyword), patch("transformers.BertTokenizerFast", constructor):
                iglm.load_tokenizer(self.source)
                self.assertEqual(calls, [(str(self.source / "vocab.txt"), False)])


class IgLMCommandAndLayoutReviewTests(unittest.TestCase):
    def test_original_checkpoint_can_explicitly_select_the_ar_command(self):
        args = generate.parse_args(["--model_dir", "/tmp/native-iglm", "--model_type", "IgLM",
                                    "--method", "ar", "--target_loop", "h3"])
        with patch("generate.read_run_config", return_value=None), \
                patch("cdrseqgen.methods.ar.gen_cli.main", return_value=sentinel.result) as command:
            self.assertIs(generate.main(args), sentinel.result)
        self.assertIs(command.call_args.args[0], args)

    def test_explicit_method_cannot_reinterpret_a_saved_checkpoint(self):
        args = generate.parse_args(["--model_dir", "/tmp/saved-checkpoint", "--method", "ar"])
        with patch("generate.read_run_config", return_value={"method": "mlm"}):
            with self.assertRaisesRegex(ValueError, "Checkpoint method"):
                generate.main(args)

    def test_nonstring_public_fields_raise_configuration_errors(self):
        for value in (None, 3, ["target_region"], {"species_token": "human"}):
            with self.subTest(value=value):
                layout = replace(CDRSEQGEN_CAUSAL, public_fields=(value,))
                with self.assertRaisesRegex(ValueError, "public_fields"):
                    validate_layout(layout, causal_adapter())


if __name__ == "__main__":
    unittest.main()
