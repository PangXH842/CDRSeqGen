"""Tests merged from: test_iglm_backbone.py, test_iglm_integration.py, test_iglm_review.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

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
import shutil
from cdrseqgen.datasets.dataset_ar import span_layout_metadata
from cdrseqgen.methods.ar.backbones import record_training_layout, resolve
from cdrseqgen.methods.ar.generation import generate_case, prepare_generation
from dataclasses import replace
from unittest.mock import patch, sentinel
from safetensors.torch import load_file, save_file
from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel
import generate
from cdrseqgen.datasets.dataset_ar import (
    CDRSEQGEN_CAUSAL,
    CausalInfillingDataset,
    build_causal_prompt,
    causal_adapter,
    span_layout_metadata,
    validate_layout,
)
from cdrseqgen.methods.ar.evaluation import run_generative_eval


# ====================================================================================================
# from test_iglm_backbone.py
# Native IgLM integration, using tiny local GPT-2 weights and controlled draws only.
# ====================================================================================================

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


# ====================================================================================================
# from test_iglm_integration.py
# IgLM edges beyond test_iglm_backbone: native releases, species names and generation records.
#
# Tiny local GPT-2 weights only. The release fixture copies the two original releases' config
# fingerprint (name, dimensions, conditioning flags) onto random weights, so recognition is exercised
# without the real checkpoints.
# ====================================================================================================

class RandomResidueModel:
    """Complete every prompt with one residue drawn from the ambient RNG, then [CLS]."""

    config = SimpleNamespace(n_positions=512)

    def eval(self):
        return self

    def generate(self, input_ids=None, **kwargs):
        residue = int(torch.randint(5, 25, (1,)).item())
        suffix = torch.tensor([[residue, 2]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


class RecordingModel:
    """Return a fixed completion and keep the prompt each generate() call received."""

    def __init__(self, completion):
        self.completion = completion
        self.prompts = []
        self.config = SimpleNamespace(n_positions=512)

    def eval(self):
        return self

    def generate(self, input_ids=None, **kwargs):
        self.prompts.append(input_ids[0].tolist())
        suffix = torch.tensor([self.completion], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


def generation_args(**overrides):
    args = dict(decoding=None, min_length=None, max_length=None, length_temperature=1.0,
                max_new_tokens=8, max_len=256, partner_chain=None, antigen=None, num_samples=2,
                temperature=1.0, top_k=0, top_p=0.9, skip_perplexity=True, seed=1, target_loop=None)
    args.update(overrides)
    return SimpleNamespace(**args)


class ConditioningTests(unittest.TestCase):
    def species(self, given):
        return iglm.resolve_conditioning({"target_region": "h3", "species_token": given})["species_token"]

    def test_native_tokens_and_iglm_species_names_map_to_native_species_tokens(self):
        for given, expected in (("[HUMAN]", "[HUMAN]"), ("human", "[HUMAN]"), ("mouse", "[MOUSE]"),
                                ("[mouse]", "[MOUSE]"), ("mouse_C57BL/6", "[MOUSE]"),
                                ("HIS-mouse", "[MOUSE]"), ("rat_SD", "[RAT]"), ("rhesus", "[RHESUS]"),
                                ("Camel", "[CAMEL]"), ("rabbit", "[RABBIT]")):
            with self.subTest(given=given):
                self.assertEqual(self.species(given), expected)

    def test_unknown_species_is_rejected_rather_than_treated_as_human(self):
        for given in ("zebrafish", "", "[HEAVY]"):
            with self.subTest(given=given), self.assertRaises(ValueError):
                self.species(given)

    def test_species_defaults_to_human_and_a_chain_is_required(self):
        self.assertEqual(iglm.resolve_conditioning({"target_region": "l1"}),
                         {"chain_token": "[LIGHT]", "species_token": "[HUMAN]"})
        self.assertEqual(iglm.resolve_conditioning({"chain_token": "heavy", "species_token": None}),
                         {"chain_token": "[HEAVY]", "species_token": "[HUMAN]"})
        for options in ({}, {"species_token": "human"}, {"target_region": "x9"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                iglm.resolve_conditioning(options)


class NativeReleaseTests(unittest.TestCase):
    RELEASE_NAME = "trained_models/IgLM_XS_chain_species_95/checkpoint-180000/"

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.release = cls.root / "trained_models" / "IgLM-S"
        cls.release.mkdir(parents=True)
        (cls.release.parent / "vocab.txt").write_text("\n".join(iglm.NATIVE_VOCAB) + "\n")
        config = GPT2Config(vocab_size=33, n_positions=512, n_ctx=512, n_embd=192, n_layer=3,
                            n_head=6, n_inner=768, bos_token_id=50256, eos_token_id=50256)
        GPT2LMHeadModel(config).save_pretrained(cls.release)
        cls.edit_config(cls.release, _name_or_path=cls.RELEASE_NAME, condition_on_chain=True,
                        condition_on_species=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def edit_config(directory, **changes):
        path = Path(directory) / "config.json"
        config = json.loads(path.read_text())
        config.update(changes)
        path.write_text(json.dumps(config))

    def variant(self, name, **config_changes):
        """A copy of the release in its own trained_models/ folder, with the vocabulary beside it."""
        directory = self.root / name / "trained_models" / "IgLM-S"
        shutil.copytree(self.release, directory)
        shutil.copy(self.release.parent / "vocab.txt", directory.parent / "vocab.txt")
        self.edit_config(directory, **config_changes)
        return directory

    def test_original_release_is_recognized_and_imported_without_layout_metadata(self):
        self.assertTrue(iglm.is_native_checkpoint(self.release))
        self.assertTrue(iglm.matches(str(self.release)))
        backbone = resolve(str(self.release))
        self.assertEqual(backbone.id, "iglm")
        tokenizer = backbone.build_tokenizer(str(self.release), checkpoint_path=self.release)
        self.assertEqual(span_layout_metadata(tokenizer)["id"], "iglm_infill")
        model = backbone.load_pretrained_model(self.release)
        self.assertEqual((model.config.eos_token_id, model.config.pad_token_id), (2, 0))
        self.assertIsNone(model.config.bos_token_id)

    def test_anything_that_is_not_exactly_a_release_is_not_native(self):
        changes = {"saved run": dict(cdrseqgen_backbone="iglm"),
                   "recorded layout": dict(span_layout={"id": "iglm_infill"}),
                   "other name": dict(_name_or_path="trained_models/other/checkpoint-1/"),
                   "other width": dict(n_embd=96),
                   "other context": dict(n_positions=256),
                   "no chain conditioning": dict(condition_on_chain=False)}
        for label, change in changes.items():
            with self.subTest(label=label):
                self.assertFalse(iglm.is_native_checkpoint(self.variant(label.replace(" ", "_"), **change)))

    def test_a_cdrseqgen_run_directory_cannot_claim_native_status(self):
        directory = self.variant("run_dir")
        (directory / "run_config.json").write_text("{}")
        self.assertFalse(iglm.is_native_checkpoint(directory))
        backbone = resolve("iglm:" + str(directory))
        with self.assertRaisesRegex(ValueError, "span layout metadata"):
            backbone.build_tokenizer("iglm:" + str(directory), checkpoint_path=directory)

    def test_vocabulary_must_be_the_unchanged_native_one(self):
        directory = self.variant("bad_vocab")
        lines = list(iglm.NATIVE_VOCAB)
        lines[5], lines[6] = lines[6], lines[5]
        (directory.parent / "vocab.txt").write_text("\n".join(lines) + "\n")
        self.assertFalse(iglm.is_native_checkpoint(directory))
        with self.assertRaisesRegex(ValueError, "native 33-token"):
            iglm.load_tokenizer(str(directory))

    def test_a_saved_tokenizer_with_extra_tokens_is_rejected(self):
        directory = self.variant("saved_tokenizer")
        backend = iglm.load_tokenizer(str(directory)).backend_tokenizer
        backend.save_pretrained(directory)
        self.assertEqual(iglm.load_tokenizer(str(directory)).vocab_size, 33)
        backend.add_tokens(["[NEW]"])
        backend.save_pretrained(directory)
        with self.assertRaisesRegex(ValueError, "native vocabulary ids"):
            iglm.load_tokenizer(str(directory))

    def test_training_max_len_must_hold_the_answer_reserve_and_fit_the_checkpoint(self):
        model_args = SimpleNamespace(model_type="iglm:" + str(self.release))
        backbone = resolve(model_args.model_type)

        def data(max_len):
            return SimpleNamespace(include_partner_chain=False, include_antigen=False, max_len=max_len)

        for accepted in (iglm.MIN_MAX_LEN, 256, 512):
            backbone.validate_training_args(model_args, data(accepted))
        for rejected in (iglm.MIN_MAX_LEN - 1, 100, 513):
            with self.subTest(max_len=rejected), self.assertRaises(ValueError):
                backbone.validate_training_args(model_args, data(rejected))
        self.assertEqual(iglm.MIN_MAX_LEN, 101)

    def test_training_records_the_sampling_protocol_beside_the_layout(self):
        tokenizer = iglm.load_tokenizer(str(self.release))
        model = iglm.load_model(str(self.release))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run_config.json").write_text("{}")
            record_training_layout(model, tokenizer, directory)
            saved = json.loads((Path(directory) / "run_config.json").read_text())
        for recorded in (saved, model.config.to_dict()):
            self.assertEqual(recorded["span_layout"], span_layout_metadata(tokenizer))
            self.assertEqual(recorded["causal_sampling"], tokenizer.causal_sampling_metadata)
        self.assertEqual(saved["causal_sampling"]["bad_token_ids"], [0, 1, *range(25, 33)])
        self.assertFalse(saved["causal_sampling"]["forced_eos"])

    def test_the_default_layout_records_no_sampling_protocol(self):
        scratch = resolve("scratch")
        tokenizer = scratch.build_tokenizer("scratch")
        model = scratch.load_training_model("scratch", vocab_size=tokenizer.vocab_size, hidden_size=16,
                                            num_layers=1, num_attention_heads=2, max_len=128)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run_config.json").write_text("{}")
            record_training_layout(model, tokenizer, directory)
            saved = json.loads((Path(directory) / "run_config.json").read_text())
        self.assertNotIn("causal_sampling", saved)
        self.assertFalse(hasattr(model.config, "causal_sampling"))

    def test_a_missing_vocabulary_is_reported_not_guessed(self):
        directory = self.variant("no_vocab")
        (directory.parent / "vocab.txt").unlink()
        with self.assertRaises(ValueError):
            iglm.load_tokenizer(str(directory))

    def test_model_loading_rejects_foreign_checkpoints(self):
        for label, change in {"vocabulary": dict(vocab_size=25), "family": dict(model_type="bert"),
                              "capacity": dict(n_positions=0)}.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                iglm.load_model(str(self.variant(f"foreign_{label}", **change)))

    def test_decoding_beyond_the_checkpoint_context_is_rejected(self):
        model = RecordingModel([5, 2])
        tokenizer = iglm.load_tokenizer(str(self.release))
        prompt = [31, 26, 5, 4, 6, 3]
        self.assertEqual(len(iglm.sample_loops(model, tokenizer, prompt, device="cpu",
                                               max_new_tokens=97)), 1)
        model.config.n_positions = 100          # 6 prompt tokens + 97 decode steps no longer fit
        with self.assertRaisesRegex(ValueError, "capacity"):
            iglm.sample_loops(model, tokenizer, prompt, device="cpu", max_new_tokens=97)


class GenerationRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name) / "tiny-iglm"
        cls.directory.mkdir()
        (cls.directory / "vocab.txt").write_text("\n".join(iglm.NATIVE_VOCAB) + "\n")
        config = GPT2Config(vocab_size=33, n_positions=512, n_ctx=512, n_embd=16, n_layer=1, n_head=4)
        config.cdrseqgen_backbone = "iglm"
        GPT2LMHeadModel(config).save_pretrained(cls.directory)
        cls.tokenizer = iglm.load_tokenizer(str(cls.directory))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def record(self, case, *, config=None, model=None, **arg_changes):
        args = generation_args(**arg_changes)
        prepare_generation(args, [case], {"target_region": "h3"} if config is None else config)
        model = model or RecordingModel([5, 6, 2])
        result = generate_case(model, self.tokenizer, "cpu", args, torch.Generator().manual_seed(1),
                               case, None, lambda *a, **k: None)
        return result, model

    def test_the_explicit_generator_decides_the_draws(self):
        prompt = [31, 26, 5, 4, 6, 3]

        def draws(seed):
            records = iglm.sample_loops(RandomResidueModel(), self.tokenizer, prompt, device="cpu",
                                        num_samples=8, generator=torch.Generator().manual_seed(seed))
            return [record["loop"] for record in records]

        self.assertEqual(draws(1), draws(1))
        self.assertNotEqual(draws(1), draws(2))

    def test_the_run_region_and_the_case_species_reach_the_prompt_and_the_record(self):
        result, model = self.record(dict(pt1="AC", pt2="FG", species="mouse_C57BL/6"))
        self.assertEqual(model.prompts[0], [31, 27, 5, 6, 4, 9, 10, 3])   # [HEAVY][MOUSE] AC [MASK] FG [SEP]
        self.assertEqual(result["conditioning"], {"chain_token": "[HEAVY]", "species_token": "[MOUSE]"})
        self.assertEqual([s["seq"] for s in result["samples"]], ["AC", "AC"])

    def test_target_loop_beats_the_checkpoint_region_and_a_case_beats_both(self):
        args = generation_args(target_loop="l3")
        prepare_generation(args, [], {"target_region": "h3"})
        self.assertEqual(args.target_region, "l3")
        _, model = self.record(dict(pt1="AC", pt2="FG"), target_loop="l3")
        self.assertEqual(model.prompts[0][:2], [32, 26])
        result, model = self.record(dict(pt1="AC", pt2="FG", target_region="h2"), target_loop="l3")
        self.assertEqual(model.prompts[0][:2], [31, 26])
        self.assertEqual(result["conditioning"]["chain_token"], "[HEAVY]")

    def test_a_run_with_no_region_anywhere_fails_instead_of_guessing_a_chain(self):
        with self.assertRaisesRegex(ValueError, "target_region"):
            self.record(dict(pt1="AC", pt2="FG"), config={})

    def test_generation_conditioning_is_absent_for_the_default_layout(self):
        scratch = resolve("scratch")
        tokenizer = scratch.build_tokenizer("scratch")
        model = scratch.load_training_model("scratch", vocab_size=tokenizer.vocab_size, hidden_size=16,
                                            num_layers=1, num_attention_heads=2, max_len=128)
        args = generation_args(max_len=128, max_new_tokens=4)
        prepare_generation(args, [], {"target_region": "h3"})
        result = generate_case(model, tokenizer, "cpu", args, torch.Generator().manual_seed(1),
                               dict(pt1="ACD", pt2="FG"), None, lambda *a, **k: None)
        self.assertNotIn("conditioning", result)


# ====================================================================================================
# from test_iglm_review.py
# IgLM review regressions, using temporary vocabularies and tiny CPU checkpoints.
# ====================================================================================================

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
