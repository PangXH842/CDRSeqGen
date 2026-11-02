"""IgLM edges beyond test_iglm_backbone: native releases, species names and generation records.

Tiny local GPT-2 weights only. The release fixture copies the two original releases' config
fingerprint (name, dimensions, conditioning flags) onto random weights, so recognition is exercised
without the real checkpoints.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from cdrseqgen.datasets.dataset_ar import span_layout_metadata
from cdrseqgen.methods.ar.backbones import record_training_layout, resolve
from cdrseqgen.methods.ar.generation import generate_case, prepare_generation
from cdrseqgen.methods.ar.models import iglm


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


if __name__ == "__main__":
    unittest.main()
