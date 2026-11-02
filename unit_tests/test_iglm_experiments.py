"""The IgLM experiment specs, the random-init control script and the replay harness.

Specs are pinned the way the other shipped recipes are: against the completed AR no-partner control, so a
difference between an IgLM arm and a scratch arm is only what the protocol says it is. Scripts run on tiny
local GPT-2 weights; nothing here needs the released checkpoints.
"""
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from benchmark._common import stable_example_seed
from cdrseqgen.datasets.dataset_ar import span_layout_metadata
from cdrseqgen.experiments import spec_trials
from cdrseqgen.methods.ar.backbones import resolve
from cdrseqgen.methods.ar.models import iglm
from cdrseqgen.methods.ar.sampling import sample_loops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import iglm_replay
import make_iglm_randinit

SPECS = ROOT / "configs/experiments"
# What an IgLM arm may differ in from the completed scratch AR control: the backbone, the scratch
# dimensions that do not apply to a pretrained model, and the PLM optimizer settings of the ESM-2 controls.
ALLOWED = {"model_type", "hidden_size", "num_layers", "num_attention_heads", "learning_rate",
           "warmup_ratio", "weight_decay", "do_train"}


def trials(stem):
    with contextlib.chdir(ROOT):
        return spec_trials(f"configs/experiments/20261102_{stem}.json", {})[0]


def control():
    with contextlib.chdir(ROOT):
        found = spec_trials("configs/experiments/20261006_no_partner_controls.json", {})[0]
    return next(t for t in found if t["suffix"] == "ar_scratch_no_partner_s42")["config"]


def differing(first, second):
    return {key for key in first.keys() | second.keys() if first.get(key, "<absent>") != second.get(key, "<absent>")}


def released(config):
    return Path(config["model_type"].removeprefix("iglm:")).name


class SpecTests(unittest.TestCase):
    def test_trial_matrices(self):
        zero = trials("iglm_zeroshot")
        self.assertEqual([(released(t["config"]), t["config"]["do_train"]) for t in zero],
                         [("IgLM-S", False), ("IgLM", False)])
        small = trials("iglm_s_finetune")
        self.assertEqual([t["config"]["learning_rate"] for t in small], [1e-5, 3e-5, 1e-4, 3e-4])
        self.assertEqual({released(t["config"]) for t in small}, {"IgLM-S"})
        large = trials("iglm_l_finetune")
        self.assertEqual([t["config"]["learning_rate"] for t in large], [1e-5, 3e-5])
        self.assertEqual({released(t["config"]) for t in large}, {"IgLM"})
        controls = trials("iglm_controls")
        self.assertEqual([t["config"]["learning_rate"] for t in controls[:3]], [3e-5, 1e-4, 3e-4])
        self.assertEqual({t["config"]["model_type"] for t in controls[:3]}, {"iglm:results/controls/iglm_s_random_init"})
        self.assertEqual(controls[3]["config"]["model_type"], "scratch")

    def test_every_iglm_arm_differs_from_the_scratch_control_only_as_declared(self):
        reference = control()
        for stem in ("iglm_zeroshot", "iglm_s_finetune", "iglm_l_finetune", "iglm_controls"):
            for trial in trials(stem):
                if trial["config"]["model_type"] == "scratch":
                    continue
                with self.subTest(trial=trial["suffix"]):
                    self.assertEqual(differing(trial["config"], reference) - ALLOWED, set())
                    self.assertNotIn("hidden_size", trial["config"])      # the checkpoint fixes its own shape

    def test_the_size_matched_scratch_control_changes_only_its_size(self):
        scratch = trials("iglm_controls")[3]["config"]
        self.assertEqual(differing(scratch, control()),
                         {"hidden_size", "num_layers", "num_attention_heads", "intermediate_size"})
        self.assertEqual((scratch["hidden_size"], scratch["num_layers"], scratch["num_attention_heads"],
                          scratch["intermediate_size"]), (192, 3, 6, 768))     # IgLM-S's shape

    def test_shared_protocol_matches_the_completed_controls(self):
        for stem in ("iglm_zeroshot", "iglm_s_finetune", "iglm_l_finetune", "iglm_controls"):
            for trial in trials(stem):
                config = trial["config"]
                with self.subTest(trial=trial["suffix"]):
                    self.assertFalse(config["include_partner_chain"])
                    self.assertEqual((config["seed"], config["data_seed"]), (42, 42))
                    self.assertEqual((config["gen_num_examples"], config["gen_num_samples"], config["gen_seed"]),
                                     (150, 20, 42))
                    self.assertEqual(config["gen_judge_partner_context"], "with")      # Table 8's judge policy
                    self.assertEqual(config["eval_split"], "validation")
                    self.assertEqual(config["gen_max_new_tokens"], 97)
                    self.assertTrue(config["run_generative_eval"])
                    self.assertTrue(iglm.MIN_MAX_LEN <= config["max_len"] <= 512)

    def test_zero_shot_trials_are_the_only_ones_that_skip_training(self):
        for stem in ("iglm_s_finetune", "iglm_l_finetune", "iglm_controls"):
            self.assertTrue(all(t["config"]["do_train"] for t in trials(stem)))


class RandomInitTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "release"
        self.source.mkdir()
        GPT2LMHeadModel(GPT2Config(vocab_size=33, n_positions=512, n_ctx=512, n_embd=16, n_layer=1,
                                   n_head=4)).save_pretrained(self.source)

    def test_control_is_a_loadable_untrained_iglm_source_with_provenance(self):
        record = make_iglm_randinit.create(self.source, self.root / "control", seed=3)
        backbone = resolve("iglm:" + str(self.root / "control"))
        self.assertEqual(backbone.id, "iglm")
        self.assertEqual(resolve(str(self.root / "control")).id, "iglm")     # recognised by its marker, without the prefix
        model = backbone.load_training_model("iglm:" + str(self.root / "control"), vocab_size=33, max_len=384)
        self.assertEqual(sum(p.numel() for p in model.parameters()), record["parameters"])
        released_weights = GPT2LMHeadModel.from_pretrained(self.source).state_dict()["transformer.wte.weight"]
        self.assertFalse(torch.equal(model.state_dict()["transformer.wte.weight"], released_weights))
        self.assertEqual(json.loads((self.root / "control/RANDOM_INIT.json").read_text())["seed"], 3)
        self.assertEqual((self.root / "control/vocab.txt").read_text().split(), list(iglm.NATIVE_VOCAB))

    def test_initialisation_is_seeded_and_never_overwrites_silently(self):
        first = make_iglm_randinit.create(self.source, self.root / "a", seed=1)
        again = make_iglm_randinit.create(self.source, self.root / "b", seed=1)
        other = make_iglm_randinit.create(self.source, self.root / "c", seed=2)
        self.assertEqual(first["weights_sha256"], again["weights_sha256"])
        self.assertNotEqual(first["weights_sha256"], other["weights_sha256"])
        with self.assertRaises(FileExistsError):
            make_iglm_randinit.create(self.source, self.root / "a", seed=1)
        make_iglm_randinit.create(self.source, self.root / "a", seed=2, force=True)

    def test_a_foreign_architecture_is_refused(self):
        foreign = self.root / "foreign"
        foreign.mkdir()
        GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=64, n_embd=16, n_layer=1, n_head=4)).save_pretrained(foreign)
        with self.assertRaises(ValueError):
            make_iglm_randinit.create(foreign, self.root / "out", seed=0)

    def test_force_cannot_destroy_the_source_or_an_existing_control_on_bad_input(self):
        output = self.root / 'control'
        first = make_iglm_randinit.create(self.source, output, seed=1)
        for destination in (self.source, self.root, self.source / 'nested'):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, 'must not overlap'):
                make_iglm_randinit.create(self.source, destination, seed=1, force=True)
        foreign = self.root / 'foreign'
        foreign.mkdir()
        (foreign / 'config.json').write_text('{"model_type":"gpt2","vocab_size":50,"n_positions":64}')
        with self.assertRaises(ValueError):
            make_iglm_randinit.create(foreign, output, seed=1, force=True)
        self.assertEqual(make_iglm_randinit.sha256(output / 'model.safetensors'), first['weights_sha256'])
        self.assertTrue((self.source / 'config.json').is_file())


class ReplayTests(unittest.TestCase):
    MANIFEST = dict(task=dict(target_region="h3"), records=[
        dict(example_id=f"e{i}", pt1="ACDEFGHIK" * 3, pt2="LMNPQRS" * 2, chain_token="[HEAVY]",
             species_token="[HUMAN]") for i in range(3)])

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name) / "tiny"
        cls.directory.mkdir()
        (cls.directory / "vocab.txt").write_text("\n".join(iglm.NATIVE_VOCAB) + "\n")
        torch.manual_seed(0)
        config = GPT2Config(vocab_size=33, n_positions=512, n_ctx=512, n_embd=16, n_layer=1, n_head=4)
        config.cdrseqgen_backbone = "iglm"
        GPT2LMHeadModel(config).save_pretrained(cls.directory)
        config.span_layout = span_layout_metadata(iglm.load_tokenizer(str(cls.directory)))
        config.save_pretrained(cls.directory)
        (cls.directory / "span_layout.json").write_text(json.dumps(config.span_layout))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def stored(self, attempts=4):
        """A stored batch made by the same sampling procedure; the random tiny model gives valid and failed draws."""
        backbone = resolve("iglm:" + str(self.directory))
        tokenizer = backbone.build_tokenizer("iglm:" + str(self.directory), checkpoint_path=str(self.directory))
        model = backbone.load_pretrained_model(str(self.directory)).eval()
        sampling = dict(length_policy="natural", num_samples=attempts, temperature=1.0, top_p=0.9, base_seed=42,
                        max_new_tokens=97)
        records = []
        for record in self.MANIFEST["records"]:
            seed = stable_example_seed(42, record["example_id"])
            iglm_replay.seed_everything(seed)
            framework = dict(pt1=record["pt1"], pt2=record["pt2"], target_length=0, max_len=256, partner_chain=None,
                             antigen=None, target_region="h3", chain_token="[HEAVY]", species_token="[HUMAN]")
            from cdrseqgen.datasets.dataset_ar import build_causal_prompt
            prompt, _ = build_causal_prompt(tokenizer, framework)
            draws = sample_loops(model, tokenizer, prompt, device=torch.device("cpu"), num_samples=attempts,
                                 temperature=1.0, top_k=0, top_p=0.9, max_new_tokens=97, generator=None)
            outcome = iglm_replay.ours_attempts(draws)
            records.append(dict(
                example_id=record["example_id"], seed=seed,
                raw_generations=[dict(attempt=n, loop=v, valid_generation=True)
                                 for n, (kind, v) in outcome.items() if kind == "ok"],
                failures=[dict(attempt=n, code=v) for n, (kind, v) in outcome.items() if kind == "fail"]))
        return dict(sampling=sampling, model=dict(name="IgLM", checkpoint_sha256=None), records=records)

    def replay(self, generations, **options):
        return iglm_replay.replay(generations, self.MANIFEST, self.directory, torch.device("cpu"),
                                  options.pop("max_len", 256), **options)

    def test_a_batch_made_by_the_same_procedure_replays_exactly(self):
        generations = self.stored()
        self.assertTrue(any(r["raw_generations"] for r in generations["records"]))
        report = self.replay(generations)
        self.assertEqual((report["examples_identical"], report["examples"]), (3, 3))
        self.assertEqual(report["attempts_identical"], 12)
        self.assertFalse(report["weights_match"])          # no stored hash to match

    def test_one_changed_draw_is_found_at_its_attempt(self):
        generations = self.stored()
        draw = next(r for r in generations["records"] if r["raw_generations"])
        first = draw["raw_generations"][0]
        first["loop"] = first["loop"] + "A"
        report = self.replay(generations)
        self.assertEqual([(row["example_id"], row["first_difference"]) for row in report["different"]],
                         [(draw["example_id"], first["attempt"])])
        self.assertEqual(report["examples_identical"], 2)

    def test_a_seed_that_is_not_the_derived_one_and_a_truncating_budget_are_refused(self):
        generations = self.stored()
        generations["records"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "derived seed"):
            self.replay(generations)
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.replay(self.stored(), max_len=110)

    def test_comparison_rules(self):
        stored = {1: ("ok", "AC"), 2: ("fail", "x"), 3: ("ok", "DE")}
        self.assertEqual(iglm_replay.compare(stored, {1: ("ok", "AC"), 2: ("fail", "y"), 3: ("ok", "DE")}, 3), (None, 3))
        self.assertEqual(iglm_replay.compare(stored, {1: ("ok", "AC"), 2: ("ok", "F"), 3: ("ok", "DE")}, 3), (2, 2))
        self.assertEqual(iglm_replay.compare(stored, {1: ("ok", "AC")}, 3), (2, 1))
        self.assertEqual(iglm_replay.compare({}, {}, 2), (1, 0))
        self.assertEqual(iglm_replay.compare({1: ('fail', 'x')}, {}, 1), (1, 0))
        draws = [dict(loop="", complete=True, valid=True), dict(loop="AC", complete=False, valid=False),
                 dict(loop="AC", complete=True, valid=True)]
        self.assertEqual([kind for kind, _ in iglm_replay.ours_attempts(draws).values()], ["fail", "fail", "ok"])


if __name__ == "__main__":
    unittest.main()
