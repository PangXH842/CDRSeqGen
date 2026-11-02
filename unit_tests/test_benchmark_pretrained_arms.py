"""Pretrained AR/S2S checkpoints as local benchmark arms, with scratch prompts and plans unchanged.

Tiny local GPT-2 weights and hand-made checkpoint directories only; no model is run.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel

from benchmark import arms_recipe, run
from benchmark.adapters.cdrseqgen import EOSSampler, pretrained_input_files, run_example
from cdrseqgen.datasets.dataset_ar import build_causal_prompt, causal_adapter
from cdrseqgen.datasets.dataset_s2s import build_encoder_input, s2s_adapter
from cdrseqgen.methods.ar.models import iglm
from unit_tests.test_cdrseqgen_adapter import manifest_record

CONDITIONED = dict(manifest_record(), chain_token="[HEAVY]", species_token="[HUMAN]")


def sampler(tokenizer, builder, **run_config):
    result = EOSSampler.__new__(EOSSampler)
    result.tokenizer, result.build_prompt = tokenizer, builder
    result.run_config, result.max_new_tokens = {"max_len": 256, **run_config}, 97
    return result


class ConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        directory = Path(cls.temp.name) / "tiny-iglm"
        directory.mkdir()
        (directory / "vocab.txt").write_text("\n".join(iglm.NATIVE_VOCAB) + "\n")
        config = GPT2Config(vocab_size=33, n_positions=512, n_ctx=512, n_embd=16, n_layer=1, n_head=4)
        config.cdrseqgen_backbone = "iglm"
        GPT2LMHeadModel(config).save_pretrained(directory)
        cls.tokenizer = iglm.load_tokenizer(str(directory))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def prompt(self, record, **run_config):
        arm = sampler(self.tokenizer, build_causal_prompt, **run_config)
        arm.prepare(record)
        return arm

    def test_manifest_tokens_and_checkpoint_region_reach_an_iglm_prompt(self):
        arm = self.prompt(dict(CONDITIONED, species_token="[MOUSE]"), target_region="h3")
        encode = self.tokenizer.encode
        self.assertEqual(arm.prompt_ids, [31, 27, *encode("QQ"), 4, *encode("WW"), 3])
        self.assertEqual(arm.prompt_info["conditioning"],
                         {"chain_token": "[HEAVY]", "species_token": "[MOUSE]"})

    def test_the_hidden_native_loop_and_length_never_change_the_prompt(self):
        arm = self.prompt(CONDITIONED, target_region="h3")
        before = list(arm.prompt_ids)
        arm.prepare(dict(CONDITIONED, native="W" * 40, native_length=40))
        self.assertEqual(before, arm.prompt_ids)

    def test_the_region_alone_is_enough_and_a_conflicting_chain_is_rejected(self):
        arm = self.prompt(manifest_record(), target_region="l3")
        self.assertEqual(arm.prompt_ids[:2], [32, 26])            # [LIGHT][HUMAN] by default
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.prompt(CONDITIONED, target_region="l3")

    def test_no_region_and_no_chain_fails_before_any_decoding(self):
        with self.assertRaisesRegex(ValueError, "target_region or an explicit chain_token"):
            self.prompt(manifest_record())

    def test_default_layouts_ignore_the_conditioning_keys(self):
        for tokenizer, builder in ((causal_adapter(), build_causal_prompt),
                                   (s2s_adapter(), build_encoder_input)):
            plain = sampler(tokenizer, builder)
            plain.prepare(manifest_record())
            conditioned = sampler(tokenizer, builder, target_region="h3")
            conditioned.prepare(CONDITIONED)
            self.assertEqual(plain.prompt_ids, conditioned.prompt_ids)
            self.assertEqual(plain.prompt_info, conditioned.prompt_info)
            self.assertNotIn("conditioning", conditioned.prompt_info)


class ArmRuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name)
        for name in ("model.safetensors", "config.json", "run_config.json", "tokenizer_config.json",
                     "tokenizer.json", "span_layout.json"):
            (self.checkpoint / name).write_text("{}")

    def paths(self, method, model_type, **arm):
        arm = dict(label="x", method=method, training_seed=42, **arm)
        config = dict(method=method, model_type=model_type)
        names = run.arm_paths(arm, config, self.checkpoint, self.checkpoint / "run_config.json")
        return [path.name for path in names]

    def test_scratch_arms_pin_exactly_what_they_always_did(self):
        base = ["model.safetensors", "config.json", "run_config.json"]
        for method in ("ar", "s2s"):
            self.assertEqual(self.paths(method, "scratch"), base)   # saved vocabulary files are not pinned
        (self.checkpoint / "length_config.json").write_text(json.dumps({"seed": 42}))
        (self.checkpoint / "length_head.safetensors").write_text("{}")
        self.assertEqual(self.paths("mlm", "scratch"),
                         base + ["length_config.json", "length_head.safetensors"])
        (self.checkpoint / "length_config.json").write_text(json.dumps({"seed": 43}))
        with self.assertRaises(AssertionError):
            self.paths("mlm", "scratch")

    def test_pretrained_ar_and_s2s_arms_also_pin_their_saved_vocabulary_and_layout(self):
        for method, model_type in (("ar", "iglm:/somewhere/IgLM-S"), ("ar", "nferruz/ProtGPT2"),
                                   ("s2s", "facebook/esm2_t6_8M_UR50D")):
            with self.subTest(method=method, model_type=model_type):
                self.assertEqual(self.paths(method, model_type),
                                 ["model.safetensors", "config.json", "run_config.json",
                                  "tokenizer_config.json", "tokenizer.json", "span_layout.json"])

    def test_pretrained_arms_require_saved_tokenizers_and_freeze_generation_settings(self):
        (self.checkpoint / 'generation_config.json').write_text('{"repetition_penalty": 1.1}')
        self.assertIn('generation_config.json', self.paths('ar', 'iglm:/somewhere/IgLM-S'))
        (self.checkpoint / 'tokenizer_config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'require a saved tokenizer_config.json'):
            self.paths('ar', 'iglm:/somewhere/IgLM-S')
        (self.checkpoint / 'tokenizer_config.json').write_text('{}')
        (self.checkpoint / 'tokenizer.json').unlink()
        with self.assertRaisesRegex(ValueError, 'vocabulary beside their weights'):
            pretrained_input_files(self.checkpoint)

    def test_malformed_terminated_draws_keep_their_decoder_failure(self):
        class FailedSampler:
            track = 'length_free'
            def prepare(self, record):
                pass
            def draw_many(self, **options):
                return [dict(loop='', complete=False, valid=False, terminated=True,
                             failure='invalid_token_layout', raw_token_ids=[31, 26, 4, 3, 3, 2]),
                        dict(loop='', complete=False, valid=False, terminated=False,
                             failure='missing_terminator', raw_token_ids=[31, 26, 4, 3, 5])]
        result = run_example(CONDITIONED, FailedSampler(), num_samples=2, decoding='causal',
                             temperature=1., top_k=0, top_p=1., length_temperature=1., base_seed=42)
        self.assertEqual([item['code'] for item in result['failures']],
                         ['invalid_token_layout', 'missing_terminator'])
        self.assertEqual([item['terminated'] for item in result['raw_generations']], [True, False])
        self.assertEqual(result['invalid_attempts'], 2)
        self.assertEqual(result['raw_generations'][0]['raw_token_ids'], [31, 26, 4, 3, 3, 2])

    def test_pretrained_mlm_and_unknown_backbones_are_refused_at_plan_time(self):
        with self.assertRaisesRegex(ValueError, "pretrained mlm checkpoints are not served"):
            self.paths("mlm", "facebook/esm2_t6_8M_UR50D")
        with self.assertRaisesRegex(ValueError, "No ar backbone registered"):
            self.paths("ar", "not-a-backbone")

    def batch(self, trials, name="batch"):
        directory = Path(self.temp.name) / name
        directory.mkdir()
        (directory / "plan.json").write_text(json.dumps({"trials": [
            dict(trial_id=name, suffix=suffix, config=dict(method=method, model_type=model_type, seed=42))
            for name, suffix, method, model_type in trials]}))
        with (directory / "runs.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["trial_id", "status", "run_dir"])
            for name, *_ in trials:     # run directories need not exist; only their place in the repository
                writer.writerow([name, "completed", str(run.ROOT / "results/runs/_unit_test" / name)])
        return directory

    def test_recipe_arms_accept_pretrained_ar_and_s2s_but_not_mlm(self):
        accepted = [("a", "ar_s42_scratch_no_partner", "ar", "scratch"),
                    ("b", "ar_iglm_s42", "ar", "iglm:/somewhere/IgLM-S"),
                    ("c", "s2s_esm_s42", "s2s", "facebook/esm2_t6_8M_UR50D")]
        arms = arms_recipe.batch_arms(self.batch(accepted))
        self.assertEqual([arm["label"] for arm in arms], ["ar_s42", "ar_iglm_s42", "s2s_esm_s42"])
        self.assertEqual([arm["method"] for arm in arms], ["ar", "ar", "s2s"])
        mlm = accepted + [("d", "mlm_esm_s42", "mlm", "facebook/esm2_t6_8M_UR50D")]
        with self.assertRaisesRegex(ValueError, "pretrained mlm model"):
            arms_recipe.batch_arms(self.batch(mlm, "with_mlm"))

    def test_skip_pretrained_still_skips_every_non_scratch_trial(self):
        trials = [("a", "ar_s42_scratch_no_partner", "ar", "scratch"),
                  ("b", "ar_iglm_s42", "ar", "iglm:/somewhere/IgLM-S"),
                  ("d", "mlm_esm_s42", "mlm", "facebook/esm2_t6_8M_UR50D")]
        arms = arms_recipe.batch_arms(self.batch(trials), skip_pretrained=True)
        self.assertEqual([arm["label"] for arm in arms], ["ar_s42"])

    def test_an_untrained_zero_shot_trial_is_refused_with_a_clear_message(self):
        directory = self.batch([("a", "iglm_s_zeroshot", "ar", "iglm:/somewhere/IgLM-S")], "zeroshot")
        plan = json.loads((directory / "plan.json").read_text())
        plan["trials"][0]["config"]["do_train"] = False
        (directory / "plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "not trained .do_train false."):
            arms_recipe.batch_arms(directory)
        self.assertEqual(arms_recipe.batch_arms(directory, skip_pretrained=True), [])

    def test_the_two_modules_agree_on_which_methods_serve_pretrained_backbones(self):
        self.assertEqual(run.EOS_METHODS, arms_recipe.EOS_METHODS)


if __name__ == "__main__":
    unittest.main()
