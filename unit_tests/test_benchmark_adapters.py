"""Tests merged from: test_cdrseqgen_adapter.py, test_cdrseqgen_sampler.py, test_benchmark_methods.py, test_benchmark_pretrained_arms.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import json
import tempfile
import unittest
from pathlib import Path
from benchmark.adapters.cdrseqgen import (
    assert_resume_compatible,
    load_run_config,
    run_example,
    stable_example_seed,
    validate_checkpoint_contract,
)
from types import SimpleNamespace
import torch
from benchmark.adapters.cdrseqgen import CDRSeqGenSampler
from unittest.mock import Mock, patch
from benchmark.adapters.cdrseqgen import (
    EOSSampler,
    build_sampler,
    resolve_decoding,
    run_example,
    validate_checkpoint_contract,
)
from benchmark.adapters.iglm import IgLMSampler, InvalidGeneration, extract_infilled_loop
from benchmark.run import generation_commands, validate_frozen_plan
from cdrseqgen.datasets.dataset_ar import build_causal_prompt, causal_adapter
from cdrseqgen.datasets.dataset_s2s import build_encoder_input, s2s_adapter
import csv
from transformers import GPT2Config, GPT2LMHeadModel
from benchmark import arms_recipe, run
from benchmark.adapters.cdrseqgen import EOSSampler, pretrained_input_files, run_example
from cdrseqgen.methods.ar.models import iglm


# ====================================================================================================
# from test_cdrseqgen_adapter.py
# ====================================================================================================

def manifest_record():
    return {
        "example_id": "oas:7:example",
        "parent_sequence": "QQAAAWW",
        "pt1": "QQ",
        "native": "AAA",
        "pt2": "WW",
        "native_length": 3,
        "infill_range": [2, 5],
    }


class FakeSampler:
    def __init__(self, samples):
        self.samples = samples
        self.prepared = None
        self.received_seed = None

    def prepare(self, record):
        self.prepared = record["example_id"]

    def draw_many(self, **kwargs):
        self.received_seed = kwargs["seed"]
        return list(self.samples)


class CDRSeqGenAdapterTests(unittest.TestCase):
    def test_load_run_config_accepts_legacy_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            legacy = checkpoint / "run_meta.json"
            legacy.write_text(json.dumps({"target_region": "h3"}))

            config, source = load_run_config(checkpoint)

        self.assertEqual(config["target_region"], "h3")
        self.assertEqual(source, legacy)

    def test_load_run_config_prefers_current_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            legacy = checkpoint / "run_meta.json"
            current = checkpoint / "run_config.json"
            legacy.write_text(json.dumps({"target_region": "l2"}))
            current.write_text(json.dumps({"target_region": "h3"}))

            config, source = load_run_config(checkpoint)

        self.assertEqual(config["target_region"], "h3")
        self.assertEqual(source, current)

    def test_load_run_config_refuses_a_checkpoint_without_one_and_names_the_accepted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                load_run_config(Path(tmp))

        self.assertIn("run_config.json", str(caught.exception))
        self.assertIn("run_meta.json", str(caught.exception))

    def test_load_run_config_does_not_hide_a_corrupt_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "run_config.json").write_text("{not json")
            with self.assertRaises(json.JSONDecodeError):
                load_run_config(Path(tmp))

    def test_records_samples_and_preserves_duplicates(self):
        sampler = FakeSampler(["CCC", "CCC", "DDD"])
        result = run_example(
            manifest_record(), sampler,
            num_samples=3, decoding="iterative", temperature=1.0,
            top_k=0, top_p=0.9, base_seed=42,
        )
        self.assertEqual(result["samples"], ["CCC", "CCC", "DDD"])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["length_matches"], 3)
        self.assertEqual(
            result["length_policy"],
            {
                "mode": "fixed",
                "source": "native_mask_count",
                "minimum": 3,
                "maximum": 3,
                "selected_length": 3,
            },
        )

        self.assertTrue(result["complete"])
        self.assertEqual(sampler.prepared, manifest_record()["example_id"])
        self.assertEqual(
            sampler.received_seed,
            stable_example_seed(42, manifest_record()["example_id"]),
        )

    def test_invalid_output_is_audited_not_scored(self):
        sampler = FakeSampler(["CC", "C?C", "CCC"])
        result = run_example(
            manifest_record(), sampler,
            num_samples=3, decoding="independent", temperature=1.0,
            top_k=0, top_p=1.0, base_seed=42,
        )
        self.assertEqual(result["samples"], ["CCC"])
        self.assertEqual(result["length_mismatches"], 1)
        self.assertEqual(result["valid_attempts"], 2)
        self.assertEqual(result["invalid_attempts"], 1)
        self.assertFalse(result["complete"])
        self.assertEqual({f["code"] for f in result["failures"]}, {"length_mismatch", "invalid_amino_acid"})

    def test_length_free_track_retains_all_valid_lengths(self):
        sampler = FakeSampler(["CC", "CCC", "CCCC"])
        sampler.track = "length_free"
        sampler.length_metadata = {"mode": "inferred", "selected_lengths": [2, 3, 4]}
        result = run_example(
            manifest_record(), sampler,
            num_samples=3, decoding="independent", temperature=1.0,
            top_k=0, top_p=1.0, base_seed=42,
        )
        self.assertEqual(result["samples"], ["CC", "CCC", "CCCC"])
        self.assertEqual(result["length_matches"], 1)
        self.assertEqual(result["length_mismatches"], 2)
        self.assertFalse(any(failure["code"] == "length_mismatch" for failure in result["failures"]))
        self.assertEqual(result["length_policy"]["mode"], "inferred")

    def test_checkpoint_contract_rejects_context_or_dataset_mismatch(self):
        manifest = {
            "task": {
                "target_region": "h3",
                "track": "fixed_native_length",
                "partner_chain_context": False,
                "antigen_context": False,
            },
            "source": {"processed_file": "processed.json"},
        }
        config = {
            "target_region": "h3",
            "include_partner_chain": False,
            "include_antigen": False,
            "data_path": "data/processed.json",
        }
        validate_checkpoint_contract(config, manifest)
        with self.assertRaisesRegex(ValueError, "include_partner_chain"):
            validate_checkpoint_contract({**config, "include_partner_chain": True}, manifest)
        with self.assertRaisesRegex(ValueError, "data file"):
            validate_checkpoint_contract({**config, "data_path": "other.json"}, manifest)

    def test_a_failing_example_is_recorded_rather_than_aborting_the_run(self):
        class ExplodingSampler(FakeSampler):
            def prepare(self, record):
                raise ValueError("assembled length 700 exceeds checkpoint max_len 512")

        result = run_example(
            manifest_record(), ExplodingSampler([]),
            num_samples=3, decoding="iterative", temperature=1.0,
            top_k=0, top_p=0.9, base_seed=42,
        )
        self.assertTrue(result["generation_failed"])
        self.assertEqual(result["samples"], [])
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(result["failures"][0]["code"], "generation_exception")
        self.assertEqual(result["length_policy"]["source"], "generation_failed")

    def test_a_failed_example_does_not_inherit_the_previous_length_policy(self):
        """length_metadata is sampler state; a failure must not report the last example's."""
        class StatefulSampler(FakeSampler):
            def __init__(self, samples):
                super().__init__(samples)
                self.calls = 0

            def prepare(self, record):
                self.length_metadata = None
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("CUDA out of memory")
                self.length_metadata = {"mode": "inferred", "selected_length": 3}

        sampler = StatefulSampler(["CCC"])
        kwargs = dict(num_samples=1, decoding="iterative", temperature=1.0,
                      top_k=0, top_p=0.9, base_seed=42)
        first = run_example(manifest_record(), sampler, **kwargs)
        second = run_example(manifest_record(), sampler, **kwargs)

        self.assertEqual(first["length_policy"]["mode"], "inferred")
        self.assertEqual(second["length_policy"]["mode"], "unavailable")

    def test_resume_rejects_changed_checkpoint_or_sampling(self):
        expected = {
            "schema_version": "v1", "adapter": "cdrseqgen", "benchmark_id": "b",
            "manifest": {"sha256": "m"}, "model": {"checkpoint_sha256": "c"},
            "sampling": {"decoding": "iterative"},
        }
        assert_resume_compatible(dict(expected), expected)
        with self.assertRaisesRegex(ValueError, "model"):
            assert_resume_compatible({**expected, "model": {"checkpoint_sha256": "x"}}, expected)


# ====================================================================================================
# from test_cdrseqgen_sampler.py
# ====================================================================================================

def _sampler(track):
    """A sampler with the checkpoint-loading __init__ bypassed, so no weights are needed."""
    sampler = CDRSeqGenSampler.__new__(CDRSeqGenSampler)
    sampler.torch = torch
    sampler.device = torch.device("cpu")
    sampler.track = track
    sampler.model = object()
    sampler.tokenizer = SimpleNamespace(decode=lambda ids: "AC")
    sampler.max_len = 128
    sampler.length_predictor = None
    sampler.record = {"pt1": "QQ", "pt2": "WW"}
    sampler.length_policy = SimpleNamespace(mode="fixed", length=2)
    return sampler


class CDRSeqGenSamplerTests(unittest.TestCase):
    """Both benchmark tracks go through sample_length_conditioned_spans; a fixed native length is
    just a policy whose length is known up front, not a separate sampling path."""

    def _capture(self, track):
        sampler = _sampler(track)
        captured = {}

        def fake_shared(model, tokenizer, device, generator, **kwargs):
            captured["model"] = model
            captured.update(kwargs)
            return [SimpleNamespace(token_ids=[5, 6])], {"prompts_by_length": {"2": {}}}

        sampler.sample_length_conditioned_spans = fake_shared
        samples = sampler.draw_many(
            num_samples=1,
            decoding="independent",
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            length_temperature=1.0,
            seed=42,
        )
        return sampler, captured, samples

    def test_fixed_track_uses_the_shared_length_conditioned_path(self):
        sampler, captured, samples = self._capture("fixed_native_length")

        self.assertIs(captured["model"], sampler.model)
        self.assertIs(captured["policy"], sampler.length_policy)
        self.assertEqual(captured["decoding"], "independent")
        self.assertEqual(samples, ["AC"])
        self.assertEqual(sampler.prompt_info, {"2": {}})

    def test_length_free_track_uses_the_same_path(self):
        sampler, captured, samples = self._capture("length_free")

        self.assertIs(captured["model"], sampler.model)
        self.assertIs(captured["length_predictor"], sampler.length_predictor)
        self.assertEqual(samples, ["AC"])

    def test_seeding_is_identical_across_tracks(self):
        """Draw seeding must not depend on the track, or the two tracks are not comparable."""
        _, fixed, _ = self._capture("fixed_native_length")
        _, free, _ = self._capture("length_free")
        self.assertEqual(fixed["num_samples"], free["num_samples"])
        self.assertEqual(fixed["temperature"], free["temperature"])


# ====================================================================================================
# from test_benchmark_methods.py
# CPU contracts for the unified benchmark; no model loading or generation.
# ====================================================================================================

class BenchmarkMethodTests(unittest.TestCase):
    def test_decoders_resolve_from_checkpoint_method_and_saved_settings(self):
        for config, expected in [({}, 'independent'), ({'masking_policy': 'mixed'}, 'independent'),
                                 ({'masking_policy': 'mixed', 'gen_decoding': 'iterative'}, 'iterative'),
                                 ({'method': 'ar'}, 'causal'), ({'method': 's2s'}, 'seq2seq')]:
            self.assertEqual(resolve_decoding(config), expected)
        with self.assertRaisesRegex(ValueError, 'ar decoding'):
            resolve_decoding({'method': 'ar'}, 'independent')

    def test_factory_rejects_native_length_for_eos_before_loading_weights(self):
        with patch('benchmark.adapters.cdrseqgen.EOSSampler') as sampler:
            with self.assertRaisesRegex(ValueError, 'length_free'):
                build_sampler(Path('unused'), {'method': 'ar'}, 'cpu', 'fixed_native_length')
            sampler.assert_not_called()
            build_sampler(Path('unused'), {'method': 's2s'}, 'cpu', 'length_free', 97)
            sampler.assert_called_once_with(Path('unused'), {'method': 's2s'}, 'cpu', 97)

    def test_invalid_eos_cap_is_rejected_before_backbone_or_weights(self):
        for method in ['ar', 's2s']:
            with self.assertRaises(ValueError):
                EOSSampler(Path('/unused/no-checkpoint'), {'method': method}, 'cpu', max_new_tokens=0)

    def test_incomplete_and_empty_eos_draws_are_audited_and_excluded(self):
        sampler = FakeSampler([{'loop': 'AAA', 'complete': False},
                               {'loop': '', 'complete': True}, {'loop': 'CCC', 'complete': True}])
        sampler.track = 'length_free'
        sampler.length_metadata = {'mode': 'inferred', 'source': 'eos_token'}
        result = run_example(manifest_record(), sampler, num_samples=3, decoding='causal',
                             temperature=1., top_k=0, top_p=.9, base_seed=42)
        self.assertEqual(result['samples'], ['CCC'])
        self.assertEqual(result['attempts'], 3)
        self.assertEqual(result['invalid_attempts'], 2)
        self.assertEqual(result['length_matches'], 1)
        self.assertEqual([r['valid_generation'] for r in result['raw_generations']], [False, False, True])
        self.assertEqual({r['code'] for r in result['failures']}, {'incomplete_generation', 'empty_generation'})

    def test_mlm_empty_draw_is_not_a_valid_candidate(self):
        sampler = FakeSampler(['', 'CCC']); sampler.track = 'length_free'
        result = run_example(manifest_record(), sampler, num_samples=2, decoding='independent',
                             temperature=1., top_k=0, top_p=.9, base_seed=42)
        self.assertEqual(result['valid_attempts'], 1)
        self.assertEqual(result['samples'], ['CCC'])

    def test_eos_prompts_do_not_depend_on_native_sequence_or_length(self):
        for tokenizer, builder in [(causal_adapter(), build_causal_prompt), (s2s_adapter(), build_encoder_input)]:
            sampler = EOSSampler.__new__(EOSSampler)
            sampler.tokenizer, sampler.build_prompt = tokenizer, builder
            sampler.run_config, sampler.max_new_tokens = {'max_len': 384}, 97
            record = manifest_record(); sampler.prepare(record)
            prompt = list(sampler.prompt_ids)
            sampler.prepare({**record, 'native': 'W' * 49, 'native_length': 49})
            self.assertEqual(prompt, sampler.prompt_ids)
            self.assertEqual(sampler.length_metadata['source'], 'eos_token')

    def test_eos_sampler_forwards_cap_and_preserves_raw_results(self):
        sampler = EOSSampler.__new__(EOSSampler)
        sampler.torch, sampler.device = torch, torch.device('cpu')
        sampler.model, sampler.tokenizer = object(), object()
        sampler.prompt_ids, sampler.max_new_tokens = [2, 3], 97
        attempts = [{'loop': '', 'complete': True}, {'loop': 'AC', 'complete': False}]
        sampler.sample_loops = Mock(return_value=attempts)
        result = sampler.draw_many(num_samples=2, decoding='causal', temperature=1., top_k=0,
                                   top_p=.9, length_temperature=1., seed=42)
        self.assertIs(result, attempts)
        self.assertEqual(sampler.sample_loops.call_args.kwargs['max_new_tokens'], 97)
        self.assertEqual(sampler.sample_loops.call_args.kwargs['generator'].initial_seed(), 42)

    def test_same_filename_with_different_corpus_hash_is_rejected(self):
        config = {'target_region': 'h3', 'data_path': 'x.json',
                  'provenance': {'dataset': {'declared_processed_file_sha256': 'first'}}}
        manifest = {'task': {'target_region': 'h3', 'track': 'length_free'},
                    'source': {'processed_file': 'x.json', 'processed_file_sha256': 'second'}}
        with self.assertRaisesRegex(ValueError, 'corpus hash'):
            validate_checkpoint_contract(config, manifest)

    def test_iglm_rejects_an_empty_infill(self):
        with self.assertRaises(InvalidGeneration) as error:
            extract_infilled_loop('QQWW', 'QQ', 'WW')
        self.assertEqual(error.exception.code, 'empty_generation')

    def test_iglm_natural_cap_cannot_force_eos(self):
        for policy, forced in [('natural', None), ('filter', None), ('constrain', 9)]:
            sampler = IgLMSampler.__new__(IgLMSampler)
            sampler.torch = torch
            sampler.iglm = SimpleNamespace(model=Mock(), tokenizer=SimpleNamespace(
                pad_token_id=0, cls_token_id=9, decode=lambda ids: 'QQCCCWW'))
            sampler.iglm.model.generate.return_value = torch.tensor([[1, 2, 9]])
            sampler.bad_word_ids, sampler.starting_tokens = [], torch.tensor([[1, 2]])
            sampler.length_policy, sampler.target_total_length, sampler.decode_max_length = policy, 6, 99
            sampler.validate_generated_seq = lambda ids, tok: True
            sampler.iglm_to_infilled = lambda ids, tok: ids
            sampler.draw(temperature=1., top_p=.9)
            kwargs = sampler.iglm.model.generate.call_args.kwargs
            self.assertEqual(kwargs['forced_eos_token_id'], forced)
            self.assertEqual(kwargs['max_length'], 6 if policy == 'constrain' else 99)
            self.assertEqual('min_length' in kwargs, policy == 'constrain')

    def test_iglm_decode_cap_is_relative_to_prompt_without_native_length(self):
        sampler = IgLMSampler.__new__(IgLMSampler)
        sampler.max_length, sampler.max_new_tokens, sampler.length_policy = 256, 97, 'natural'
        sampler._starting_tokens = Mock(return_value=torch.zeros(1, 100))
        sampler.prepare(manifest_record())
        self.assertEqual(sampler.decode_max_length, 197)
        sampler._starting_tokens.return_value = torch.zeros(1, 160)
        with self.assertRaisesRegex(ValueError, 'decode cap'):
            sampler.prepare(manifest_record())

    def test_batch_commands_match_attempt_budgets_and_method_decoders(self):
        plan = dict(cdr_python='cdr-python', iglm_python='iglm-python', manifest='inputs.json',
                    sampling=dict(num_samples=20, temperature=1., top_p=.9, top_k=0,
                                  length_temperature=1., seed=42, max_new_tokens=97),
                    iglm=dict(model_name='IgLM', max_length=256), arms=[
                        dict(label='mlm', method='mlm', checkpoint='mlm', decoding='independent'),
                        dict(label='ar', method='ar', checkpoint='ar', decoding='causal')])
        commands = generation_commands(plan, Path('/tmp/example-only'))
        self.assertEqual(len(commands), 3)
        for _, _, command in commands:
            self.assertEqual(command[command.index('--num-samples') + 1], '20')
        self.assertNotIn('--max-new-tokens', commands[0][2])
        self.assertIn('--max-new-tokens', commands[1][2])
        iglm = commands[-1][2]
        self.assertEqual(iglm[iglm.index('--max-attempts') + 1], '20')
        self.assertEqual(iglm[iglm.index('--max-errors') + 1], '20')

    def test_resume_rejects_changed_source_before_experiment_commands(self):
        with patch('benchmark.run.execution_source_identity', return_value={'sha256': 'new'}):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                validate_frozen_plan({'execution_source': {'sha256': 'old'}})


# ====================================================================================================
# from test_benchmark_pretrained_arms.py
# Pretrained AR/S2S checkpoints as local benchmark arms, with scratch prompts and plans unchanged.
#
# Tiny local GPT-2 weights and hand-made checkpoint directories only; no model is run.
# ====================================================================================================

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
