"""CPU contracts for the unified benchmark; no model loading or generation."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from benchmark.adapters.cdrseqgen import EOSSampler, build_sampler, resolve_decoding, run_example, validate_checkpoint_contract
from benchmark.adapters.iglm import IgLMSampler, InvalidGeneration, extract_infilled_loop
from benchmark.run import generation_commands, validate_frozen_plan
from cdrseqgen.datasets.dataset_ar import build_causal_prompt, causal_adapter
from cdrseqgen.datasets.dataset_s2s import build_encoder_input, s2s_adapter
from unit_tests.test_cdrseqgen_adapter import FakeSampler, manifest_record


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


if __name__ == '__main__':
    unittest.main()
