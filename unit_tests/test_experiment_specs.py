"""Specs must merge predictably and feed the existing plan identity/resume machinery."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cdrseqgen.experiments import build_plan, spec_trials, SUITES


class ExperimentSpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'base.json'
        self.base.write_text(json.dumps(dict(method='s2s', model_type='scratch',
            data_path='original.json', learning_rate=1e-4, seed=42)))
        self.path = self.root / 'spec.json'
        self.spec = dict(name='probe', base_config=str(self.base),
            overrides=dict(learning_rate=3e-5), trials=[dict(name='first'),
                dict(name='second', overrides=dict(learning_rate=1e-5))])

    def write(self):
        self.path.write_text(json.dumps(self.spec))
        return str(self.path)

    def test_merge_and_corpus_override(self):
        trials, name = spec_trials(self.write(), {'DATA_PATH': 'relocated.json'})
        self.assertEqual(name, 'probe')
        self.assertEqual([t['config']['learning_rate'] for t in trials], [3e-5,1e-5])
        self.assertTrue(all(t['config']['data_path']=='relocated.json' for t in trials))
        self.assertNotIn('s2s_igbert_diagnostic', SUITES)
        other = self.root / 'other.json'
        other.write_text(json.dumps(dict(method='ar',model_type='scratch',data_path='x.json')))
        self.spec['trials'][1]['base_config'] = str(other)
        self.assertEqual(spec_trials(self.write(), {})[0][1]['config']['method'],'ar')

    def test_invalid_specs_fail_early(self):
        original = json.loads(json.dumps(self.spec))
        cases = [dict(trials=[]), dict(trials=[dict(name='../bad')]),
                 dict(trials=[dict(name='same'),dict(name='same')]),
                 dict(trials=[dict(name='a'),dict(name='b')]),
                 dict(overrides=[]), dict(unknown=True), dict(name=None),
                 dict(trials=[dict(name='a',overrides=[])]),
                 dict(trials=[dict(name='a',typo=True)])]
        for change in cases:
            with self.subTest(change=change):
                self.spec = {**original, **change}
                with self.assertRaises(ValueError): spec_trials(self.write(),{})
        self.spec = original
        for key in ('SEED','FREEZE_ENCODER','ENCODER_INIT'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                spec_trials(self.write(),{key:'42'})

    def test_plan_identity_and_shared_execution_format(self):
        data=self.root/'data.json'
        data.write_text(json.dumps([dict(split='train'),dict(split='validation')]))
        self.spec['overrides'].update(data_path=str(data),eval_split='validation')
        with patch('cdrseqgen.experiments.source_identity',return_value={'source_sha256':'fixed'}):
            plan, devices = build_plan(['0,1','spec',self.write()],{})
            self.assertEqual(plan['mode'],'suite')
            self.assertEqual(devices,['0','1'])
            self.spec['trials'][1]['overrides']['learning_rate']=2e-5
            changed,_=build_plan(['2','spec',self.write()],{})
        self.assertEqual(plan['trials'][0]['trial_id'],changed['trials'][0]['trial_id'])
        self.assertNotEqual(plan['trials'][1]['trial_id'],changed['trials'][1]['trial_id'])
        with self.assertRaises(ValueError):build_plan(['0','spec'],{})

    def test_identity_ignores_where_the_corpus_lives(self):
        """Same bytes at two paths = same experiment. data_path is provenance, not identity."""
        payload = json.dumps([dict(split='train'), dict(split='validation')])
        first, second = self.root/'a'/'corpus.json', self.root/'b'/'corpus.json'
        for path in (first, second):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        seen, plans = [], []
        with patch('cdrseqgen.experiments.source_identity', return_value={'source_sha256': 'fixed'}):
            for path in (first, second):
                self.spec['overrides'].update(data_path=str(path), eval_split='validation')
                plan, _ = build_plan(['0', 'spec', self.write()], {})
                seen.append([t['trial_id'] for t in plan['trials']])
                plans.append(plan)
        self.assertEqual(seen[0], seen[1], "moving the corpus must not change trial identity")
        self.assertEqual([t['setting_id'] for t in plans[0]['trials']],
                         [t['setting_id'] for t in plans[1]['trials']])
        # Location is still recorded, just not fingerprinted.
        self.assertEqual({t['config']['data_path'] for t in plans[1]['trials']},
                         {str(second.resolve())})
        # A genuinely different corpus must still separate.
        (self.root/'c').mkdir(exist_ok=True)
        third = self.root/'c'/'corpus.json'
        third.write_text(json.dumps([dict(split='train'), dict(split='validation'),
                                     dict(split='validation')]))
        with patch('cdrseqgen.experiments.source_identity', return_value={'source_sha256': 'fixed'}):
            self.spec['overrides'].update(data_path=str(third), eval_split='validation')
            other, _ = build_plan(['0', 'spec', self.write()], {})
        self.assertNotEqual(seen[0], [t['trial_id'] for t in other['trials']])


class FollowupRecipeTests(unittest.TestCase):
    def test_mlm_stages_complete_the_same_context_matrix_without_native_length(self):
        from transformers import HfArgumentParser, TrainingArguments
        from cdrseqgen.methods.common.arguments import parse_config_dict, validate_common_args
        from cdrseqgen.methods.mlm.arguments import ModelArguments, DataArguments, GenerationArguments
        from cdrseqgen.methods.mlm.length import resolve_generation_length_policy
        from cdrseqgen.methods.mlm.losses import loss_config_from_args

        controls, _ = spec_trials('configs/experiments/20261007_mlm_no_partner_controls.json', {})
        screen, _ = spec_trials('configs/experiments/20261007_mlm_dropout_screen.json', {})
        self.assertEqual(len(controls), 2)
        self.assertEqual(len(screen), 4)
        trials = controls + screen
        self.assertEqual(
            {(t['config']['include_partner_chain'], t['config']['context_dropout'],
              t['config']['seed']) for t in trials},
            {(partner, dropout, seed) for partner, dropout in
             ((False, 0.0), (True, 0.0), (True, 0.5)) for seed in (42, 43)})
        common = None
        evaluations = 0
        for trial in trials:
            config = dict(trial['config'])
            with self.subTest(trial=trial['suffix']):
                self.assertEqual(config['data_seed'], config['seed'])
                self.assertEqual(config['gen_eval_context_ablation'], config['include_partner_chain'])
                self.assertEqual(config['eval_split'], 'validation')
                self.assertEqual(config['max_len'], 384)
                self.assertFalse(config['include_antigen'])
                self.assertTrue(config['train_length_predictor'])
                self.assertEqual(config['gen_judge_partner_context'], 'with')
                self.assertEqual((config['gen_num_examples'], config['gen_num_samples'],
                                  config['gen_seed']), (150, 20, 42))
                parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments,
                                          TrainingArguments))
                model, data, gen, training = parse_config_dict(
                    parser, {**config, 'use_cpu': True, 'report_to': 'none'})
                validate_common_args(model, data, gen, expected_method='mlm')
                self.assertEqual(loss_config_from_args(model, training).loss_type, 'cross_entropy')
                self.assertEqual(data.masking_policy, 'full_span')
                policy = resolve_generation_length_policy(
                    gen.gen_length_mode, min_length=gen.gen_min_length,
                    max_length=gen.gen_max_length)
                self.assertEqual(policy.mode, 'inferred')
                self.assertTrue(policy.requires_predictor)
                self.assertEqual(training.get_warmup_steps(1000), 0)
                evaluations += 1 + int(config['gen_eval_context_ablation'])
                for key in ('seed', 'data_seed', 'include_partner_chain', 'context_dropout',
                            'gen_eval_context_ablation'):
                    config.pop(key)
                if common is None:
                    common = config
                self.assertEqual(config, common)
        self.assertEqual(evaluations, 10)

    def test_no_partner_controls_cover_each_model_and_seed_with_fixed_judge_context(self):
        from transformers import HfArgumentParser, TrainingArguments
        from cdrseqgen.methods.common.arguments import parse_config_dict, validate_common_args
        from cdrseqgen.methods.registry import resolve
        controls, _ = spec_trials('configs/experiments/20261006_no_partner_controls.json', {})
        self.assertEqual(len(controls), 6)
        self.assertEqual(len({(t['config']['method'], t['config']['model_type'], t['config']['seed'])
                             for t in controls}), 6)
        for trial in controls:
            config = trial['config']
            with self.subTest(trial=trial['suffix']):
                self.assertFalse(config['include_partner_chain'])
                self.assertEqual(config['context_dropout'], 0)
                self.assertEqual(config['gen_judge_partner_context'], 'with')
                self.assertFalse(config['gen_eval_context_ablation'])
                self.assertEqual(config['num_train_epochs'], 20)
                self.assertEqual(config['gen_num_examples'], 150)
                self.assertEqual(config['gen_num_samples'], 20)
                self.assertEqual(config['gen_seed'], 42)
                self.assertEqual(config['data_seed'], config['seed'])
                self.assertEqual(config['eval_split'], 'validation')
                self.assertEqual(config['learning_rate'], 3e-5 if 'esm2' in config['model_type'] else 1e-4)
                module = resolve(config['method']).module('arguments')
                parser = HfArgumentParser((module.ModelArguments, module.DataArguments,
                    module.GenerationArguments, TrainingArguments))
                model, data, gen, training = parse_config_dict(parser, {**config, 'use_cpu': True, 'report_to': 'none'})
                validate_common_args(model, data, gen, expected_method=config['method'])
                self.assertEqual(training.get_warmup_steps(1000), 60 if 'esm2' in config['model_type'] else 0)

    def test_confirmation_pairs_change_only_dropout_and_use_new_shared_seeds(self):
        from transformers import HfArgumentParser, TrainingArguments
        from cdrseqgen.methods.common.arguments import parse_config_dict
        from cdrseqgen.methods.s2s.arguments import ModelArguments, DataArguments, GenerationArguments, validate_args
        trials, _ = spec_trials('configs/experiments/20261006_esm2_dropout_confirmation.json', {})
        self.assertEqual(len(trials), 4)
        for index, seed in ((0, 44), (2, 45)):
            first, second = (dict(t['config']) for t in trials[index:index+2])
            self.assertEqual(first.pop('context_dropout'), 0)
            self.assertEqual(second.pop('context_dropout'), .5)
            self.assertEqual(first, second)
            self.assertEqual(first['seed'], seed)
            self.assertEqual(first['data_seed'], seed)
            self.assertTrue(first['gen_eval_context_ablation'])
            self.assertTrue(first['include_partner_chain'])
            self.assertEqual(first['gen_judge_partner_context'], 'with')
            self.assertEqual(first['learning_rate'], 3e-5)
        for trial in trials:
            parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
            model, data, gen, training = parse_config_dict(parser,
                {**trial['config'], 'use_cpu': True, 'report_to': 'none'})
            validate_args(model, data, gen)
            self.assertEqual(training.get_warmup_steps(1000), 60)

if __name__=='__main__':unittest.main()
