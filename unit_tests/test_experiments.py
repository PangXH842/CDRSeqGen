"""Tests merged from: test_experiment_specs.py, test_queue_tools.py, test_weekend_queue.py, test_iglm_experiments.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.plan import build_plan
from experiments.spec import spec_trials, SUITES
import copy
import csv
import sys
from benchmark.arms_recipe import ROOT, batch_arms, build
import os
import subprocess
import contextlib
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from benchmark._common import stable_example_seed
from cdrseqgen.datasets.dataset_ar import span_layout_metadata
from experiments.spec import spec_trials
from cdrseqgen.methods.ar.backbones import resolve
from cdrseqgen.methods.ar.models import iglm
from cdrseqgen.methods.ar.sampling import sample_loops


# ====================================================================================================
# from test_experiment_specs.py
# Specs must merge predictably and feed the existing plan identity/resume machinery.
# ====================================================================================================

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
        with patch('experiments.plan.source_identity',return_value={'source_sha256':'fixed'}):
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
        with patch('experiments.plan.source_identity', return_value={'source_sha256': 'fixed'}):
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
        with patch('experiments.plan.source_identity', return_value={'source_sha256': 'fixed'}):
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


# ====================================================================================================
# from test_queue_tools.py
# The weekend queue's recipe generator and spec writer, on synthetic batches and the committed templates.
# ====================================================================================================

sys.path.insert(0, str(ROOT / 'experiments/tools'))
import queue_specs
FIELDS = ['seed', 'status', 'run_dir', 'trial_id']

def make_batch(directory, trials, rows):
    """trials: [(suffix, method, seed, model_type)], rows: [(trial index, status, run dir name)] in file order."""
    directory.mkdir(parents=True)
    plan = dict(trials=[dict(suffix=s, trial_id=f'id{i}', config=dict(method=m, seed=seed, model_type=t))
                        for i, (s, m, seed, t) in enumerate(trials)])
    (directory / 'plan.json').write_text(json.dumps(plan))
    with (directory / 'runs.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter='\t')
        writer.writeheader()
        for index, status, run in rows:
            writer.writerow(dict(seed=trials[index][2], status=status, run_dir=str(ROOT / 'results/runs' / run), trial_id=f'id{index}'))
    return directory


class ArmsRecipeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_completed_scratch_trials_become_labelled_arms(self):
        batch = make_batch(self.root / 'b', [('mlm_scratch_no_partner_s44', 'mlm', 44, 'scratch'),
                                             ('ar_scratch_no_partner_L_s42', 'ar', 42, 'scratch')],
                           [(0, 'completed', 'b/mlm44'), (1, 'completed', 'b/ar')])
        self.assertEqual(batch_arms(batch), [
            dict(label='mlm_s44', method='mlm', training_seed=44, checkpoint='results/runs/b/mlm44'),
            dict(label='ar_L_s42', method='ar', training_seed=42, checkpoint='results/runs/b/ar')])

    def test_a_retry_row_replaces_the_failed_one(self):
        batch = make_batch(self.root / 'b', [('ar_scratch_no_partner_s42', 'ar', 42, 'scratch')],
                           [(0, 'failed', 'b/ar'), (0, 'completed', 'b/ar_2')])
        self.assertEqual(batch_arms(batch)[0]['checkpoint'], 'results/runs/b/ar_2')

    def test_unfinished_and_pretrained_trials_stop_the_script_unless_waived(self):
        batch = make_batch(self.root / 'b', [('a_scratch_no_partner_s1', 'ar', 1, 'scratch'), ('c_esm2', 'mlm', 1, 'esm2'),
                                             ('d_scratch_no_partner_s2', 'ar', 2, 'scratch')],
                           [(0, 'completed', 'b/a'), (1, 'completed', 'b/c'), (2, 'running', 'b/d')])
        with self.assertRaisesRegex(ValueError, 'pretrained mlm model'):    # pretrained AR/S2S are served now
            batch_arms(batch)
        with self.assertRaisesRegex(ValueError, 'has not completed'):
            batch_arms(batch, skip_pretrained=True)
        self.assertEqual([a['label'] for a in batch_arms(batch, skip_pretrained=True, skip_incomplete=True)], ['a_s1'])

    def test_recipe_keeps_the_reference_protocol_and_rejects_duplicate_labels(self):
        batch = make_batch(self.root / 'b', [('mlm_scratch_no_partner_s44', 'mlm', 44, 'scratch')], [(0, 'completed', 'b/m')])
        recipe = build('queue', [batch])
        reference = json.loads((ROOT / 'benchmark/recipes/oas4_h3_length_free.json').read_text())
        for key in ('manifest', 'smoke_manifest', 'sampling', 'judge', 'endpoint'):
            self.assertEqual(recipe[key], reference[key])
        self.assertNotIn('iglm', recipe)
        self.assertEqual(len(recipe['limits']), len(reference['limits']) + 1)
        with self.assertRaisesRegex(ValueError, 'unique labels'):
            build('queue', [batch, batch])
        with self.assertRaisesRegex(ValueError, 'unique labels'):
            build('queue', [])


class QueueSpecTests(unittest.TestCase):
    def test_twelve_specs_with_the_intended_trials(self):
        specs = queue_specs.specs()
        self.assertEqual(len(specs), 12)
        for method in queue_specs.METHODS:
            self.assertEqual([t['name'] for t in specs[f'seeds_{method}']['trials']],
                             [f'{method}_scratch_no_partner_s{s}' for s in queue_specs.SEEDS])
            for k in (1, 2):
                trials = specs[f'repeat_{method}_r{k}']['trials']
                self.assertEqual((len(trials), trials[0]['overrides']['seed']), (1, 42))
            capacity = specs[f'capacity_{method}']['trials']
            self.assertEqual(len(capacity), 6)
            self.assertEqual(len({t['name'] for t in capacity}), 6)

    def test_trials_change_only_the_seed_or_the_size(self):
        for stem, spec in queue_specs.specs().items():
            method = stem.split('_')[1]  # seeds_<m>, repeat_<m>_r<k>, capacity_<m>
            _, source = queue_specs.template(method)
            for trial in spec['trials']:
                changed = {k for k in set(trial['overrides']) | set(source['overrides'])
                           if trial['overrides'].get(k) != source['overrides'].get(k)}
                self.assertLessEqual(changed, {'seed', 'data_seed', 'hidden_size', 'num_layers',
                                               'num_encoder_layers', 'num_decoder_layers'}, (stem, trial['name']))
                self.assertEqual(trial.get('base_config'), source.get('base_config'))

    def test_s2s_depth_is_set_on_both_sides_and_other_methods_leave_it_alone(self):
        specs = queue_specs.specs()
        xl = next(t for t in specs['capacity_s2s']['trials'] if '_XL_' in t['name'])['overrides']
        self.assertEqual((xl['num_encoder_layers'], xl['num_decoder_layers'], xl['num_layers']), (4, 4, 8))
        for method in ('mlm', 'ar'):
            self.assertTrue(all('num_encoder_layers' not in t['overrides'] for t in specs[f'capacity_{method}']['trials']))

    def test_committed_specs_match_the_generator(self):
        for stem, spec in queue_specs.specs().items():
            self.assertEqual(json.loads((queue_specs.OUT / f'{stem}.json').read_text()), spec, stem)


# ====================================================================================================
# from test_weekend_queue.py
# Control flow of scripts/weekend_queue.sh with its heavy commands stubbed: order, skipping, isolation, retries.
# ====================================================================================================

WEEKEND_QUEUE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = WEEKEND_QUEUE_ROOT / 'scripts/weekend_queue.sh'
ENVIRONMENTS = [Path('/home/pang/miniconda3/envs/cdrprj/bin/python'), Path('/mnt/hdd/pang/envs/igcraft/bin/python')]

class QueueFlowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.queue = Path(temp.name) / 'queue'

    def run_queue(self, command='start', **env):
        # A distinct tag, so that even a bug could not touch the real weekend batch directories.
        environment = dict(os.environ, QUEUE_STUB='1', QUEUE_TAG='queue-test', QUEUE_DIR=str(self.queue), RETRY_SLEEP='0', **env)
        result = subprocess.run(['bash', str(SCRIPT), command], cwd=WEEKEND_QUEUE_ROOT, env=environment, capture_output=True, text=True, timeout=120)
        self.assert_nothing_real(result)
        return result

    def assert_nothing_real(self, result):
        """Stub mode must never reach a heavy command: no refusal, no batch directory, no generated recipe."""
        self.assertNotIn('REFUSED', result.stdout + result.stderr)
        for pattern in ('results/runs/queue-test_*', 'results/benchmark/queue-test_*'):
            self.assertEqual(list(WEEKEND_QUEUE_ROOT.glob(pattern)), [], pattern)
        if self.queue.exists():
            for log in (self.queue / 'logs').glob('*.log'):
                self.assertNotIn('REFUSED', log.read_text())

    def lines(self, prefix):
        return [line.split(prefix, 1)[1].strip() for line in (self.queue / 'queue.log').read_text().splitlines() if prefix in line]

    def test_a_failure_injected_into_stub_mode_does_not_fall_through_to_real_work(self):
        # The earlier fail-open hook ran real training exactly here: a matching STUB_FAIL fell through.
        result = self.run_queue(STUB_FAIL='train_batch seeds_ar')
        self.assertIn('STUB train_batch seeds_ar', (self.queue / 'logs/train_seeds_seeds_ar.log').read_text())
        self.assertNotIn('Batch resume', (self.queue / 'logs/train_seeds_seeds_ar.log').read_text())
        self.assertEqual(list(WEEKEND_QUEUE_ROOT.glob('results/runs/queue-test_*')), [])

    def test_every_stage_runs_once_in_order_and_a_rerun_skips_them(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        started = self.lines('start  ')
        self.assertEqual(started[:2], ['blind_all', 'igcraft_t1'])
        self.assertEqual(len(started), 17)
        for earlier, later in [('igcraft_t1', 'train_seeds_seeds_mlm'), ('train_seeds_seeds_s2s', 'bench_seeds'),
                               ('bench_seeds', 'train_rep1_repeat_ar_r1'), ('train_rep2_repeat_s2s_r2', 'bench_repeats'),
                               ('bench_repeats', 'train_capacity_capacity_ar'), ('train_capacity_capacity_s2s', 'bench_capacity')]:
            self.assertLess(started.index(earlier), started.index(later), (earlier, later))
        self.assertEqual(len(list((self.queue / 'done').iterdir())), 17)
        again = self.run_queue()
        self.assertEqual(again.returncode, 0)
        self.assertEqual(len(self.lines('skip   ')), 17)
        self.assertEqual(self.lines('start  ')[17:], [])

    def test_a_failed_training_stage_is_retried_isolated_and_rerun_alone(self):
        result = self.run_queue(STUB_FAIL='train_batch seeds_ar')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sorted(p.name for p in (self.queue / 'failed').iterdir()), ['train_seeds_seeds_ar'])
        self.assertIn('attempt 2/2 failed', (self.queue / 'logs/train_seeds_seeds_ar.log').read_text())
        for later in ('bench_seeds', 'train_capacity_capacity_s2s', 'bench_capacity'):
            self.assertTrue((self.queue / 'done' / later).exists(), later)
        fixed = self.run_queue()
        self.assertEqual(fixed.returncode, 0, fixed.stdout)
        self.assertEqual(self.lines('start  ')[17:], ['train_seeds_seeds_ar'])
        self.assertFalse(list((self.queue / 'failed').iterdir()))

    def test_status_lists_progress_and_a_missing_queue_is_reported(self):
        self.assertNotEqual(self.run_queue('status').returncode, 0)
        self.run_queue()
        status = self.run_queue('status')
        self.assertIn('bench_capacity', status.stdout)
        self.assertIn('failed:  \n', status.stdout)

    def test_a_second_queue_cannot_start_while_one_holds_the_lock(self):
        self.queue.mkdir(parents=True)
        (self.queue / 'logs').mkdir()
        holder = subprocess.Popen(['bash', '-c', f'exec 9> {self.queue}/lock; flock -n 9; sleep 20'])
        self.addCleanup(holder.kill)
        import time
        time.sleep(1)
        result = self.run_queue()
        self.assertEqual(result.returncode, 1)
        self.assertIn('already running', result.stdout)


# ====================================================================================================
# from test_iglm_experiments.py
# The IgLM experiment specs, the random-init control script and the replay harness.
#
# Specs are pinned the way the other shipped recipes are: against the completed AR no-partner control, so a
# difference between an IgLM arm and a scratch arm is only what the protocol says it is. Scripts run on tiny
# local GPT-2 weights; nothing here needs the released checkpoints.
# ====================================================================================================

IGLM_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IGLM_EXPERIMENTS_ROOT / "experiments/tools"))
import iglm_replay
import make_iglm_randinit
SPECS = IGLM_EXPERIMENTS_ROOT / "configs/experiments"
# What an IgLM arm may differ in from the completed scratch AR control: the backbone, the scratch
# dimensions that do not apply to a pretrained model, and the PLM optimizer settings of the ESM-2 controls.
ALLOWED = {"model_type", "hidden_size", "num_layers", "num_attention_heads", "learning_rate",
           "warmup_ratio", "weight_decay", "do_train"}

def trials(stem):
    with contextlib.chdir(IGLM_EXPERIMENTS_ROOT):
        return spec_trials(f"configs/experiments/20261102_{stem}.json", {})[0]


def control():
    with contextlib.chdir(IGLM_EXPERIMENTS_ROOT):
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
