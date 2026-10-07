"""Exercise both shell entry points and their shared runner in a temporary repository.

Only training and reporting are replaced. The real planner, dataset gate, subprocess scheduling,
artifact checks, journal, and resume contract all execute; nothing needs a GPU or research corpus.
"""
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        shutil.copytree(ROOT / 'configs', self.workspace / 'configs')
        shutil.copytree(ROOT / 'scripts', self.workspace / 'scripts')
        self.corpus = self.workspace / 'alternate corpus.json'
        # Entries carry both chains: a corpus with a null chain is a real case (unpaired OAS),
        # and validate_datasets now refuses partner context a corpus cannot supply.
        self.corpus.write_text(json.dumps(
            [{'split': s, 'heavy': {'full_aa': 'H'}, 'light': {'full_aa': 'L'}}
             for s in ('train', 'validation', 'test')]))
        for p in (self.workspace / 'configs').rglob('*.json'):
            config = json.loads(p.read_text())
            if 'target_region' not in config:
                continue
            config.update(data_path=str(self.corpus), eval_split='validation')
            p.write_text(json.dumps(config))
        (self.workspace / 'train.py').write_text('''
import json, os, sys, time, uuid
from pathlib import Path
c = json.loads(Path(sys.argv[1]).read_text())
method = c.get('method', 'mlm')
name = c['target_region'] + '_' + ('' if method == 'mlm' else method + '_') + 'scratch'
name += '_lc' if c.get('include_partner_chain') else ''
capture = Path('captured')
capture.mkdir(exist_ok=True)
(capture / (str(uuid.uuid4()) + '.json')).write_text(json.dumps({
    'name': name, 'config': c, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
    'batch': os.environ['CDRSEQGEN_RESULTS_ROOT']}))
if os.environ.get('CDR_TEST_SLOW'):
    time.sleep(30)
r = Path(os.environ['CDRSEQGEN_RESULTS_ROOT']) / (name + '_' + str(uuid.uuid4()))
r.mkdir()
print('Workspace initialized at: ' + str(r), flush=True)
if name == os.environ.get('CDR_TEST_FAIL_CONFIG'):
    sys.exit(1)
saved = dict(c)
if isinstance(saved.get('report_to'), str):
    saved['report_to'] = [] if saved['report_to'] == 'none' else [saved['report_to']]
if saved.get('warmup_ratio') is not None:
    saved['warmup_steps'] = saved['warmup_ratio']
(r / 'run_config.json').write_text(json.dumps(saved))
(r / 'epoch_metrics.csv').write_text('epoch,eval_core_accuracy,eval_accuracy\\n1,0.3,0.4\\n')
if name != os.environ.get('CDR_TEST_INCOMPLETE'):
    (r / 'generation_eval.json').write_text(json.dumps({
        'records': [{'eval_index': 0}], 'summary': {'judge_perplexity_mean': 3.}}))
''')
        (self.workspace / 'driver.py').write_text('''
import json, sys
from cdrseqgen import experiments
def report(plan, output):
    (output / 'report_called.json').write_text(json.dumps({
        'mode': plan['mode'], 'rank_metric': plan['rank_metric']}))
experiments.report = report
raise SystemExit(experiments.main(sys.argv[1:]))
''')
        (self.workspace / 'named_driver.py').write_text('''
import json, sys
from cdrseqgen import experiments
from scripts import experiments as commands
def report(plan, output):
    (output / 'report_called.json').write_text(json.dumps({
        'mode': plan['mode'], 'rank_metric': plan['rank_metric']}))
experiments.report = report
raise SystemExit(commands.main(sys.argv[1:]))
''')
        interpreter = self.workspace / 'python'
        interpreter.write_text('#!' + sys.executable + '\n' + '''
import os, sys
args = sys.argv[1:]
if args[:2] == ['-m', 'cdrseqgen.experiments']:
    args = ['driver.py', *args[2:]]
elif args[:1] == ['scripts/experiments.py']:
    args = ['named_driver.py', *args[1:]]
os.execv(sys.executable, [sys.executable, *args])
''')
        interpreter.chmod(0o755)
        self.env = dict(os.environ, PYTHON=str(interpreter), PYTHONPATH=str(ROOT),
                        PYTHONDONTWRITEBYTECODE='1')
        for key in ('EXPERIMENT_DIR', 'SWEEP_DIR', 'DRY_RUN', 'DATA_PATH', 'SEED',
                    'ENCODER_INIT', 'FREEZE_ENCODER',
                    'LRS', 'WARMUPS', 'WEIGHT_DECAYS', 'SEEDS', 'EPOCH_LIST', 'HIDDEN_SIZES',
                    'NUM_LAYERS', 'LOSS_TYPES', 'CORE_WEIGHTS', 'ANCHOR_WEIGHTS', 'GEN_EVAL',
                    'RANK_METRIC', 'SELECT_METRIC', 'GEN_EXAMPLES', 'GEN_SAMPLES',
                    'PATIENCE', 'STOP_THRESHOLD', 'SCHEDULER',
                    'CDR_TEST_FAIL_CONFIG', 'CDR_TEST_INCOMPLETE', 'CDR_TEST_SLOW'):
            self.env.pop(key, None)

    def invoke(self, *args, script='run.sh', **overrides):
        return subprocess.run(['bash', str(self.workspace / 'scripts' / script), *args],
                              cwd=self.workspace.parent, env=dict(self.env, **overrides),
                              capture_output=True, text=True, timeout=30)

    def records(self):
        return [json.loads(p.read_text()) for p in (self.workspace / 'captured').glob('*.json')]

    def plan(self, *args, **overrides):
        result = self.invoke(*args, DRY_RUN='1', **overrides)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout[result.stdout.index('{'):])

    def test_s2s_plm_suites_and_controls_share_the_runner(self):
        plan = self.plan('0', 's2s_plm', 's2s_esm2')
        configs = [trial['config'] for trial in plan['trials']]
        self.assertEqual(len(configs), 6)
        self.assertEqual({c['method'] for c in configs}, {'s2s'})
        self.assertEqual(len({c['model_type'] for c in configs}), 3)
        self.assertTrue(all(c['encoder_init'] == 'pretrained' for c in configs))
        self.assertTrue(all('num_encoder_layers' not in c for c in configs))
        frozen = self.plan('0', 's2s_esm2', FREEZE_ENCODER='true')
        random = self.plan('0', 's2s_esm2', ENCODER_INIT='random')
        self.assertTrue(all(t['config']['freeze_encoder'] for t in frozen['trials']))
        self.assertTrue(all(t['config']['encoder_init'] == 'random' for t in random['trials']))
        self.assertNotEqual(frozen['trials'][0]['trial_id'], random['trials'][0]['trial_id'])
        rejected = self.invoke('0', 's2s_heavy', DRY_RUN='1', FREEZE_ENCODER='true')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('require an S2S PLM', rejected.stderr)

    def test_named_discovery_and_planning_match_legacy_without_creating_outputs(self):
        listed = self.invoke('list')
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn('mlm_no_partner_controls', listed.stdout)
        self.assertIn('sweep:lr', listed.stdout)
        named = self.invoke('plan', 'mlm_no_partner_controls', '--gpus', '2,5', '--json')
        self.assertEqual(named.returncode, 0, named.stderr)
        legacy = self.plan('2,5', 'spec', 'configs/experiments/20261007_mlm_no_partner_controls.json')
        self.assertEqual(json.loads(named.stdout), legacy)
        sweep = self.invoke('plan', 'sweep:lr', '--config', 'configs/s2s/h3_s2s_scratch.json',
                            '--grid', 'lr=1e-4', '--grid', 'seed=42,43', '--json')
        self.assertEqual(sweep.returncode, 0, sweep.stderr)
        legacy_sweep = self.plan('0', 'sweep', 'lr', 'configs/s2s/h3_s2s_scratch.json',
                                 LRS='1e-4', SEEDS='42 43')
        self.assertEqual(json.loads(sweep.stdout), legacy_sweep)
        relocated = self.invoke('plan', 'mlm_no_partner_controls', '--data', str(self.corpus), '--json')
        self.assertEqual(relocated.returncode, 0, relocated.stderr)
        self.assertEqual(json.loads(relocated.stdout), legacy)
        self.assertFalse((self.workspace / 'results').exists())
        self.assertFalse(self.records())
        empty = self.invoke()
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self.assertIn('usage:', empty.stdout)
        self.assertFalse(self.records())

    def test_named_commands_reject_hidden_choices_ambiguous_names_and_invalid_devices(self):
        for key, value in (('SEED', '99'), ('EXPERIMENT_DIR', 'old_batch'), ('DRY_RUN', '1'),
                           ('LRS', '1e-3'), ('DATA_PATH', str(self.corpus))):
            with self.subTest(key=key):
                result = self.invoke('run', 'mlm_no_partner_controls', **{key: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('unset ' + key, result.stderr)
        for args in (('plan', 'nonexistent'), ('plan', 'compare', '--gpus', '0,0'),
                     ('plan', 'compare', '--grid', 'seed=42'),
                     ('plan', 'sweep:lr', '--grid', 'seed=42', '--grid', 'seed=43')):
            with self.subTest(args=args):
                self.assertNotEqual(self.invoke(*args).returncode, 0)
        original = self.workspace / 'configs/experiments/20261007_mlm_no_partner_controls.json'
        shutil.copyfile(original, original.with_name('duplicate.json'))
        result = self.invoke('list')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Ambiguous experiment name', result.stderr)
        explicit = self.invoke('plan', str(original), '--json')
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertEqual(json.loads(explicit.stdout)['name'], 'mlm_no_partner_controls')
        self.assertFalse(self.records())

    def test_named_status_and_resume_use_frozen_configs_and_retry_only_incomplete_trials(self):
        output = self.workspace / 'named batch'
        first = self.invoke('run', 'mlm_no_partner_controls', '--gpus', '2,5', '--output', str(output))
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(len(self.records()), 2)
        status = self.invoke('status', str(output), '--json')
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(status.stdout)['completed'], 2)
        retry_as_new = self.invoke('run', 'mlm_no_partner_controls', '--output', str(output))
        self.assertNotEqual(retry_as_new.returncode, 0)
        self.assertIn('use run.sh resume', retry_as_new.stderr)
        spec_path = self.workspace / 'configs/experiments/20261007_mlm_no_partner_controls.json'
        spec = json.loads(spec_path.read_text())
        spec['trials'][0]['overrides']['seed'] = 99
        spec_path.write_text(json.dumps(spec))
        row = json.loads(status.stdout)['trials'][0]
        (Path(row['run_dir']) / 'generation_eval.json').unlink()
        incomplete = self.invoke('status', str(output), '--json')
        self.assertEqual(json.loads(incomplete.stdout)['completed'], 1)
        self.assertEqual(json.loads(incomplete.stdout)['trials'][0]['status'], 'incomplete')
        resumed = self.invoke('resume', str(output), '--gpus', '5')
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertEqual(resumed.stdout.count('SKIP completed'), 1)
        self.assertEqual(len(self.records()), 3)
        self.assertEqual({r['config']['seed'] for r in self.records()}, {42, 43})
        self.assertEqual(self.records()[-1]['gpu'], '5')
        self.corpus.write_text(self.corpus.read_text() + '\n')
        rejected = self.invoke('resume', str(output))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('Dataset differs', rejected.stderr)
        self.assertEqual(len(self.records()), 3)

    def test_named_resume_checks_source_trial_identity_and_active_lock(self):
        output = self.workspace / 'frozen'
        launched = self.invoke('run', 'mlm_no_partner_controls', '--output', str(output))
        self.assertEqual(launched.returncode, 0, launched.stdout + launched.stderr)
        plan_path = output / 'plan.json'
        original = json.loads(plan_path.read_text())
        changed = json.loads(plan_path.read_text())
        changed['runtime']['source_sha256'] = 'changed'
        plan_path.write_text(json.dumps(changed))
        rejected = self.invoke('resume', str(output))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('Source/runtime differs', rejected.stderr)
        changed = json.loads(json.dumps(original))
        changed['trials'][0]['config']['seed'] = 99
        plan_path.write_text(json.dumps(changed))
        rejected = self.invoke('resume', str(output))
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('Frozen trial identity differs', rejected.stderr)
        plan_path.write_text(json.dumps(original))
        with (output / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status = self.invoke('status', str(output), '--json')
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertTrue(json.loads(status.stdout)['active'])
            rejected = self.invoke('resume', str(output))
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('Another runner', rejected.stderr)
        self.assertEqual(len(self.records()), 2)

    def test_compare_matches_generation_protocol_and_shards_on_one_or_multiple_devices(self):
        for devices in ('2', '2,5'):
            with self.subTest(devices=devices):
                before = len(self.records())
                result = self.invoke(devices, 'compare')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                # Each invocation has its own batch, even when launched within the same second.
                batches = {}
                for r in self.records():
                    batches.setdefault(r['batch'], []).append(r)
                records = list(batches.values())[-1] if before == 0 else next(
                    rs for batch, rs in batches.items()
                    if batch == str((self.workspace / 'results/runs/latest').resolve()))
                self.assertEqual(len(records), 6)
                self.assertEqual({r['gpu'] for r in records}, set(devices.split(',')))
                for r in records:
                    c = r['config']
                    self.assertEqual((c['max_len'], c['gen_top_p'], c['gen_seed']), (384, .9, 42))
                    self.assertEqual((c['gen_num_examples'], c['gen_num_samples']), (150, 20))
                    self.assertEqual(c['data_path'], str(self.corpus))
                    if c['method'] == 'mlm':
                        self.assertEqual(c['gen_length_mode'], 'inferred')
                    if c['method'] == 's2s':
                        self.assertEqual((c['num_encoder_layers'], c['num_decoder_layers']), (2, 2))
                batch = Path(records[0]['batch'])
                self.assertTrue((batch / 'plan.json').is_file())
                with (batch / 'runs.tsv').open() as f:
                    rows = list(csv.DictReader(f, delimiter='\t'))
                self.assertEqual(len(rows), 6)
                self.assertTrue(all(r['status'] == 'completed' and Path(r['log']).is_file() for r in rows))
                self.assertEqual((self.workspace / 'results/runs/latest').resolve(), batch)

    def test_overlapping_suites_are_deduplicated_and_training_seed_keeps_generation_seed(self):
        plan = self.plan('2', 'compare', 'ar_heavy', SEED='43')
        self.assertEqual(len(plan['trials']), 10)
        self.assertTrue(plan['name'].endswith('_s43'))
        for trial in plan['trials']:
            self.assertEqual(trial['config']['seed'], 43)
            self.assertEqual(trial['config']['data_seed'], 43)
            self.assertEqual(trial['config'].get('gen_seed', 42), 42)

    def test_sweep_alias_produces_identical_plan_and_dry_run_writes_no_outputs(self):
        first = self.plan('2,5', 'sweep', 'lr', 'configs/s2s/h3_s2s_scratch.json')
        result = self.invoke('2,5', 'lr', 'configs/s2s/h3_s2s_scratch.json',
                             script='sweep.sh', DRY_RUN='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(first, json.loads(result.stdout[result.stdout.index('{'):]))
        self.assertFalse((self.workspace / 'results').exists())
        self.assertFalse(self.records())

    def test_each_architecture_dimension_can_vary_without_the_other(self):
        for method in ('ar', 's2s'):
            base = f'configs/{method}/h3_{method}_scratch.json'
            plan = self.plan('0', 'sweep', 'lr', base, NUM_LAYERS='2 6', LRS='1e-4', SEEDS='42')
            self.assertEqual([t['config']['num_layers'] for t in plan['trials']], [2, 6])
            self.assertEqual({t['config']['hidden_size'] for t in plan['trials']}, {128})
            if method == 's2s':
                self.assertEqual([t['config']['num_decoder_layers'] for t in plan['trials']], [2, 6])
            plan = self.plan('0', 'sweep', 'lr', base, HIDDEN_SIZES='128 192', LRS='1e-4', SEEDS='42')
            self.assertEqual([t['config']['hidden_size'] for t in plan['trials']], [128, 192])
            self.assertEqual({t['config']['num_layers'] for t in plan['trials']}, {4})

    def test_an_unpaired_corpus_refuses_partner_context_instead_of_ignoring_it(self):
        """Unpaired OAS has no light chain. Asking for it must fail, not train uncontextualised.

        Without this the run trains with no context, is still named `_lc`, and lands in the
        partner-context column of every table -- a silently invalid comparison.
        """
        unpaired = self.workspace / 'unpaired.json'
        unpaired.write_text(json.dumps([{'split': s, 'heavy': {'full_aa': 'H'}, 'light': None}
                                        for s in ('train', 'validation', 'test')]))
        for name in ('configs/mlm/heavy/h3_scratch_lc.json', 'configs/mlm/heavy/h3_scratch.json'):
            path = self.workspace / name
            config = json.loads(path.read_text())
            config['data_path'] = str(unpaired)
            path.write_text(json.dumps(config))

        result = self.invoke('0', 'sweep', 'lr', 'configs/mlm/heavy/h3_scratch_lc.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('partner-chain', result.stderr)
        self.assertIn('light', result.stderr)
        self.assertFalse(self.records(), "nothing may train before the gate rejects the plan")

        # The same corpus is fine for a run that never asks for the partner chain.
        plan = self.plan('0', 'sweep', 'lr', 'configs/mlm/heavy/h3_scratch.json')
        self.assertEqual({t['config']['data_path'] for t in plan['trials']}, {str(unpaired)})

    def test_both_modes_honor_config_corpus_and_reject_missing_antigen_before_training(self):
        for args in (('0', 'mlm_antigen'), ('0', 'sweep', 'lr', 'configs/mlm/antigen/h3_scratch_ag.json')):
            result = self.invoke(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('antigen', result.stderr)
            self.assertFalse(self.records())
        base = self.workspace / 'configs/mlm/antigen/h3_scratch_ag.json'
        config = json.loads(base.read_text())
        antigen = self.workspace / 'antigen.json'
        antigen.write_text(json.dumps(
            [{'split': s, 'antigen': {'sequence': 'ACD'},
              'heavy': {'full_aa': 'H'}, 'light': {'full_aa': 'L'}}
             for s in ('train', 'validation', 'test')]))
        config['data_path'] = str(antigen)
        base.write_text(json.dumps(config))
        plan = self.plan('0', 'sweep', 'lr', str(base))
        self.assertEqual({t['config']['data_path'] for t in plan['trials']}, {str(antigen)})
        result = self.invoke('0', 'sweep', 'lr', str(base), DATA_PATH=str(self.corpus))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('antigen', result.stderr)

    def test_sweep_requires_validation_and_rejects_unsupported_loss_method(self):
        self.corpus.write_text(json.dumps(
            [{'split': s, 'heavy': {'full_aa': 'H'}, 'light': {'full_aa': 'L'}}
             for s in ('train', 'test')]))
        result = self.invoke('0', 'sweep', 'lr')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('validation', result.stderr)
        result = self.invoke('0', 'sweep', 'loss', 'configs/ar/h3_ar_scratch.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Loss sweeps require MLM', result.stderr)
        self.assertFalse(self.records())

    def test_loss_grid_deduplicates_equivalent_objectives_and_seeds(self):
        plan = self.plan('0', 'sweep', 'loss', SEEDS='42 42 43 44')
        self.assertEqual(len(plan['trials']), 9)
        self.assertEqual(len({t['setting_id'] for t in plan['trials']}), 3)
        self.assertEqual(plan['rank_metric'], 'judge_ppl')

    def test_resume_skips_only_matching_completed_trials_and_retries_missing_artifacts(self):
        output = str(self.workspace / 'resume')
        args = ('0,1', 'sweep', 'lr', 'configs/s2s/h3_s2s_scratch.json')
        overrides = dict(EXPERIMENT_DIR=output, LRS='1e-4', SEEDS='42 43')
        first = self.invoke(*args, **overrides)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(len(self.records()), 2)
        second = self.invoke(*args, **overrides)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(second.stdout.count('SKIP completed'), 2)
        self.assertEqual(len(self.records()), 2)
        with (Path(output) / 'runs.tsv').open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        (Path(rows[0]['run_dir']) / 'epoch_metrics.csv').unlink()
        third = self.invoke(*args, **overrides)
        self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
        self.assertEqual(len(self.records()), 3)
        with (Path(output) / 'runs.tsv').open() as f:
            self.assertEqual(len(list(csv.DictReader(f, delimiter='\t'))), 2)
        self.assertFalse((self.workspace / 'results/runs/latest').exists())

    def test_changed_method_scheduler_protocol_or_dataset_cannot_resume(self):
        output = str(self.workspace / 'resume')
        args = ('0', 'sweep', 'lr', 'configs/ar/h3_ar_scratch.json')
        options = dict(EXPERIMENT_DIR=output, LRS='1e-4', SEEDS='42')
        result = self.invoke(*args, **options)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for changed_args, extra in (
            (('0', 'sweep', 'lr', 'configs/s2s/h3_s2s_scratch.json'), {}),
            (args, {'SCHEDULER': 'cosine'}),
            (args, {'GEN_EVAL': 'true'}),
            (args, {'SELECT_METRIC': 'accuracy'}),
        ):
            result = self.invoke(*changed_args, **dict(options, **extra))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Experiment differs', result.stderr)
        self.corpus.write_text(self.corpus.read_text() + '\n')
        result = self.invoke(*args, **options)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Experiment differs', result.stderr)
        self.assertEqual(len(self.records()), 1)

    def test_failure_and_missing_generation_are_nonzero_but_remaining_trials_finish(self):
        for devices in ('0', '0,1'):
            result = self.invoke(devices, 'compare', CDR_TEST_FAIL_CONFIG='h3_ar_scratch',
                                 CDR_TEST_INCOMPLETE='h3_s2s_scratch')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('4/6 completed', result.stdout)
        self.assertEqual(len(self.records()), 12)

    def test_unknown_suites_duplicate_devices_and_legacy_resume_fail_before_training(self):
        for args in (('0', 'made_up'), ('0,0', 'compare')):
            self.assertNotEqual(self.invoke(*args).returncode, 0)
        output = self.workspace / 'legacy'
        output.mkdir()
        (output / 'runs.tsv').write_text('old data')
        result = self.invoke('0', 'compare', EXPERIMENT_DIR=str(output))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no matching plan.json', result.stderr)
        self.assertFalse(self.records())

    def test_interrupt_stops_active_training_and_does_not_launch_remaining_trials(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen(
                ['bash', str(self.workspace / 'scripts/run.sh'), '0', 'compare'],
                cwd=self.workspace.parent, env=dict(self.env, CDR_TEST_SLOW='1'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not self.records() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(len(self.records()), 1)
                process.send_signal(signal.SIGINT)
                self.assertNotEqual(process.wait(timeout=5), 0)
                self.assertEqual(len(self.records()), 1)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == '__main__':
    unittest.main()
