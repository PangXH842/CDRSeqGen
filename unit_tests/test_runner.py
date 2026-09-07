"""Execute the real batch runner; replace only Conda, training, and plotting boundaries.

All files and subprocess output stay in a temporary workspace. The fake training process records
its effective config and assigned device; no GPU, downloaded model, or user Conda setup is touched.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BatchRunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        shutil.copytree(ROOT / 'configs', self.workspace / 'configs')
        for config in (self.workspace / 'configs').glob('*.json'):
            data_path = json.loads(config.read_text()).get('data_path')
            if data_path:
                target = self.workspace / data_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('[]')
        self.corpus = self.workspace / 'alternate corpus.json'
        self.corpus.write_text('[{}]')
        startup = self.workspace / 'shell-startup'
        startup.write_text('source() { :; }\nconda() { :; }\n')
        binary_dir = self.workspace / 'bin'
        binary_dir.mkdir()
        wrapper = binary_dir / 'python'
        wrapper.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == 'train.py':
    config = json.loads(Path(args[1]).read_text())
    for flag, value in zip(args[2::2], args[3::2]):
        config[flag.removeprefix('--')] = value
    output = Path('captured')
    output.mkdir(exist_ok=True)
    (output / Path(args[1]).name).write_text(json.dumps({
        'config': config, 'argv': args, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
        'batch': os.environ['CDRSEQGEN_RESULTS_ROOT']}))
    sys.exit(1 if Path(args[1]).name == os.environ.get('CDR_TEST_FAIL_CONFIG') else 0)
if args[0] == '-m':
    sys.exit(0)  # plotting is outside the runner contract
os.execv(sys.executable, [sys.executable, *args])  # execute real JSON/data preflight checks
''')
        wrapper.chmod(0o755)
        self.env = dict(os.environ, BASH_ENV=str(startup),
                        PATH=str(binary_dir) + os.pathsep + os.environ['PATH'],
                        DATA_PATH=str(self.corpus))
        for key in ('CDRSEQGEN_RESULTS_ROOT', 'CDRSEQGEN_SKIP_LATEST', 'CDR_TEST_FAIL_CONFIG'):
            self.env.pop(key, None)

    def run_batch(self, devices, *suites):
        result = subprocess.run(['bash', str(ROOT / 'scripts/run.sh'), devices, *suites],
                                cwd=self.workspace, env=self.env, text=True,
                                capture_output=True, timeout=30)
        records = {p.stem: json.loads(p.read_text())
                   for p in (self.workspace / 'captured').glob('*.json')}
        return result, records

    def test_compare_preserves_corpus_and_matches_sampling_on_one_or_multiple_devices(self):
        # Clear captured calls between subcases so a missed invocation cannot pass on stale output.
        for devices in ('2', '2,5'):
            with self.subTest(devices=devices):
                result, records = self.run_batch(devices, 'compare')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(set(records), {'h3_scratch', 'h3_scratch_lc',
                                                'h3_ar_scratch', 'h3_ar_scratch_lc',
                                                'h3_s2s_scratch', 'h3_s2s_scratch_lc'})
                self.assertEqual({r['gpu'] for r in records.values()}, set(devices.split(',')))
                self.assertEqual(len({r['batch'] for r in records.values()}), 1)
                for suffix in ('', '_lc'):
                    masked = records['h3_scratch' + suffix]['config']
                    causal = records['h3_ar_scratch' + suffix]['config']
                    seq2seq = records['h3_s2s_scratch' + suffix]['config']
                    # Only the masked arm needs forcing: the other two choose length natively and
                    # already ship the matched sampling policy.
                    self.assertEqual(masked['gen_length_mode'], 'inferred')
                    for key in ('max_len', 'gen_top_p'):
                        self.assertEqual(float(masked[key]), float(causal[key]))
                        self.assertEqual(float(seq2seq[key]), float(causal[key]))
                    for config in (masked, causal, seq2seq):
                        self.assertEqual(config['data_path'], str(self.corpus))
                    # Capacity is the one axis the suite deliberately does NOT match, so the
                    # encoder-decoder arm must carry explicit depths rather than inherit num_layers.
                    self.assertEqual(int(seq2seq['num_encoder_layers']), 2)
                    self.assertEqual(int(seq2seq['num_decoder_layers']), 2)
                if ',' in devices:
                    latest = self.workspace / 'results/runs/latest'
                    self.assertTrue(latest.is_symlink())
                    self.assertEqual(latest.resolve(), (self.workspace / next(iter(records.values()))['batch']).resolve())
                for p in (self.workspace / 'captured').glob('*.json'):
                    p.unlink()

    def test_overlapping_suites_do_not_launch_duplicate_configs(self):
        # compare (6) + ar_heavy (6) share h3_ar_scratch{,_lc}, so the union is 10, not 12.
        result, records = self.run_batch('2', 'compare', 'ar_heavy')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(records), 10)
        self.assertEqual(result.stdout.count('Starting run:'), 10)

    def test_seed_override_tags_the_batch_and_reaches_every_run(self):
        """Multi-seeding is how the README's single-seed caveat gets discharged, so the override
        has to reach the runs AND keep them findable. The seed tags the batch FOLDER, never the run
        name: batch_tables.model_family would read a name suffix as an unknown model family and
        drop the run from every table without erroring."""
        self.env['SEED'] = '43'
        result, records = self.run_batch('2', 's2s_heavy')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(set(records), {'h3_s2s_scratch', 'h3_s2s_scratch_lc'})
        for name, record in records.items():
            with self.subTest(run=name):
                self.assertEqual(str(record['config']['seed']), '43')
                self.assertTrue(record['batch'].endswith('_s2s_heavy_s43'), record['batch'])
                # gen_seed picks WHICH frameworks are evaluated. Holding it fixed is what keeps
                # two seeds scoring the same 150 and therefore comparable.
                self.assertNotIn('--gen_seed', record['argv'])

    def test_without_the_override_the_batch_carries_no_seed_tag(self):
        result, records = self.run_batch('2', 's2s_heavy')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for record in records.values():
            self.assertNotIn('_s', os.path.basename(record['batch']).replace('_s2s_heavy', ''))
            self.assertNotIn('--seed', record['argv'])

    def test_antigen_suite_rejects_a_corpus_without_antigen_before_training(self):
        result, records = self.run_batch('2', 'mlm_antigen')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('antigen', result.stdout)
        self.assertEqual(records, {})

    def test_failed_trial_does_not_hide_failure_or_prevent_remaining_trials(self):
        self.env['CDR_TEST_FAIL_CONFIG'] = 'h3_scratch.json'
        for devices in ('2', '2,5'):
            with self.subTest(devices=devices):
                result, records = self.run_batch(devices, 'compare')
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(records), 6)
                self.assertIn('5/6 succeeded', result.stdout)
                for p in (self.workspace / 'captured').glob('*.json'):
                    p.unlink()

    def test_unknown_suite_fails_before_training(self):
        result, records = self.run_batch('2', 'not_a_suite')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(records, {})


if __name__ == '__main__':
    unittest.main()
