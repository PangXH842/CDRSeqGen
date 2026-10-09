"""Control flow of scripts/weekend_queue.sh with its heavy commands stubbed: order, skipping, isolation, retries."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/weekend_queue.sh'
ENVIRONMENTS = [Path('/home/pang/miniconda3/envs/cdrprj/bin/python'), Path('/mnt/hdd/pang/envs/igcraft/bin/python')]


@unittest.skipUnless(all(p.exists() for p in ENVIRONMENTS), 'the queue preflight needs the project environments')
class QueueFlowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.queue = Path(temp.name) / 'queue'

    def run_queue(self, command='start', **env):
        # A distinct tag, so that even a bug could not touch the real weekend batch directories.
        environment = dict(os.environ, QUEUE_STUB='1', QUEUE_TAG='queue-test', QUEUE_DIR=str(self.queue), RETRY_SLEEP='0', **env)
        result = subprocess.run(['bash', str(SCRIPT), command], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
        self.assert_nothing_real(result)
        return result

    def assert_nothing_real(self, result):
        """Stub mode must never reach a heavy command: no refusal, no batch directory, no generated recipe."""
        self.assertNotIn('REFUSED', result.stdout + result.stderr)
        for pattern in ('results/runs/queue-test_*', 'results/benchmark/queue-test_*'):
            self.assertEqual(list(ROOT.glob(pattern)), [], pattern)
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
        self.assertEqual(list(ROOT.glob('results/runs/queue-test_*')), [])

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


if __name__ == '__main__':
    unittest.main()
