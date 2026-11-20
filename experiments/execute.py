"""One lifecycle for every plan: lock, freeze, shard trials over GPUs, journal, resume, report."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from cdrseqgen.artifacts import atomic_write_json
from cdrseqgen.visualization.sweep_tables import KEY

from .core import completed_artifacts
from .report import report


RUN_FIELDS = [*KEY, "seed", "scheduler", "status", "run_dir", "trial_id", "setting_id",
              "config", "gpu", "log", "error"]


def execute(plan, devices, output, env):
    """One lifecycle for all trials, with one sequential shard per device."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another runner is using {output}") from None
        plan_path = output / 'plan.json'
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError(f"Experiment differs from {plan_path}; use a new EXPERIMENT_DIR. "
                                 "Resume requires unchanged configs, data, source, and protocol.")
        elif any(p.name != '.lock' for p in output.iterdir()):
            raise ValueError(f"{output} has no matching plan.json; use a fresh EXPERIMENT_DIR. "
                             "Legacy results remain readable but cannot be auto-resumed.")
        else:
            atomic_write_json(plan_path, plan)
        _execute_locked(plan, devices, output, env)
        with (output / 'runs.tsv').open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        failed = sum(r['status'] != 'completed' for r in rows)
        report(plan, output)
        if plan['mode'] == 'suite' and not failed:
            try:
                link = Path('results/runs/latest')
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    link.unlink()
                if not link.exists():
                    link.symlink_to(os.path.relpath(output, link.parent.resolve()))
            except OSError as exc:
                print(f"Could not update latest: {exc}", file=sys.stderr)
        print(f"{'FAILED' if failed else 'Done'}: {len(rows)-failed}/{len(rows)} completed; {output}")
        return 1 if failed else 0


def _execute_locked(plan, devices, output, env):
    configs_dir, logs_dir = output / '.configs', output / 'logs'
    configs_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    map_path = output / 'runs.tsv'
    states = {}
    if map_path.exists():
        with map_path.open() as handle:
            states = {r['trial_id']: r for r in csv.DictReader(handle, delimiter='\t')}
    mutex = threading.Lock()
    stop = threading.Event()
    active = set()

    def persist():
        pending = map_path.with_suffix('.tsv.tmp')
        with pending.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, delimiter='\t')
            writer.writeheader()
            writer.writerows(states[t['trial_id']] for t in plan['trials'] if t['trial_id'] in states)
        pending.replace(map_path)

    def shard(gpu, trials):
        for trial in trials:
            if stop.is_set():
                break
            identity, config = trial['trial_id'], trial['config']
            with mutex:
                old = states.get(identity, {})
                if old.get('status') == 'completed' and completed_artifacts(old.get('run_dir'), config):
                    print(f"SKIP completed: {trial['source']} {trial['suffix']}", flush=True)
                    continue
            config_path = configs_dir / f'{identity}.json'
            atomic_write_json(config_path, config)
            log = logs_dir / f"{Path(trial['source']).stem}_{identity[:12]}.log"
            child_env = dict(env, CUDA_VISIBLE_DEVICES=gpu, CDRSEQGEN_RESULTS_ROOT=str(output),
                             CDRSEQGEN_SKIP_LATEST='1', CDRSEQGEN_RUN_SUFFIX=trial['suffix'])
            row = {k: trial['grid'].get(k, '__keep__') for k in KEY}
            row.update(seed=config.get('seed', 42), scheduler=config.get('lr_scheduler_type', 'linear'),
                       status='running', run_dir='', trial_id=identity, setting_id=trial['setting_id'],
                       config=trial['source'], gpu=gpu, log=str(log), error='')
            with mutex:
                states[identity] = dict(row)
                persist()
            print(f"RUN gpu {gpu}: {trial['source']} {trial['suffix']} | {log}", flush=True)
            try:
                with log.open('a', buffering=1) as handle:
                    handle.write(f"\n--- attempt {datetime.now().isoformat()} ---\n")
                    with subprocess.Popen([sys.executable, 'train.py', str(config_path)], env=child_env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, bufsize=1) as process:
                        with mutex:
                            active.add(process)
                            if stop.is_set():
                                process.terminate()
                        for line in process.stdout:
                            handle.write(line)
                            if len(devices) == 1:
                                print(line, end='', flush=True)
                            marker = 'Workspace initialized at: '
                            if marker in line:
                                row['run_dir'] = line.split(marker, 1)[1].strip()
                        code = process.wait()
                        with mutex:
                            active.discard(process)
                row['status'] = ('completed' if code == 0 and completed_artifacts(row['run_dir'], config)
                                 else 'failed' if code else 'incomplete')
                if row['status'] != 'completed':
                    row['error'] = f'exit {code}' if code else 'Missing or incompatible run artifacts'
            except OSError as exc:
                row.update(status='failed', error=str(exc))
            with mutex:
                states[identity] = row
                persist()
            print(f"{row['status'].upper()}: {trial['source']} {trial['suffix']}", flush=True)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(shard, gpu, plan['trials'][i::len(devices)])
                   for i, gpu in enumerate(devices)]
        try:
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            with mutex:
                for process in active:
                    if process.poll() is None:
                        process.terminate()
            raise
