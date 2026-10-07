"""Named experiment commands; the existing planner/executor owns all training.

  list    -- discover recipes, suites and sweep presets without loading data
  plan    -- validate and inspect a batch without creating files
  run     -- start a fresh batch with explicit GPU/output choices
  status  -- read the frozen plan, journal and completion artifacts
  resume  -- retry a frozen batch, skipping its verified completed trials

Scientific settings belong in configs. This interface lives outside the training
package so adding commands does not invalidate an ongoing batch's source identity.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdrseqgen import experiments as runner


LEGACY_OVERRIDES = {
    'DATA_PATH', 'DRY_RUN', 'EXPERIMENT_DIR', 'SWEEP_DIR', 'SEED',
    'ENCODER_INIT', 'FREEZE_ENCODER', 'GEN_EVAL', 'RANK_METRIC', 'SELECT_METRIC',
    'GEN_EXAMPLES', 'GEN_SAMPLES', 'PATIENCE', 'STOP_THRESHOLD', 'SCHEDULER',
    *runner.DIMENSIONS.values(),
}


def recipes():
    """Index top-level specs by their declared name; reject ambiguous names."""
    found = {}
    for path in sorted(Path('configs/experiments').glob('*.json')):
        spec = json.loads(path.read_text())
        name = spec['name']
        if name in found or name in runner.SUITES or name.startswith('sweep:'):
            raise ValueError(f'Ambiguous experiment name {name!r}: {path}; use unique names')
        found[name] = path
    return found


def table(headers, rows):
    """Print a small aligned table, preserving readable names and missing values."""
    rows = [[str(value) for value in row] for row in rows]
    widths = [max([len(header), *(len(row[i]) for row in rows)])
              for i, header in enumerate(headers)]
    for row in [headers, *rows]:
        print('  '.join(value.ljust(width) for value, width in zip(row, widths)).rstrip())


def show_list():
    """List available choices without hashing corpora or importing model code."""
    rows = []
    for name, path in recipes().items():
        trials, _ = runner.spec_trials(path, {})
        rows.append(('recipe', name, len(trials), str(path)))
    rows.extend(('suite', name, len(set(paths)), 'built-in model/loop group')
                for name, paths in runner.SUITES.items())
    rows.extend(('sweep', f'sweep:{name}', 'grid', 'use --config and optionally --grid')
                for name in runner.PRESETS)
    table(['KIND', 'NAME', 'TRIALS', 'DEFINITION'], rows)


def explicit_environment():
    """Refuse hidden legacy experiment choices; keep ordinary runtime/cache vars."""
    stale = sorted(key for key in LEGACY_OVERRIDES if os.environ.get(key))
    if stale:
        raise ValueError('Named commands use configs and flags. Clear legacy overrides first: '
                         'unset ' + ' '.join(stale))
    return dict(os.environ)


def make_plan(args):
    """Translate a named recipe, suite or explicit sweep grid into the shared plan."""
    env = explicit_environment()
    if args.data:
        env['DATA_PATH'] = args.data
    if args.experiment.startswith('sweep:'):
        preset = args.experiment.removeprefix('sweep:')
        base = args.config or 'configs/mlm/heavy/h3_scratch.json'
        for entry in args.grid:
            dimension, separator, values = entry.partition('=')
            if not separator or dimension not in runner.DIMENSIONS:
                raise ValueError('Use --grid DIM=VALUE,VALUE; dimensions: '
                                 + ', '.join(runner.DIMENSIONS))
            parts = values.split(',')
            if not all(part.strip() for part in parts):
                raise ValueError(f'Empty grid value: {entry}')
            key = runner.DIMENSIONS[dimension]
            if key in env:
                raise ValueError(f'Duplicate grid dimension: {dimension}')
            env[key] = ' '.join(part.strip() for part in parts)
        arguments = [args.gpus, 'sweep', preset, base]
    else:
        if args.config or args.grid:
            raise ValueError('--config and --grid apply to sweep:PRESET only')
        if args.experiment.endswith('.json'):
            arguments = [args.gpus, 'spec', args.experiment]
        elif args.experiment in runner.SUITES:
            arguments = [args.gpus, args.experiment]
        else:
            path = recipes().get(args.experiment)
            if path is None:
                raise ValueError(f'Unknown experiment {args.experiment!r}; use run.sh list')
            arguments = [args.gpus, 'spec', str(path)]
    plan, devices = runner.build_plan(arguments, env)
    return plan, devices, env


def show_plan(plan, devices):
    """Show resolved arms and the evaluation information that matters to a comparison."""
    print(f"{plan['name']}: {len(plan['trials'])} trials; GPU(s) {','.join(devices)}")
    rows = []
    for trial in plan['trials']:
        c = trial['config']
        partner = 'with' if c.get('include_partner_chain', False) else 'without'
        evaluation = 'with+without' if c.get('gen_eval_context_ablation') else partner
        length = c.get('gen_length_mode', 'fixed_native') if c['method'] == 'mlm' else 'EOS'
        budget = (f"{c.get('gen_num_examples', 150)}x{c.get('gen_num_samples', 20)}"
                  if runner.needs_generation(c) else 'disabled')
        rows.append((trial['suffix'] or Path(trial['source']).stem, c['method'],
                     c['model_type'], c.get('seed', 42), partner,
                     c.get('context_dropout', 0), evaluation, length, budget,
                     c.get('gen_judge_partner_context', 'trained')))
    table(['TRIAL', 'METHOD', 'MODEL', 'SEED', 'TRAIN', 'DROP', 'EVAL',
           'LENGTH', 'DRAWS', 'JUDGE CONTEXT'], rows)
    for path, data in plan['datasets'].items():
        print(f"Data: {path}\nSHA-256: {data['sha256']}; splits: {data['splits']}")
    print(f"Source: {plan['runtime']['source_sha256']}")


def read_plan(batch):
    """Read a planned batch; historical unplanned folders cannot be resumed."""
    path = batch / 'plan.json'
    if not path.is_file():
        raise ValueError(f'{batch} has no plan.json; historical results remain in RESULTS.md')
    plan = json.loads(path.read_text())
    if (not isinstance(plan, dict) or plan.get('schema_version') != 1
            or plan.get('mode') not in {'suite', 'sweep'}
            or not isinstance(plan.get('trials'), list) or not plan['trials']):
        raise ValueError(f'Invalid frozen plan: {path}')
    return plan


def batch_active(batch):
    """Read the existing batch lock without creating or changing any files."""
    path = batch / '.lock'
    if not path.exists():
        return False
    with path.open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def batch_status(batch):
    """Treat a completed journal row as complete only when its artifacts agree."""
    plan = read_plan(batch)
    ledger = batch / 'runs.tsv'
    states = {}
    if ledger.is_file():
        with ledger.open() as handle:
            states = {row['trial_id']: row for row in csv.DictReader(handle, delimiter='\t')}
    rows = []
    for trial in plan['trials']:
        row = states.get(trial['trial_id'], {})
        status = row.get('status', 'pending')
        if status == 'completed' and not runner.completed_artifacts(
                row.get('run_dir'), trial['config']):
            status = 'incomplete'
        rows.append(dict(trial=trial['suffix'] or Path(trial['source']).stem,
                         seed=trial['config'].get('seed', 42), status=status,
                         run_dir=row.get('run_dir', ''), log=row.get('log', ''),
                         error=row.get('error', '')))
    return dict(batch=str(batch), name=plan['name'], active=batch_active(batch),
                completed=sum(row['status'] == 'completed' for row in rows),
                total=len(rows), trials=rows)


def frozen_resume_plan(batch):
    """Validate stored settings against current code/data, without rereading recipes.

    Recompute trial identities as well as corpus hashes so a modified frozen
    config cannot be executed under an old trial ID or skip an incompatible run.
    """
    plan = read_plan(batch)
    if runner.source_identity() != plan['runtime']:
        raise ValueError('Source/runtime differs from the frozen plan; use the original '
                         'checkout/environment or start a new batch')
    checked = copy.deepcopy(plan)
    actual_data = runner.validate_datasets(checked['trials'], checked['mode'])
    if actual_data != plan['datasets'] or checked != plan:
        raise ValueError('Dataset differs from the frozen plan; restore the original corpus')
    seen = set()
    for trial in plan['trials']:
        config = {k: v for k, v in trial['config'].items() if k != 'data_path'}
        dataset = actual_data[trial['config']['data_path']]['sha256']
        identity = runner.fingerprint(dict(config=config, dataset=dataset, runtime=plan['runtime']))
        setting = {k: v for k, v in config.items() if k not in {'seed', 'data_seed'}}
        setting_id = runner.fingerprint(dict(config=setting, dataset=dataset, runtime=plan['runtime']))
        if (identity != trial['trial_id'] or setting_id != trial['setting_id']
                or identity in seen):
            raise ValueError('Frozen trial identity differs from its config; restore plan.json')
        seen.add(identity)
    return plan


def gpu_devices(value):
    """Parse an explicit device list with the same rules as the shared planner."""
    devices = re.split(r'[,\s]+', value.strip())
    if (any(not re.fullmatch(r'[0-9]+', device) for device in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError('Use distinct nonnegative GPU IDs, e.g. --gpus 0,1')
    return devices


def parser():
    result = argparse.ArgumentParser(
        description='Plan, launch and inspect CDRSeqGen experiments.',
        epilog='Legacy commands still work: run.sh <gpus> [suite ...], '
               'run.sh <gpus> spec FILE, run.sh <gpus> sweep [preset] [config].')
    commands = result.add_subparsers(dest='command')
    commands.add_parser('list', help='list named recipes, suites and sweeps')
    for name in ('plan', 'run'):
        command = commands.add_parser(name, help=('inspect without running' if name == 'plan'
                                                  else 'start a fresh batch'))
        command.add_argument('experiment', help='recipe name, spec JSON, suite, or sweep:PRESET')
        command.add_argument('--gpus', default='0', help='comma-separated GPU IDs (default: 0)')
        command.add_argument('--data', help='explicit corpus override')
        command.add_argument('--config', help='base config for sweep:PRESET')
        command.add_argument('--grid', action='append', default=[], metavar='DIM=VALUES',
                             help='sweep dimension, e.g. --grid seed=42,43')
        if name == 'plan':
            command.add_argument('--json', action='store_true', help='print the full resolved plan')
        else:
            command.add_argument('--output', help='new batch directory (default: timestamped)')
    status = commands.add_parser('status', help='read batch progress and verify completed artifacts')
    status.add_argument('batch', help='batch directory containing plan.json')
    status.add_argument('--json', action='store_true')
    resume = commands.add_parser('resume', help='skip completed trials; restart unfinished training')
    resume.add_argument('batch', help='batch directory containing plan.json')
    resume.add_argument('--gpus', default='0')
    return result


def main(argv=None):
    """Run only the requested command; an empty invocation displays help."""
    os.chdir(ROOT)
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        if args.command is None:
            argument_parser.print_help()
        elif args.command == 'list':
            show_list()
        elif args.command == 'status':
            status = batch_status(Path(args.batch).expanduser().resolve())
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print(f"{status['name']}: {status['completed']}/{status['total']} complete; "
                      f"batch lock {'active' if status['active'] else 'inactive'}")
                table(['TRIAL', 'SEED', 'STATUS', 'RUN DIRECTORY'],
                      [(row['trial'], row['seed'], row['status'], row['run_dir'])
                       for row in status['trials']])
                if not status['active'] and any(row['status'] == 'running' for row in status['trials']):
                    print('The running journal row has an inactive batch lock; inspect its log.')
        elif args.command == 'resume':
            env = explicit_environment()
            devices = gpu_devices(args.gpus)
            batch = Path(args.batch).expanduser().resolve()
            plan = frozen_resume_plan(batch)
            show_plan(plan, devices)
            print('Batch resume skips verified completed trials. Unfinished training restarts '
                  'from scratch; checkpoint recovery is a separate operation.', flush=True)
            return runner.execute(plan, devices, batch, env)
        else:
            plan, devices, env = make_plan(args)
            if args.command == 'plan':
                if args.json:
                    print(json.dumps(plan, indent=2))
                else:
                    show_plan(plan, devices)
            else:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                tag = ('sweep_' if plan['mode'] == 'sweep' else '') + plan['name']
                batch = Path(args.output or f'results/runs/{stamp}_{tag}').expanduser().resolve()
                if batch.exists() and (not batch.is_dir() or any(batch.iterdir())):
                    raise ValueError(f'{batch} already exists; use run.sh resume BATCH')
                show_plan(plan, devices)
                return runner.execute(plan, devices, batch, env)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Interrupted. Inspect the saved batch with run.sh status BATCH.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
