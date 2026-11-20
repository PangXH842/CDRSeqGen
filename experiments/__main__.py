"""One entry point for experiments: `python -m experiments ...`, run from the repository root.

  list | plan | run | status | resume      named recipes, suites and sweeps (was scripts/experiments.py)
  <gpus> [suite ...] | <gpus> sweep ... | <gpus> spec FILE    the legacy forms (was python -m cdrseqgen.experiments)
  benchmark [followup|variability|extend|leakage] ...          forwarded to the benchmark tools

The first argument picks the interface, exactly as scripts/run.sh does. run.sh keeps launching the old pair of
modules until those are retired; afterwards it only has to call this one.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from .core import batch_status, flag
from .execute import execute
from .plan import build_plan, frozen_resume_plan, gpu_devices
from .recipes import explicit_environment, make_plan, show_list, show_plan, table
from .spec import PRESETS, SUITES

ROOT = Path(__file__).resolve().parents[1]
NAMED_COMMANDS = {'list', 'plan', 'run', 'status', 'resume'}
BENCHMARK_TOOLS = ('followup', 'variability', 'extend', 'leakage')
PROG = 'python -m experiments'
LEGACY_DESCRIPTION = """Plan and execute experiments: suites and sweeps share everything after trial expansion.

The shell scripts only locate the repository and interpreter. A plan freezes merged input configs,
dataset hashes, and source/runtime identity; train.py still owns argument defaults and training.
One executor handles GPU sharding, logs, completion, resume, and reporting for either plan.
No model libraries are imported here, so orchestration tests need no GPU or downloaded weights.
"""


def parser():
    result = argparse.ArgumentParser(
        prog=PROG, description='Plan, launch and inspect CDRSeqGen experiments.',
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


def named_main(argv=None):
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
            return execute(plan, devices, batch, env)
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
                return execute(plan, devices, batch, env)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Interrupted. Inspect the saved batch with run.sh status BATCH.', file=sys.stderr)
        return 130


def legacy_main(argv=None):
    parser = argparse.ArgumentParser(prog=PROG, description=LEGACY_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Suites: {', '.join(SUITES)}\nPresets: {', '.join(PRESETS)}\n\n"
               "run.sh <gpus> [suite ...]\nrun.sh <gpus> sweep [preset] [base_config]\nrun.sh <gpus> spec <experiment.json>\n\n"
               "Shared overrides: PYTHON, DATA_PATH, DRY_RUN=1, EXPERIMENT_DIR (exact resume).\n"
               "S2S PLM controls (suites/sweeps): ENCODER_INIT=pretrained|random, FREEZE_ENCODER=true|false.\n"
               "Spec: base_config + shared/per-trial overrides; put seed/encoder controls in the file.\n"
               "Suite: SEED. Sweep: LRS, WARMUPS, WEIGHT_DECAYS, SEEDS, EPOCH_LIST, HIDDEN_SIZES,\n"
               "NUM_LAYERS, LOSS_TYPES, CORE_WEIGHTS, ANCHOR_WEIGHTS, GEN_EVAL, RANK_METRIC,\n"
               "SELECT_METRIC, GEN_EXAMPLES, GEN_SAMPLES, PATIENCE, STOP_THRESHOLD, SCHEDULER.\n"
               "SWEEP_DIR remains an alias for EXPERIMENT_DIR in sweep mode.")
    parser.add_argument('arguments', nargs='*', metavar='ARG')
    args = parser.parse_args(argv)
    try:
        plan, devices = build_plan(args.arguments, os.environ)
        print(f"{plan['mode']} {plan['name']}: {len(plan['trials'])} trials on GPU(s) {','.join(devices)}")
        if flag(os.environ.get('DRY_RUN', 'false')):
            print(json.dumps(plan, indent=2))
            return 0
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        tag = ('sweep_' if plan['mode'] == 'sweep' else '') + plan['name']
        output = os.environ.get('EXPERIMENT_DIR')
        legacy = os.environ.get('SWEEP_DIR') if plan['mode'] == 'sweep' else None
        if output and legacy and Path(output).resolve() != Path(legacy).resolve():
            raise ValueError('EXPERIMENT_DIR and SWEEP_DIR disagree')
        return execute(plan, devices, output or legacy or f'results/runs/{stamp}_{tag}', os.environ)
    except (OSError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


def benchmark_main(arguments):
    """Forward to the benchmark tool run.sh would have started (same names, same arguments)."""
    tool = arguments[0] if arguments and arguments[0] in BENCHMARK_TOOLS else None
    script = f"benchmark/{tool or 'run'}.py"
    rest = arguments[1:] if tool else arguments
    os.execv(sys.executable, [sys.executable, script, *rest])


def main(argv=None):
    """Choose the interface from the first argument, exactly as scripts/run.sh does."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    os.chdir(ROOT)
    if arguments[:1] == ['benchmark']:
        return benchmark_main(arguments[1:])
    first = arguments[0] if arguments else ''
    if first in NAMED_COMMANDS | {'', '-h', '--help'}:
        return named_main(arguments)
    return legacy_main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())

