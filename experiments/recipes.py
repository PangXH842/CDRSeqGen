"""Named experiment commands: discover recipes, show and plan them, refuse hidden legacy overrides."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .core import needs_generation
from .plan import build_plan
from .spec import DIMENSIONS, SUITES, PRESETS, spec_trials


LEGACY_OVERRIDES = {
    'DATA_PATH', 'DRY_RUN', 'EXPERIMENT_DIR', 'SWEEP_DIR', 'SEED',
    'ENCODER_INIT', 'FREEZE_ENCODER', 'GEN_EVAL', 'RANK_METRIC', 'SELECT_METRIC',
    'GEN_EXAMPLES', 'GEN_SAMPLES', 'PATIENCE', 'STOP_THRESHOLD', 'SCHEDULER',
    *DIMENSIONS.values(),
}


def recipes():
    """Index top-level specs by their declared name; reject ambiguous names."""
    found = {}
    for path in sorted(Path('experiments/specs').glob('*.json')):
        spec = json.loads(path.read_text())
        name = spec['name']
        if name in found or name in SUITES or name.startswith('sweep:'):
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
        trials, _ = spec_trials(path, {})
        rows.append(('recipe', name, len(trials), str(path)))
    rows.extend(('suite', name, len(set(paths)), 'built-in model/loop group')
                for name, paths in SUITES.items())
    rows.extend(('sweep', f'sweep:{name}', 'grid', 'use --config and optionally --grid')
                for name in PRESETS)
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
            if not separator or dimension not in DIMENSIONS:
                raise ValueError('Use --grid DIM=VALUE,VALUE; dimensions: '
                                 + ', '.join(DIMENSIONS))
            parts = values.split(',')
            if not all(part.strip() for part in parts):
                raise ValueError(f'Empty grid value: {entry}')
            key = DIMENSIONS[dimension]
            if key in env:
                raise ValueError(f'Duplicate grid dimension: {dimension}')
            env[key] = ' '.join(part.strip() for part in parts)
        arguments = [args.gpus, 'sweep', preset, base]
    else:
        if args.config or args.grid:
            raise ValueError('--config and --grid apply to sweep:PRESET only')
        if args.experiment.endswith('.json'):
            arguments = [args.gpus, 'spec', args.experiment]
        elif args.experiment in SUITES:
            arguments = [args.gpus, args.experiment]
        else:
            path = recipes().get(args.experiment)
            if path is None:
                raise ValueError(f'Unknown experiment {args.experiment!r}; use run.sh list')
            arguments = [args.gpus, 'spec', str(path)]
    plan, devices = build_plan(arguments, env)
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
                  if needs_generation(c) else 'disabled')
        rows.append((trial['suffix'] or Path(trial['source']).stem, c['method'],
                     c['model_type'], c.get('seed', 42), partner,
                     c.get('context_dropout', 0), evaluation, length, budget,
                     c.get('gen_judge_partner_context', 'trained')))
    table(['TRIAL', 'METHOD', 'MODEL', 'SEED', 'TRAIN', 'DROP', 'EVAL',
           'LENGTH', 'DRAWS', 'JUDGE CONTEXT'], rows)
    for path, data in plan['datasets'].items():
        print(f"Data: {path}\nSHA-256: {data['sha256']}; splits: {data['splits']}")
    print(f"Source: {plan['runtime']['source_sha256']}")
