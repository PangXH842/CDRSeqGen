"""Plan/run the inference-only comparison; scripts/run.sh benchmark exposes this file.

  plan   -- validate inputs and show the budget without model execution
  smoke  -- five frameworks, five draws, training seed 42 plus every external generator
  run    -- 150 frameworks, twenty draws, both training seeds plus every external generator
  status -- inspect saved outputs
  resume -- reuse a frozen batch plan and compatible adapter outputs

Scientific choices live in benchmark/recipes; trained checkpoints are read only. External generators
(benchmark/externals.py) run in their own environments and meet the scorer only through adapter-output
JSON, so a recipe may also name externals without any local arms to add a generator to a finished batch.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark._common import atomic_write_json, execution_source_identity, sha256_file, validate_manifest_record
from benchmark.adapters.cdrseqgen import load_run_config, resolve_decoding, validate_checkpoint_contract
from benchmark.externals import (
    DEFAULT_PACKAGES, EXTERNALS, RESERVED_LABELS, default_python, plan_externals, recipe_externals)


def read(path):
    return json.loads(Path(path).read_text())


def environment_identity(python, packages=DEFAULT_PACKAGES):
    """Read interpreter/package versions; do not import models or inspect GPUs.

    A package the environment lacks is recorded as null rather than aborting, since externals need not
    share the training stack. Resume passes the names a plan stored, so its comparison stays like-for-like.
    """
    script = ("import sys,json,importlib.metadata as m\n"
              "def version(name):\n"
              "    try: return m.version(name)\n"
              "    except m.PackageNotFoundError: return None\n"
              f"print(json.dumps(dict(python=sys.version,packages={{n:version(n) for n in {list(packages)!r}}})))")
    return json.loads(subprocess.check_output([str(python), '-c', script], text=True))


def freeze_external_inputs(entry, frozen_files):
    """Hash an external's input files into the plan, refusing any that is not its pinned digest."""
    for path, pin in EXTERNALS[entry['adapter']].inputs(entry['options']).items():
        digest = sha256_file(ROOT / path)
        if pin and digest != pin:
            raise ValueError(f"{entry['label']} input {path} has digest {digest}, not its pinned {pin}")
        frozen_files[str(path)] = digest


def build_plan(recipe_path, stage, cdr_python, external_pythons=None):
    """Validate a recipe into a frozen plan; `external_pythons` maps an external's label to its interpreter."""
    recipe = read(recipe_path)
    external_pythons = external_pythons or {}
    externals = [dict(label=entry['label'], adapter=entry['adapter'], options=entry['options'],
                      python=str(Path(external_pythons.get(entry['label']) or default_python(entry['adapter'])).resolve()))
                 for entry in recipe_externals(recipe)]
    unknown = set(external_pythons) - {entry['label'] for entry in externals}
    if unknown:
        raise ValueError(f'Interpreters given for externals this recipe does not name: {sorted(unknown)}')
    arm_labels = [arm['label'] for arm in recipe.get('arms', [])]
    labels = arm_labels + [entry['label'] for entry in externals]
    if not labels or len(labels) != len(set(labels)) or RESERVED_LABELS.intersection(labels):
        raise ValueError(f'Recipe needs at least one arm or external, with unique non-reserved labels: {labels}')
    for entry in externals:
        if not Path(entry['python']).is_file():
            raise ValueError(f"No interpreter for external {entry['label']!r} at {entry['python']}; "
                             f"create its environment or pass --external-python {entry['label']}=PATH")
    manifest_path = ROOT / recipe['smoke_manifest' if stage == 'smoke' else 'manifest']
    manifest = read(manifest_path)
    digest = sha256_file(manifest_path)
    assert digest == manifest_path.with_suffix('.json.sha256').read_text().split()[0]
    assert manifest['task']['track'] == 'length_free'
    assert manifest['selection']['split'] == 'validation'
    records = manifest['records']
    assert len(records) == (5 if stage == 'smoke' else 150)
    assert len({r['example_id'] for r in records}) == len(records)
    for record in records:
        validate_manifest_record(record)
        for entry in externals:
            EXTERNALS[entry['adapter']].check_record(record, recipe['sampling'], entry['options'])
    source = ROOT / 'data/processed' / manifest['source']['processed_file']
    assert sha256_file(source) == manifest['source']['processed_file_sha256']
    frozen_files = {str(manifest_path.relative_to(ROOT)): digest,
                    str(recipe_path.relative_to(ROOT)): sha256_file(recipe_path)}
    for entry in externals:
        freeze_external_inputs(entry, frozen_files)
    arms = []
    for arm in recipe.get('arms', []):
        if stage == 'smoke' and arm['training_seed'] != 42:
            continue
        checkpoint = ROOT / arm['checkpoint']
        config, config_path = load_run_config(checkpoint)
        validate_checkpoint_contract(config, manifest)
        assert config['method'] == arm['method'] and config['seed'] == arm['training_seed']
        assert config['model_type'] == 'scratch'
        decoding = resolve_decoding(config)
        paths = [checkpoint / 'model.safetensors', checkpoint / 'config.json', config_path]
        if arm['method'] == 'mlm':
            paths.extend([checkpoint / 'length_config.json', checkpoint / 'length_head.safetensors'])
            assert read(checkpoint / 'length_config.json')['seed'] == arm['training_seed']
        for path in paths:
            frozen_files[str(path.relative_to(ROOT))] = sha256_file(path)
        arms.append({**arm, 'decoding': decoding})
    samples = recipe['sampling']['smoke_num_samples' if stage == 'smoke' else 'num_samples']
    # One interpreter may serve several roles, so fingerprint the union of what each role needs.
    packages = {str(cdr_python): list(DEFAULT_PACKAGES)}
    for entry in externals:
        names = packages.setdefault(entry['python'], [])
        names.extend(n for n in EXTERNALS[entry['adapter']].packages if n not in names)
    return dict(schema_version='cdrseqgen.benchmark_batch.v2', name=recipe['name'], stage=stage,
                manifest=str(manifest_path.relative_to(ROOT)), frameworks=len(records),
                arms=arms, externals=externals, sampling={**recipe['sampling'], 'num_samples': samples},
                judge=recipe['judge'], endpoint=recipe['endpoint'], limits=recipe['limits'],
                requested_attempts=len(records) * samples * (len(arms) + len(externals)),
                cdr_python=str(cdr_python),
                environments={python: environment_identity(python, names) for python, names in packages.items()},
                execution_source=execution_source_identity(), frozen_files=frozen_files)


def validate_frozen_plan(plan):
    if plan['execution_source'] != execution_source_identity():
        raise ValueError('Benchmark source changed; preserve this batch and plan a new source-versioned run')
    for path, digest in plan['frozen_files'].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f'Frozen benchmark input changed: {path}')
    for python, expected in plan['environments'].items():
        if environment_identity(python, tuple(expected['packages'])) != expected:
            raise ValueError(f'Benchmark environment changed: {python}')


def generation_commands(plan, output):
    """Return explicit argument lists; adapters own tokenization and attempt accounting."""
    manifest = str(ROOT / plan['manifest'])
    s = plan['sampling']
    common = ['--manifest', manifest, '--num-samples', str(s['num_samples']),
              '--temperature', str(s['temperature']), '--top-p', str(s['top_p']),
              '--seed', str(s['seed']), '--device', 'cuda']
    commands = []
    for arm in plan['arms']:
        path = output / 'generations' / (arm['label'] + '.json')
        command = [plan['cdr_python'], str(ROOT / 'benchmark/adapters/cdrseqgen.py'), *common,
                   '--checkpoint', str(ROOT / arm['checkpoint']), '--output', str(path),
                   '--decoding', arm['decoding'], '--top-k', str(s['top_k']),
                   '--length-temperature', str(s['length_temperature'])]
        if arm['method'] != 'mlm':
            command += ['--max-new-tokens', str(s['max_new_tokens'])]
        commands.append((arm['label'], path, command))
    for entry in plan_externals(plan):
        spec = EXTERNALS[entry['adapter']]
        path = output / 'generations' / (entry['label'] + '.json')
        commands.append((entry['label'], path, [entry['python'], str(ROOT / spec.script), *common,
                                                *spec.arguments(path, s, entry['options'])]))
    return commands


def status(output):
    plan = read(output / 'plan.json')
    for label, path, _ in generation_commands(plan, output):
        payload = read(path) if path.exists() else {}
        records = payload.get('records', [])
        print(f"{label}: {len(records)}/{plan['frameworks']} frameworks; finished={bool(payload.get('finished_at'))}; "
              f"returned={sum(len(r['samples']) for r in records)}; "
              f"invalid={sum(r['invalid_attempts'] for r in records)}")
    print(f"Shared scores: {(output / 'scores/comparison.json').exists()}")


def execute(plan, output, gpu, resume=False):
    output.mkdir(parents=True, exist_ok=resume)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume:
            atomic_write_json(output / 'plan.json', plan)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        commands = generation_commands(plan, output)
        for label, path, command in commands:
            validate_frozen_plan(plan)
            if path.exists():
                command.append('--resume')
            print(f'Generate {label}: {path}', flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        validate_frozen_plan(plan)
        score_path = output / 'scores/comparison.json'
        if not score_path.exists():
            command = [plan['cdr_python'], str(ROOT / 'benchmark/scorer.py'),
                       '--manifest', str(ROOT / plan['manifest']), '--output', str(score_path),
                       '--judge', plan['judge']['name'], '--judge-max-len', str(plan['judge']['max_len']),
                       '--device', 'cuda']
            for label, path, _ in commands:
                command += ['--input', f'{label}={path}']
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        # The figure compares runs, so a batch that only adds one generator to a finished comparison has none yet.
        if len(commands) > 1:
            figure = output / 'comparison.png'
            subprocess.run([plan['cdr_python'], str(ROOT / 'benchmark/plot_comparison.py'),
                            '--scores', str(score_path), '--output', str(figure)], cwd=ROOT, env=env, check=True)
        status(output)


def label_path(value):
    label, separator, path = value.partition('=')
    if not (label and separator and path):
        raise argparse.ArgumentTypeError('expected LABEL=PATH')
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'smoke', 'run', 'resume', 'status'])
    parser.add_argument('batch', nargs='?', type=Path)
    parser.add_argument('--recipe', type=Path, default=ROOT / 'benchmark/recipes/oas4_h3_length_free.json')
    parser.add_argument('--stage', choices=['smoke', 'full'], default='full', help='Plan stage only')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cdr-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--external-python', type=label_path, action='append', default=[], metavar='LABEL=PATH',
                        help="Interpreter for one external generator; default is its adapter's conda environment")
    parser.add_argument('--iglm-python', type=Path, help="Shorthand for --external-python iglm=PATH")
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error('--gpu must be one GPU index, such as 0')
    if args.command in ['resume', 'status']:
        if args.batch is None:
            parser.error('resume/status requires a batch directory')
        output = args.batch.resolve()
        if args.command == 'status':
            status(output)
        else:
            plan = read(output / 'plan.json')
            validate_frozen_plan(plan)
            execute(plan, output, args.gpu, resume=True)
        return
    if args.batch is not None:
        parser.error('Use --output for a new batch')
    stage = args.stage if args.command == 'plan' else 'smoke' if args.command == 'smoke' else 'full'
    external_pythons = dict(args.external_python)
    if args.iglm_python:
        external_pythons['iglm'] = args.iglm_python
    plan = build_plan(args.recipe.resolve(), stage, args.cdr_python.resolve(), external_pythons)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        arm_labels = [a['label'] for a in plan['arms']] + [e['label'] for e in plan['externals']]
        print(f"{plan['name']} {stage}: {len(arm_labels)} arms, {plan['frameworks']} frameworks, "
              f"{plan['sampling']['num_samples']} draws = {plan['requested_attempts']:,} requested attempts")
        print('Arms: ' + ', '.join(arm_labels))
        print('Generation: heavy-chain flanks; inferred MLM length or natural EOS. Judge: heavy chain only.')
        print('Source: ' + plan['execution_source']['sha256'])
        print('Primary endpoint: ' + plan['endpoint'])
    if args.command == 'plan':
        return
    stamp = datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S_%f')
    output = args.output.resolve() if args.output else ROOT / f"results/benchmark/{stamp}_{plan['name']}_{stage}"
    execute(plan, output, args.gpu)


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
