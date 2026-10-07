"""Plan/run the inference-only comparison; scripts/run.sh benchmark exposes this file.

  plan   -- validate inputs and show the budget without model execution
  smoke  -- five frameworks, five draws, training seed 42 plus IgLM
  run    -- 150 frameworks, twenty draws, both training seeds plus IgLM
  status -- inspect saved outputs
  resume -- reuse a frozen batch plan and compatible adapter outputs

Scientific choices live in benchmark/recipes; trained checkpoints are read only.
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


def read(path):
    return json.loads(Path(path).read_text())


def environment_identity(python):
    """Read interpreter/package versions; do not import models or inspect GPUs."""
    script = "import sys,json,importlib.metadata as m; print(json.dumps(dict(python=sys.version,packages={n:m.version(n) for n in ['torch','transformers','safetensors']})))"
    return json.loads(subprocess.check_output([str(python), '-c', script], text=True))


def build_plan(recipe_path, stage, cdr_python, iglm_python):
    recipe = read(recipe_path)
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
        # IgLM has two conditioning tokens, a gap marker and a separator.
        assert len(record['pt1']) + len(record['pt2']) + 4 + recipe['sampling']['max_new_tokens'] <= recipe['iglm']['max_length']
    source = ROOT / 'data/processed' / manifest['source']['processed_file']
    assert sha256_file(source) == manifest['source']['processed_file_sha256']
    frozen_files = {str(manifest_path.relative_to(ROOT)): digest,
                    str(recipe_path.relative_to(ROOT)): sha256_file(recipe_path)}
    arms = []
    for arm in recipe['arms']:
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
    return dict(schema_version='cdrseqgen.benchmark_batch.v1', name=recipe['name'], stage=stage,
                manifest=str(manifest_path.relative_to(ROOT)), frameworks=len(records),
                arms=arms, sampling={**recipe['sampling'], 'num_samples': samples},
                iglm=recipe['iglm'], judge=recipe['judge'], endpoint=recipe['endpoint'], limits=recipe['limits'],
                requested_attempts=len(records) * samples * (len(arms) + 1),
                cdr_python=str(cdr_python), iglm_python=str(iglm_python),
                environments={str(p): environment_identity(p) for p in [cdr_python, iglm_python]},
                execution_source=execution_source_identity(), frozen_files=frozen_files)


def validate_frozen_plan(plan):
    if plan['execution_source'] != execution_source_identity():
        raise ValueError('Benchmark source changed; preserve this batch and plan a new source-versioned run')
    for path, digest in plan['frozen_files'].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f'Frozen benchmark input changed: {path}')
    for python, expected in plan['environments'].items():
        if environment_identity(python) != expected:
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
    path = output / 'generations/iglm.json'
    commands.append(('iglm', path, [plan['iglm_python'], str(ROOT / 'benchmark/adapters/iglm.py'),
        *common, '--output', str(path), '--model-name', plan['iglm']['model_name'],
        '--length-policy', 'natural', '--max-attempts', str(s['num_samples']),
        '--max-errors', str(s['num_samples']), '--max-length', str(plan['iglm']['max_length']),
        '--max-new-tokens', str(s['max_new_tokens'])]))
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
        figure = output / 'comparison.png'
        subprocess.run([plan['cdr_python'], str(ROOT / 'benchmark/plot_comparison.py'),
                        '--scores', str(score_path), '--output', str(figure)], cwd=ROOT, env=env, check=True)
        status(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'smoke', 'run', 'resume', 'status'])
    parser.add_argument('batch', nargs='?', type=Path)
    parser.add_argument('--recipe', type=Path, default=ROOT / 'benchmark/recipes/oas4_h3_length_free.json')
    parser.add_argument('--stage', choices=['smoke', 'full'], default='full', help='Plan stage only')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cdr-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--iglm-python', type=Path, default=Path.home() / 'miniconda3/envs/iglm/bin/python')
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
    plan = build_plan(args.recipe.resolve(), stage, args.cdr_python.resolve(), args.iglm_python.resolve())
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"{plan['name']} {stage}: {len(plan['arms']) + 1} arms, {plan['frameworks']} frameworks, "
              f"{plan['sampling']['num_samples']} draws = {plan['requested_attempts']:,} requested attempts")
        print('Arms: ' + ', '.join([a['label'] for a in plan['arms']] + ['iglm']))
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
