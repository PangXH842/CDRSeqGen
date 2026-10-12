"""External generators the batch runner can launch beside the trained CDRSeqGen checkpoints.

An external generator is a frozen model that lives in its own environment, so the runner starts it as a
subprocess and its only contact with the rest of the benchmark is the adapter-output JSON it writes
(adapters/README.md). Registering it here is all run.py needs to plan, fingerprint and launch it: the
runner names no specific model. The recipe says which externals a batch uses; the plan records what was
actually launched, so a saved batch never depends on this registry's defaults.

Recipes and plans written before this registry named one external, IgLM, in dedicated `iglm` /
`iglm_python` keys. Both readers below still accept that layout, because frozen recipes are hashed into
saved plans and cannot be rewritten.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]

# `empirical` is the training-only reference pool written by the follow-up, not a launched generator.
RESERVED_LABELS = frozenset({'empirical'})
# Distributions fingerprinted in each environment; the runner refuses to resume if their versions move.
DEFAULT_PACKAGES = ('torch', 'transformers', 'safetensors')


@dataclass(frozen=True)
class External:
    script: str                    # adapter path relative to the repository root
    conda_env: str                 # environment whose interpreter is the default
    arguments: Callable            # (output, sampling, options) -> adapter-specific CLI arguments
    packages: tuple = DEFAULT_PACKAGES
    # (record, sampling, options) -> None; raise ValueError when this adapter cannot serve the record.
    check_record: Callable = lambda record, sampling, options: None
    # options -> {path: pinned sha256 or None}: inputs the plan freezes, and refuses to plan if one is not its pin.
    # Paths are relative to the repository root (or absolute), like the checkpoint paths of local arms.
    inputs: Callable = lambda options: {}


def _iglm_check_record(record, sampling, options):
    # IgLM has two conditioning tokens, a gap marker and a separator.
    needed = len(record['pt1']) + len(record['pt2']) + 4 + sampling['max_new_tokens']
    if needed > options['max_length']:
        raise ValueError(f"{record['example_id']} needs {needed} IgLM tokens, above max_length {options['max_length']}")


def _iglm_arguments(output, sampling, options):
    return ['--output', str(output), '--model-name', options['model_name'], '--length-policy', 'natural',
            '--max-attempts', str(sampling['num_samples']), '--max-errors', str(sampling['num_samples']),
            '--max-length', str(options['max_length']), '--max-new-tokens', str(sampling['max_new_tokens'])]


@lru_cache(maxsize=None)
def _read_json(path):
    return json.loads((ROOT / path).read_text())


@lru_cache(maxsize=None)
def _igcraft_capacities(config):
    from benchmark.adapters.igcraft import config_capacities
    return config_capacities(ROOT / config)


def _igcraft_check_record(record, sampling, options):
    # Local import: the adapter module is light, but only IgCraft recipes need it.
    from benchmark.adapters.igcraft import capacity_problems
    from benchmark.export_regions import flank_regions
    table = _read_json(options['flank_regions'])['records']
    if record['example_id'] not in table:
        raise ValueError(f"{record['example_id']} is absent from the flank-region table")
    problems = capacity_problems(flank_regions(record, table[record['example_id']]), _igcraft_capacities(options['config']))
    if problems:
        raise ValueError(f"{record['example_id']} does not fit IgCraft: {'; '.join(problems)}")


def _igcraft_inputs(options):
    pins = options.get('sha256', {})
    return {options[name]: pins.get(name) for name in ('flank_regions', 'config', 'checkpoint')}


def _igcraft_arguments(output, sampling, options):
    arguments = ['--output', str(output), '--checkpoint', options['checkpoint'], '--config', options['config'],
                 '--flank-regions', options['flank_regions'], '--num-steps', str(options['num_steps']),
                 '--num-particles', str(options['num_particles']), '--igcraft-temperature', str(options['temperature']),
                 '--batch-size', str(options['batch_size']), '--max-errors', str(sampling['num_samples'])]
    if options.get('sha256', {}).get('checkpoint'):
        arguments += ['--checkpoint-sha256', options['sha256']['checkpoint']]
    if options.get('hf_revision'):
        arguments += ['--hf-revision', options['hf_revision']]
    return arguments + ([] if options.get('compile', True) else ['--no-compile'])


def _retrieval_check_record(record, sampling, options):
    from benchmark.adapters.retrieval import KEYS
    if options['key'] not in KEYS:
        raise ValueError(f"unknown retrieval key {options['key']!r}; known: {', '.join(sorted(KEYS))}")
    if not record['pt1'] or not record['pt2']:
        raise ValueError(f"{record['example_id']} has an empty flank")


def _retrieval_inputs(options):
    return {options['corpus']: options.get('sha256')}


def _retrieval_arguments(output, sampling, options):
    return ['--output', str(output), '--corpus', options['corpus'], '--key', options['key']]


EXTERNALS = {
    'iglm': External(script='benchmark/adapters/iglm.py', conda_env='iglm',
                     arguments=_iglm_arguments, check_record=_iglm_check_record),
    'igcraft': External(script='benchmark/adapters/igcraft.py', conda_env='igcraft', arguments=_igcraft_arguments,
                        packages=('torch', 'pytorch-lightning', 'numpy', 'anarci', 'igcraft'),
                        check_record=_igcraft_check_record, inputs=_igcraft_inputs),
    # Not a generator: a parameter-free reference that runs in the main environment, listed here so it is planned,
    # pinned, scored and merged exactly like the models it is compared with.
    'retrieval': External(script='benchmark/adapters/retrieval.py', conda_env='cdrprj', arguments=_retrieval_arguments,
                          packages=(), check_record=_retrieval_check_record, inputs=_retrieval_inputs),
}


def default_python(adapter):
    """Interpreter of the adapter's conventional conda environment, used when none is passed."""
    return Path.home() / 'miniconda3/envs' / EXTERNALS[adapter].conda_env / 'bin/python'


def recipe_externals(recipe):
    """The externals a recipe requests as [{label, adapter, options}], validated."""
    if 'externals' in recipe and 'iglm' in recipe:
        raise ValueError("Recipe has both 'externals' and the legacy 'iglm' key; declare IgLM once")
    if 'externals' in recipe:
        entries = [dict(entry, options=dict(entry.get('options', {}))) for entry in recipe['externals']]
    elif 'iglm' in recipe:
        entries = [dict(label='iglm', adapter='iglm', options=dict(recipe['iglm']))]
    else:
        entries = []
    labels = [entry['label'] for entry in entries]
    if len(labels) != len(set(labels)) or RESERVED_LABELS.intersection(labels):
        raise ValueError(f'External labels must be unique and not reserved ({sorted(RESERVED_LABELS)}): {labels}')
    for entry in entries:
        if entry['adapter'] not in EXTERNALS:
            raise ValueError(f"Unknown external adapter {entry['adapter']!r}; registered: {', '.join(EXTERNALS)}")
    return entries


def plan_externals(plan):
    """The externals a saved plan launched, whether or not it predates the registry."""
    if 'externals' in plan:
        return plan['externals']
    return [dict(label='iglm', adapter='iglm', python=plan['iglm_python'], options=plan['iglm'])]
