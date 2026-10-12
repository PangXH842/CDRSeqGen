#!/usr/bin/env python3
"""Flank-keyed retrieval reference: draw training H3 loops that share a framework's J/V flank key.

The unconditional empirical sampler ignores the framework. This reference is the cheapest conditioning there is: it
reads only the flank strings `pt1` and `pt2`, looks up the training loops whose own flanks end the same way, and draws
from that group with replacement. It has no parameters, no sampling controls and no length input, so it shows how much
of a learned generator's advantage over unconditional sampling the flanks already carry. A framework whose key never
occurs in training falls back to the unconditional pool, and the record says so.

Keys (the first is the primary, fixed before any of these were scored; the others are sensitivity checks):
  fr3tail12+fr4head6  last 12 residues of FR3 and first 6 of FR4, the V/J proxy the README documents for its length lookup
  fr4                 the whole of FR4, a J-gene proxy
  fr3tail20+fr4       last 20 residues of FR3 and the whole of FR4, a finer V/J proxy

Only the training split is searched, so no validation or test loop can be returned. benchmark/blind_check.py tests that
the output does not depend on the hidden native loop.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Direct execution puts benchmark/adapters first on sys.path; the repository root is what the imports below need.
ADAPTER_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == ADAPTER_DIR:
    sys.path.pop(0)
    sys.path.insert(0, str(ADAPTER_DIR.parents[1]))

from benchmark._common import (
    ADAPTER_SCHEMA_VERSION as SCHEMA_VERSION,
    AMINO_ACIDS,
    assert_resume_compatible,
    atomic_write_json,
    execution_source_identity,
    sha256_file,
    stable_example_seed,
    validate_manifest_record,
)

KEYS = {
    'fr3tail12+fr4head6': ('last 12 residues of FR3 and first 6 of FR4', lambda pt1, pt2: (pt1[-12:], pt2[:6])),
    'fr4': ('the whole of FR4', lambda pt1, pt2: pt2),
    'fr3tail20+fr4': ('last 20 residues of FR3 and the whole of FR4', lambda pt1, pt2: (pt1[-20:], pt2)),
}


def flank_key(name: str, pt1: str, pt2: str):
    return KEYS[name][1](pt1, pt2)


def training_groups(corpus: list[dict[str, Any]], key: str):
    """(groups, pool): training H3 loops grouped by flank key, and all of them, each as (corpus index, loop).

    pt1 is FR1+CDR1+FR2+CDR2+FR3 and pt2 is FR4, exactly as the manifest builds them. Validation and test records are
    never indexed; an invalid training loop stops the run.
    """
    groups: dict[Any, list[tuple[int, str]]] = defaultdict(list)
    pool: list[tuple[int, str]] = []
    for index, entry in enumerate(corpus):
        if entry.get('split') not in {'train', 'validation', 'test'}:
            raise ValueError(f'corpus record {index} lacks an explicit supported split')
        if entry['split'] != 'train':
            continue
        heavy = entry['heavy']
        loop = heavy['cdr3']
        if not loop or set(loop) - AMINO_ACIDS:
            raise ValueError(f'invalid training H3 at corpus index {index}')
        pt1 = heavy['fwr1'] + heavy['cdr1'] + heavy['fwr2'] + heavy['cdr2'] + heavy['fwr3']
        groups[flank_key(key, pt1, heavy['fwr4'])].append((index, loop))
        pool.append((index, loop))
    if not pool:
        raise ValueError('no training H3 loops available')
    return groups, pool


def run_example(record: dict[str, Any], groups, pool, *, key: str, num_samples: int, base_seed: int) -> dict[str, Any]:
    """Draw `num_samples` loops for one framework. Reads example_id, pt1 and pt2 only; native fields are echoed."""
    validate_manifest_record(record)
    started = time.perf_counter()
    group = groups.get(flank_key(key, record['pt1'], record['pt2']))
    candidates = group if group else pool
    rng = random.Random(stable_example_seed(base_seed, record['example_id']))
    draws = [candidates[rng.randrange(len(candidates))] for _ in range(num_samples)]
    samples = [loop for _, loop in draws]
    matches = sum(len(loop) == record['native_length'] for loop in samples)
    return {
        'example_id': record['example_id'],
        'native': record['native'],
        'native_length': record['native_length'],
        'length_policy': {'mode': 'inferred', 'source': 'training_group_lengths', 'minimum': None, 'maximum': None},
        'samples': samples,
        'attempts': num_samples,
        'valid_attempts': num_samples,
        'invalid_attempts': 0,
        'length_matches': matches,
        'length_mismatches': num_samples - matches,
        'fixed_length_acceptance_rate': matches / num_samples,
        'complete': True,
        'pool_size': len(candidates),
        'pool': 'flank_group' if group else 'unconditional_fallback',
        'raw_generations': [{'attempt': i + 1, 'training_source_index': index, 'loop': loop, 'length': len(loop),
                             'valid_generation': True, 'length_match': len(loop) == record['native_length'],
                             'full_sequence': record['pt1'] + loop + record['pt2']}
                            for i, (index, loop) in enumerate(draws)],
        'failures': [],
        'seed': stable_example_seed(base_seed, record['example_id']),
        'runtime_seconds': time.perf_counter() - started,
    }


def build_payload(args, manifest: dict[str, Any], manifest_hash: str, corpus_hash: str, groups, pool) -> dict[str, Any]:
    sizes = sorted(len(group) for group in groups.values())
    return {
        'schema_version': SCHEMA_VERSION,
        'adapter': 'retrieval',
        'benchmark_id': manifest['benchmark_id'],
        'manifest': {'path': str(args.manifest.resolve()), 'sha256': manifest_hash,
                     'schema_version': manifest.get('schema_version')},
        'model': {
            'name': 'Flank-keyed training-H3 retrieval', 'neural': False, 'parameters_network': 0,
            'key': args.key, 'key_definition': KEYS[args.key][0],
            'policy': 'uniform draws with replacement from training loops sharing the flank key; unconditional fallback',
            'conditioning': 'flank strings pt1 and pt2 only', 'corpus': str(args.corpus.resolve()), 'corpus_sha256': corpus_hash,
            'training_records': len(pool), 'groups': len(groups), 'median_group_size': statistics.median(sizes),
            'largest_group_size': sizes[-1], 'execution_source': execution_source_identity(),
        },
        'sampling': {
            'track': (manifest.get('task') or {}).get('track'), 'num_samples': args.num_samples,
            # The matched neural budget, recorded for assert_comparable; retrieval has no logits.
            'temperature': args.temperature, 'top_p': args.top_p, 'base_seed': args.seed,
            'sampling_controls_applied': False,
            'note': 'Temperature/top-p describe the matched neural budget; retrieval draws uniformly from its group',
            'record_limit': args.limit, 'duplicates_preserved': True, 'raw_generations_preserved': True,
        },
        'started_at': datetime.now(timezone.utc).isoformat(),
        'finished_at': None,
        'records': [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True, help='Processed corpus the manifest was built from')
    parser.add_argument('--key', choices=sorted(KEYS), default='fr3tail12+fr4head6')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=1.0, help='Matched budget; recorded, not applied')
    parser.add_argument('--top-p', type=float, default=0.9, help='Matched budget; recorded, not applied')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu', help='Accepted for the runner; unused')
    parser.add_argument('--limit', type=int, default=None, help='Run only the first N manifest records')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError('--num-samples must be positive')
    manifest_hash = sha256_file(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if (manifest.get('task') or {}).get('track') != 'length_free':
        raise ValueError('retrieval runs only on length_free manifests')
    records = manifest.get('records') or []
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit must be positive')
        records = records[:args.limit]
    if not records:
        raise ValueError('manifest contains no selected records')
    corpus_hash = sha256_file(args.corpus)
    if corpus_hash != (manifest.get('source') or {}).get('processed_file_sha256'):
        raise ValueError('corpus differs from the one the manifest was built from; its split would not be the same')

    groups, pool = training_groups(json.loads(args.corpus.read_text()), args.key)
    expected = build_payload(args, manifest, manifest_hash, corpus_hash, groups, pool)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f'output already exists; pass --resume to continue: {args.output}')
        payload = json.loads(args.output.read_text())
        assert_resume_compatible(payload, expected)
    else:
        payload = expected

    completed = {record['example_id'] for record in payload['records']}
    for position, record in enumerate(records, start=1):
        if record['example_id'] in completed:
            print(f"[{position}/{len(records)}] skip {record['example_id']} (already complete)")
            continue
        result = run_example(record, groups, pool, key=args.key, num_samples=args.num_samples, base_seed=args.seed)
        payload['records'].append(result)
        atomic_write_json(args.output, payload)
        print(f"[{position}/{len(records)}] {record['example_id']} pool={result['pool']}({result['pool_size']}) "
              f"matches={result['length_matches']}", flush=True)

    payload['finished_at'] = datetime.now(timezone.utc).isoformat()
    payload['summary'] = {
        'records_requested': len(records), 'records_completed': len(payload['records']),
        'records_with_full_sample_count': sum(r['complete'] for r in payload['records']),
        'total_attempts': sum(r['attempts'] for r in payload['records']),
        'total_valid_attempts': sum(r['valid_attempts'] for r in payload['records']),
        'total_invalid_attempts': 0,
        'total_length_matches': sum(r['length_matches'] for r in payload['records']),
        'total_length_mismatches': sum(r['length_mismatches'] for r in payload['records']),
        'frameworks_using_fallback': sum(r['pool'] == 'unconditional_fallback' for r in payload['records']),
        'total_runtime_seconds': sum(r['runtime_seconds'] for r in payload['records']),
    }
    atomic_write_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output} "
          f"({payload['summary']['frameworks_using_fallback']} used the unconditional fallback)")


if __name__ == '__main__':
    main()
