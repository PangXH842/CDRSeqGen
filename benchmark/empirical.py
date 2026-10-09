"""Training-only repertoire sampling and verified reconstruction of H3 split groups.

The baseline ignores framework/native information when drawing loops. Clustering
recovers the original ordered representative algorithm; it never changes splits.
"""
from collections import defaultdict
from datetime import datetime, timezone
import math
import random
import time

import numpy as np

from benchmark._common import ADAPTER_SCHEMA_VERSION, AMINO_ACIDS, stable_example_seed


def training_pool(corpus):
    """Return one H3 per training record, retaining its empirical frequency.

    Example: training loops [AAA, AAA, CCC] give AAA probability 2/3.
    Validation/test loops never enter the pool; invalid training H3s fail.
    """
    pool = []
    for index, record in enumerate(corpus):
        if record.get('split') not in {'train', 'validation', 'test'}:
            raise ValueError(f'Corpus record {index} lacks an explicit supported split')
        if record['split'] != 'train':
            continue
        loop = record['heavy']['cdr3']
        if not loop or set(loop) - AMINO_ACIDS:
            raise ValueError(f'Invalid training H3 at corpus index {index}')
        pool.append((index, loop))
    if not pool:
        raise ValueError('No training H3 loops available')
    return pool


def draw_training_loops(pool, example_id, count, seed):
    """Draw with replacement using only the training pool and stable example ID.

    Native sequence, native length and flanks are absent from this interface.
    Returned source indices make every draw auditable.
    """
    if count <= 0:
        raise ValueError('count must be positive')
    rng = random.Random(stable_example_seed(seed, example_id))
    return [pool[rng.randrange(len(pool))] for _ in range(count)]


def generate(manifest, corpus, count, seed, manifest_identity, source_identity):
    """Build a scorer-compatible baseline payload; native truth is bookkeeping."""
    pool = training_pool(corpus)
    started = datetime.now(timezone.utc).isoformat()
    records = []
    for item in manifest['records']:
        tick = time.perf_counter()
        draws = draw_training_loops(pool, item['example_id'], count, seed)
        samples = [loop for _, loop in draws]
        matches = sum(len(loop) == item['native_length'] for loop in samples)
        records.append(dict(example_id=item['example_id'], native=item['native'], native_length=item['native_length'],
            samples=samples, attempts=count, valid_attempts=count, invalid_attempts=0,
            length_matches=matches, length_mismatches=count-matches, complete=True,
            runtime_seconds=time.perf_counter()-tick,
            raw_generations=[dict(attempt=i+1, training_source_index=index, loop=loop,
                                  length=len(loop), valid_generation=True,
                                  full_sequence=item['pt1']+loop+item['pt2']) for i,(index,loop) in enumerate(draws)]))
    return dict(schema_version=ADAPTER_SCHEMA_VERSION, adapter='empirical', benchmark_id=manifest['benchmark_id'],
        manifest=manifest_identity,
        model=dict(name='Training-H3 empirical sampler', policy='uniform training records, with replacement',
                   conditioning='none', training_records=len(pool), execution_source=source_identity),
        sampling=dict(track='length_free', length_policy='empirical_training_distribution', num_samples=count,
                      temperature=1.0, top_p=0.9, base_seed=seed, duplicates_preserved=True,
                      sampling_controls_applied=False, note='Temperature/top-p describe the matched neural budget; empirical sampling has no logits'),
        started_at=started, finished_at=datetime.now(timezone.utc).isoformat(), records=records,
        summary=dict(records_requested=len(records), records_completed=len(records),
                     total_attempts=len(records)*count, total_valid_attempts=len(records)*count,
                     total_invalid_attempts=0))


def representative_clusters(sequences, identity=.85, progress=None):
    """Return exactly the original greedy same-length H3 cluster IDs, faster.

    Index disjoint blocks to shortlist representatives. If at most e residues
    differ, at least one of e+1 disjoint blocks matches exactly. Verify full
    positional identity and choose the first matching representative, preserving
    original ordering. This is not an all-pairs or clonal-family guarantee.
    """
    if not 0 < identity <= 1:
        raise ValueError('identity must be in (0, 1]')
    buckets = defaultdict(list)
    for i, sequence in enumerate(sequences):
        if not sequence or set(sequence) - AMINO_ACIDS:
            raise ValueError(f'Invalid H3 at corpus index {i}')
        buckets[len(sequence)].append(i)
    labels = [-1] * len(sequences)
    next_id = 0
    for length, indices in buckets.items():
        required = math.ceil(identity*length)
        blocks = length-required+1
        # Disjoint internal positions improve filtering when terminal anchors repeat.
        positions = np.arange(2, length-2) if length-4 >= blocks else np.arange(length)
        partitions = [part.tolist() for part in np.array_split(positions, blocks)]
        inverted = defaultdict(list)
        representatives = np.empty((len(indices), length), dtype=np.uint8)
        n_representatives = 0
        first_id = next_id
        for offset, index in enumerate(indices):
            sequence = sequences[index]
            encoded = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
            keys = [(block, ''.join(sequence[p] for p in part)) for block,part in enumerate(partitions)]
            candidates = sorted({r for key in keys for r in inverted.get(key, ())})
            selected = None
            if candidates:
                counts = (representatives[candidates] == encoded).sum(axis=1)
                eligible = np.flatnonzero(counts >= required)
                if len(eligible):
                    selected = candidates[int(eligible[0])]
            if selected is None:
                selected = n_representatives
                representatives[selected] = encoded
                n_representatives += 1
                next_id += 1
                for key in keys:
                    inverted[key].append(selected)
            labels[index] = first_id + selected
            if progress and offset and offset % 10000 == 0:
                progress(f'H3 length {length}: {offset}/{len(indices)} records')
        if progress:
            progress(f'H3 length {length}: {len(indices)} records, {n_representatives} groups')
    return labels


def verify_splits(corpus, labels, test_fraction, validation_fraction, seed):
    """Reproduce cluster allocation and require every existing split to match."""
    if test_fraction<0 or validation_fraction<0 or test_fraction+validation_fraction>=1:
        raise ValueError('Split fractions must be nonnegative and sum to less than one')
    if len(corpus) != len(labels):
        raise ValueError('Cluster labels do not cover the corpus')
    groups = defaultdict(list)
    for i,label in enumerate(labels):
        groups[label].append(i)
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    expected = ['train']*len(corpus)
    cursor = 0
    for split,fraction in [('test',test_fraction),('validation',validation_fraction)]:
        assigned = 0
        while cursor < len(shuffled) and assigned < round(fraction*len(corpus)):
            for i in groups[shuffled[cursor]]:
                expected[i] = split
                assigned += 1
            cursor += 1
    mismatches = [i for i,r in enumerate(corpus) if r.get('split') != expected[i]]
    if mismatches:
        raise ValueError(f'Reconstructed clusters do not reproduce {len(mismatches)} saved split labels; first indices: {mismatches[:5]}')
    return dict(records=len(corpus), groups=len(groups), split_labels_verified=len(corpus))
