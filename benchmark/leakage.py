"""Audit local split overlap before freezing the final H3 benchmark.

Workflow: freeze inputs -> verify saved groups -> search every same-length H3
pair across splits -> summarize overlap and possible holdout eligibility.
Commands: plan, smoke, run, status, resume. Only smoke/run/resume do the audit;
none load a model, change a split, or launch generation/training.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import fcntl
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark._common import AMINO_ACIDS, atomic_write_json, sha256_file
from benchmark.empirical import verify_splits

COMPARISONS = [('validation', 'train'), ('test', 'train'), ('test', 'validation')]
REGIONS = ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4')
EXACT_FIELDS = ('h1', 'h2', 'h3', 'l1', 'l2', 'l3', 'heavy', 'pair', 'heavy_flanks')
SOURCE_FILES = ('benchmark/leakage.py', 'benchmark/empirical.py', 'benchmark/_common.py',
                'cdrseqgen/artifacts.py')


def read(path):
    return json.loads(Path(path).read_text())


def source_identity():
    """Hash the implementation used here without depending on unrelated methods."""
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__)


def required_matches(length, identity):
    """Return the inclusive identity cutoff as an exact integer match count.

    Example: length=20, identity=0.85 -> at least 17 identical positions.
    Decimal arithmetic avoids an accidental extra match from float rounding.
    """
    threshold = Fraction(str(identity))
    if not 0 < threshold <= 1 or length <= 0:
        raise ValueError('Require a positive length and identity in (0, 1]')
    return (length * threshold.numerator + threshold.denominator - 1) // threshold.denominator


def normalize_corpus(corpus, metadata):
    """Keep stable corpus indices, donor provenance and exact comparison strings.

    heavy_flanks = (FWR1+H1+FWR2+H2+FWR3, FWR4), with H3 removed.
    H3 must contain canonical amino acids. Other regions retain their literal
    characters for exact-string diagnostics, with ambiguous regions flagged.
    Empty frameworks are allowed by preprocessing. No strings are normalized.
    """
    donors = {str(s['run']): s['subject'] for s in metadata['sources']}
    records = []
    for index, record in enumerate(corpus):
        split = record.get('split')
        if split not in {'train', 'validation', 'test'}:
            raise ValueError(f'Unsupported split at source index {index}')
        heavy, light = record['heavy'], record['light']
        noncanonical = []
        for chain_name, chain in [('heavy', heavy), ('light', light)]:
            for region in REGIONS:
                sequence = chain[region]
                if not isinstance(sequence, str) or (region.startswith('cdr') and not sequence):
                    raise ValueError(f'Invalid {region} at source index {index}')
                if set(sequence) - AMINO_ACIDS:
                    noncanonical.append(f'{chain_name}.{region}')
        if 'heavy.cdr3' in noncanonical:
            raise ValueError(f'Invalid H3 amino acids at source index {index}')
        run = str(record['source']['run'])
        if run not in donors:
            raise ValueError(f'No donor provenance for run {run}')
        full_heavy = ''.join(heavy[r] for r in REGIONS)
        full_light = ''.join(light[r] for r in REGIONS)
        records.append(dict(source_index=index, split=split, donor=donors[run], noncanonical_regions=noncanonical,
            example_id=f"oas:{run}:{record['sequence_id']}",
            h1=heavy['cdr1'], h2=heavy['cdr2'], h3=heavy['cdr3'],
            l1=light['cdr1'], l2=light['cdr2'], l3=light['cdr3'],
            heavy=full_heavy, pair=(full_heavy, full_light),
            heavy_flanks=(''.join(heavy[r] for r in REGIONS[:5]), heavy['fwr4'])))
    if len({r['example_id'] for r in records}) != len(records):
        raise ValueError('Duplicate stable example IDs in corpus')
    return records


def reference_index(records):
    """Index every reference record; duplicates retain their record frequency."""
    exact = {field: {} for field in EXACT_FIELDS}
    buckets = defaultdict(list)
    for record in sorted(records, key=lambda r: r['source_index']):
        buckets[len(record['h3'])].append(record)
        for field in EXACT_FIELDS:
            key = record[field]
            count, first = exact[field].get(key, (0, record['source_index']))
            exact[field][key] = (count + 1, first)
    encoded = {}
    for length, members in buckets.items():
        encoded[length] = (np.array([r['source_index'] for r in members], dtype=np.int64),
            np.frombuffer(''.join(r['h3'] for r in members).encode('ascii'), dtype=np.uint8).reshape(-1, length))
    return dict(exact=exact, encoded=encoded)


def nearest_h3(queries, references, identity=.85, query_tile=16, reference_tile=4096):
    """Search all same-length reference H3s, including matches below the cutoff.

    Example: query=AAAA, refs=[AAAC, AACC], cutoff=0.85
      closest identity: 3/4 = 0.75; neighbours at cutoff: 0.

    Ties select the smallest corpus source index. Different lengths are excluded;
    no same-length reference gives a None closest identity and zero neighbours.
    Tiling bounds temporary memory; it does not approximate the search.
    """
    if query_tile <= 0 or reference_tile <= 0:
        raise ValueError('Tile sizes must be positive')
    results = {}
    buckets = defaultdict(list)
    for query in queries:
        buckets[len(query['h3'])].append(query)
    for length, members in buckets.items():
        cutoff = required_matches(length, identity)
        indexed = references['encoded'].get(length)
        if indexed is None:
            for query in members:
                results[query['source_index']] = dict(same_length_references=0,
                    closest_source_index=None, closest_matches=None, closest_identity=None,
                    neighbours_at_cutoff=0, exact_h3_references=0)
            continue
        indices, encoded = indexed
        for start in range(0, len(members), query_tile):
            batch = members[start:start + query_tile]
            queries_encoded = np.frombuffer(''.join(r['h3'] for r in batch).encode('ascii'),
                dtype=np.uint8).reshape(-1, length)
            best = np.full(len(batch), -1, dtype=np.int64)
            closest = np.full(len(batch), -1, dtype=np.int64)
            neighbours = np.zeros(len(batch), dtype=np.int64)
            exact = np.zeros(len(batch), dtype=np.int64)
            for offset in range(0, len(indices), reference_tile):
                counts = (queries_encoded[:, None, :] ==
                          encoded[None, offset:offset + reference_tile, :]).sum(axis=2)
                local = counts.argmax(axis=1)
                matches = counts[np.arange(len(batch)), local]
                improved = matches > best
                closest[improved] = indices[offset + local[improved]]
                best[improved] = matches[improved]
                neighbours += (counts >= cutoff).sum(axis=1)
                exact += (counts == length).sum(axis=1)
            for i, query in enumerate(batch):
                results[query['source_index']] = dict(same_length_references=len(indices),
                    closest_source_index=int(closest[i]), closest_matches=int(best[i]),
                    closest_identity=float(best[i] / length), neighbours_at_cutoff=int(neighbours[i]),
                    exact_h3_references=int(exact[i]))
    return results


def compare_chunk(queries, references, records, labels, identity, query_tile, reference_tile):
    """Attach exact overlaps and closest-neighbour provenance to each query."""
    nearest = nearest_h3(queries, references, identity, query_tile, reference_tile)
    rows = []
    for query in queries:
        index = query['source_index']
        result = nearest[index]
        neighbour = result['closest_source_index']
        exact = {field: dict(zip(('count', 'first_source_index'),
            references['exact'][field].get(query[field], (0, None)))) for field in EXACT_FIELDS}
        if result['exact_h3_references'] != exact['h3']['count']:
            raise ValueError('Exact H3 index and exhaustive search disagree')
        rows.append(dict(source_index=index, example_id=query['example_id'], donor=query['donor'],
            h3_length=len(query['h3']), cluster_id=labels[index], **result,
            closest_example_id=records[neighbour]['example_id'] if neighbour is not None else None,
            closest_donor=records[neighbour]['donor'] if neighbour is not None else None,
            closest_cluster_id=labels[neighbour] if neighbour is not None else None, exact=exact))
    return rows


def overlap_summary(rows):
    """Count affected query records, keeping the total denominator explicit."""
    identities = [r['closest_identity'] for r in rows if r['closest_identity'] is not None]
    return dict(records=len(rows), with_same_length_reference=len(identities),
        with_h3_neighbour_at_cutoff=sum(r['neighbours_at_cutoff'] > 0 for r in rows),
        cross_split_pairs_at_cutoff=sum(r['neighbours_at_cutoff'] for r in rows),
        with_exact_overlap={f: sum(r['exact'][f]['count'] > 0 for r in rows) for f in EXACT_FIELDS},
        closest_identity_mean=float(np.mean(identities)) if identities else None,
        closest_identity_quantiles={str(q): float(np.quantile(identities, q)) for q in [.0, .5, .9, .95, 1.]}
            if identities else {})


def summarize(comparisons, benchmark_indices, identity, stage):
    """Describe residual overlap and candidate test eligibility, without resplitting.

    A candidate test record has no same-length H3 neighbour at the cutoff in
    either train or validation. This criterion says nothing about indel-aware
    homology, other loops, clonal families, or external pretraining exposure.
    """
    summaries = {}
    for name, rows in comparisons.items():
        groups = {}
        for field in ['donor', 'h3_length']:
            grouped = defaultdict(list)
            for row in rows:
                grouped[str(row[field])].append(row)
            groups[field] = {key: overlap_summary(value) for key, value in sorted(grouped.items())}
        summaries[name] = dict(all=overlap_summary(rows), by=groups)
    validation = comparisons['validation_to_train']
    benchmark = [r for r in validation if r['source_index'] in benchmark_indices]
    test_train = {r['source_index']: r for r in comparisons['test_to_train']}
    test_validation = {r['source_index']: r for r in comparisons['test_to_validation']}
    if test_train.keys() != test_validation.keys():
        raise ValueError('Test comparison populations differ')
    eligible = sorted(i for i, row in test_train.items() if row['neighbours_at_cutoff'] == 0
                      and test_validation[i]['neighbours_at_cutoff'] == 0)
    excluded = sorted(set(test_train) - set(eligible))
    eligible_set = set(eligible)
    retention = {}
    for field in ['donor', 'h3_length']:
        groups = defaultdict(list)
        for row in test_train.values():
            groups[str(row[field])].append(row['source_index'])
        retention[field] = {key: dict(audited_records=len(indices),
            eligible_records=sum(i in eligible_set for i in indices),
            excluded_records=sum(i not in eligible_set for i in indices)) for key, indices in sorted(groups.items())}
    limits = ['Different lengths and gapped/indel alignments are excluded.',
              'Shared germline/framework context is not automatically target leakage.',
              'Exact non-H3 overlap is diagnostic; this does not certify their splits.',
              'External IgLM/IgBert training membership remains unverified.']
    if stage == 'smoke':
        limits.append('Smoke covers only the first query records; it cannot support a split decision.')
    return dict(schema_version='cdrseqgen.split_overlap_summary.v1', stage=stage, identity=identity,
        comparisons=summaries, benchmark_validation=dict(expected_records=len(benchmark_indices),
            audited_records=len(benchmark), summary=overlap_summary(benchmark)),
        candidate_test=dict(audited_records=len(test_train), eligible_records=len(eligible),
            excluded_records=len(excluded), eligible_source_indices=eligible, excluded_source_indices=excluded,
            by=retention,
            rule='No same-length positional H3 identity >= cutoff in train OR validation; no model outcomes used',
            is_final_manifest=False),
        limits=limits)


def report(summary):
    """Render the counts needed to decide the final local H3 cohort."""
    lines = ['# Local split overlap audit', '', f"Stage: **{summary['stage']}**. "
        f"Inclusive same-length H3 positional identity cutoff: **{summary['identity']:.0%}**.", '',
        '| Query → reference | Queries | H3 neighbour at cutoff | Exact H3 | Exact heavy | Exact pair | Same heavy flanks |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, entry in summary['comparisons'].items():
        s = entry['all']; e = s['with_exact_overlap']
        lines.append(f"| {name} | {s['records']} | {s['with_h3_neighbour_at_cutoff']} | "
            f"{e['h3']} | {e['heavy']} | {e['pair']} | {e['heavy_flanks']} |")
    b = summary['benchmark_validation']; c = summary['candidate_test']
    lines += ['', f"Existing benchmark: {b['audited_records']}/{b['expected_records']} frameworks audited; "
        f"{b['summary']['with_h3_neighbour_at_cutoff']} have a training H3 neighbour at the cutoff.", '',
        f"Candidate test eligibility: {c['eligible_records']}/{c['audited_records']} pass the stated H3 rule; "
        f"{c['excluded_records']} are excluded. No final manifest has been selected.", '',
        'Donor/length denominators, exact non-H3 overlaps and candidate indices are in `summary.json`.', '',
        *['- ' + limit for limit in summary['limits']], '',
        'Use the full audit to choose a common cohort and reporting scope before any final model evaluation. '
        'This report never changes the processed dataset, trained models or earlier validation results.']
    return '\n'.join(lines) + '\n'


def build_plan(recipe_path, stage):
    """Freeze input hashes/settings; read metadata rather than search the corpus."""
    recipe_path = Path(recipe_path).resolve()
    recipe = read(recipe_path)
    if stage not in {'smoke', 'full'}:
        raise ValueError('Unsupported audit stage')
    identity = recipe['identity']; required_matches(20, identity)
    for field in ['chunk_size', 'query_tile', 'reference_tile', 'smoke_queries_per_split']:
        value = recipe[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{field} must be a positive integer')
    files = {str(recipe_path): sha256_file(recipe_path)}
    paths = {name: str((ROOT / recipe[name]).resolve()) for name in ['corpus', 'metadata', 'clusters', 'manifest']}
    for path in paths.values():
        files[path] = sha256_file(path)
    for name, digest in recipe['expected_hashes'].items():
        if files[paths[name]] != digest:
            raise ValueError(f'Input differs from pinned recipe: {name}')
    if set(recipe['expected_hashes']) != {'metadata', 'clusters', 'manifest'}:
        raise ValueError('Recipe must pin metadata, groups and manifest hashes')
    metadata = read(paths['metadata']); clusters = read(paths['clusters']); manifest = read(paths['manifest'])
    digest = files[paths['corpus']]
    if digest != recipe['corpus_sha256'] or metadata['processed_file_sha256'] != digest:
        raise ValueError('Processed corpus differs from pinned metadata/recipe')
    if clusters['dataset_sha256'] != digest or manifest['source']['processed_file_sha256'] != digest:
        raise ValueError('Groups/manifest belong to another corpus')
    if clusters['settings'] != metadata['preprocessing'] or identity != metadata['preprocessing']['cluster_identity']:
        raise ValueError('Audit cutoff or saved groups differ from preprocessing settings')
    if clusters['source']['files']['benchmark/empirical.py'] != source_identity()['benchmark/empirical.py']:
        raise ValueError('Saved group reconstruction source changed')
    counts = recipe['expected_splits']
    if set(counts) != {'train', 'validation', 'test'} or sum(counts.values()) != metadata['records']:
        raise ValueError('Expected split counts do not cover the corpus')
    verified = clusters['verification']
    if len(clusters['cluster_ids']) != metadata['records'] or verified['split_labels_verified'] != metadata['records']:
        raise ValueError('Saved groups were not verified for the complete corpus')
    if manifest['selection']['split'] != 'validation' or len(manifest['records']) != recipe['benchmark_frameworks']:
        raise ValueError('Unexpected benchmark cohort')
    query_counts = {s: min(counts[s], recipe['smoke_queries_per_split']) if stage == 'smoke' else counts[s]
                    for s in ['validation', 'test']}
    comparisons = [dict(name=f'{q}_to_{r}', query_split=q, reference_split=r,
        queries=query_counts[q], references=counts[r],
        chunks=(query_counts[q] + recipe['chunk_size'] - 1) // recipe['chunk_size']) for q, r in COMPARISONS]
    return dict(schema_version='cdrseqgen.split_overlap_plan.v1', name=recipe['name'], stage=stage,
        recipe=str(recipe_path), **paths, identity=identity, expected_splits=counts,
        benchmark_frameworks=recipe['benchmark_frameworks'], comparisons=comparisons,
        chunk_size=recipe['chunk_size'], query_tile=recipe['query_tile'], reference_tile=recipe['reference_tile'],
        source=source_identity(), environment=environment(), frozen_files=files,
        execution='CPU only; exact all-reference same-length H3 search; no models or split changes')


def validate_plan(plan):
    """Refuse a resumed batch with changed sources, environment or inputs."""
    if plan['source'] != source_identity() or plan['environment'] != environment():
        raise ValueError('Audit source/environment changed; preserve this batch and prepare a fresh output')
    for path, digest in plan['frozen_files'].items():
        if sha256_file(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')


def expected_chunks(plan):
    return {f"comparisons/{c['name']}/{i:05d}.json" for c in plan['comparisons'] for i in range(c['chunks'])}


def validate_progress(output, plan, progress):
    """Validate sealed chunks before consuming or extending them."""
    allowed = expected_chunks(plan) | {'verification.json', 'summary.json', 'summary.md'}
    for name, digest in progress.items():
        if name not in allowed or not (output / name).is_file() or sha256_file(output / name) != digest:
            raise ValueError(f'Sealed audit artifact changed or unexpected: {name}')


def complete_valid(output, plan, progress):
    completion = output / 'complete.json'
    if not completion.exists():
        return False
    required = expected_chunks(plan) | {'verification.json', 'summary.json', 'summary.md'}
    marker = read(completion)
    if set(progress) != required or marker.get('artifacts') != progress or marker.get('plan_sha256') != sha256_file(output / 'plan.json'):
        raise ValueError('Completion marker does not cover the frozen plan and all expected artifacts')
    return True


def status(output):
    """Verify saved seals and show progress without running the audit."""
    plan = read(output / 'plan.json')
    progress = read(output / 'progress.json') if (output / 'progress.json').exists() else {}
    validate_progress(output, plan, progress)
    for comparison in plan['comparisons']:
        sealed = sum(f"comparisons/{comparison['name']}/{i:05d}.json" in progress for i in range(comparison['chunks']))
        print(f"{comparison['name']}: {sealed}/{comparison['chunks']} chunks sealed "
              f"({comparison['queries']} queries against {comparison['references']} references)")
    print('Complete: ' + ('yes; artifact hashes verified' if complete_valid(output, plan, progress) else 'no'))
    # Old completed outputs remain inspectable after preparing later source versions.
    print('Current implementation matches saved plan: ' + str(plan['source'] == source_identity()))


def execute(plan, output, resume=False):
    """Run the frozen CPU audit and seal each independently resumable chunk.

    Interrupted, unsealed chunks are recomputed. Sealed chunks are hash-checked
    before any search and reused. Completed runs only need status inspection.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume:
            atomic_write_json(output / 'plan.json', plan)
        elif read(output / 'plan.json') != plan:
            raise ValueError('Supplied plan differs from saved plan')
        progress_path = output / 'progress.json'
        progress = read(progress_path) if progress_path.exists() else {}
        validate_progress(output, plan, progress)
        if complete_valid(output, plan, progress):
            print('Already complete; saved hashes verified. No audit rerun.', flush=True)
            return
        validate_plan(plan)

        def seal(name):
            progress[name] = sha256_file(output / name)
            atomic_write_json(progress_path, progress)

        print('Read frozen corpus and verify existing split allocation; no model execution.', flush=True)
        corpus = read(plan['corpus']); metadata = read(plan['metadata']); clusters = read(plan['clusters'])
        settings = metadata['preprocessing']; labels = clusters['cluster_ids']
        counts = Counter(r['split'] for r in corpus)
        if dict(counts) != plan['expected_splits']:
            raise ValueError('Corpus split counts differ from frozen plan')
        verification = verify_splits(corpus, labels, settings['test_fraction'], settings['validation_fraction'], settings['seed'])
        if verification != clusters['verification']:
            raise ValueError('Saved group verification disagrees with current corpus')
        records = normalize_corpus(corpus, metadata)
        del corpus
        split_records = {s: [r for r in records if r['split'] == s] for s in counts}
        manifest = read(plan['manifest']); benchmark_indices = set()
        for item in manifest['records']:
            index = item['source_index']; r = records[index]
            if r['split'] != 'validation' or r['example_id'] != item['example_id'] or r['h3'] != item['native'] or r['heavy_flanks'] != (item['pt1'], item['pt2']):
                raise ValueError('Benchmark source index/sequence/context does not match corpus')
            benchmark_indices.add(index)
        if len(benchmark_indices) != plan['benchmark_frameworks']:
            raise ValueError('Repeated benchmark source indices')
        atomic_write_json(output / 'verification.json', dict(**verification, split_records=dict(counts),
            benchmark_frameworks_verified=len(benchmark_indices), saved_groups_reused=True,
            noncanonical_region_records=dict(Counter(region for r in records for region in r['noncanonical_regions']))))
        seal('verification.json')
        references = {}
        comparisons = {}
        for comparison in plan['comparisons']:
            name = comparison['name']; ref_split = comparison['reference_split']
            queries = split_records[comparison['query_split']][:comparison['queries']]
            rows = []; tick = time.perf_counter()
            for chunk in range(comparison['chunks']):
                filename = f'comparisons/{name}/{chunk:05d}.json'
                start = chunk * plan['chunk_size']; batch = queries[start:start + plan['chunk_size']]
                if filename not in progress:
                    if ref_split not in references:
                        references[ref_split] = reference_index(split_records[ref_split])
                    result = compare_chunk(batch, references[ref_split], records, labels, plan['identity'],
                                           plan['query_tile'], plan['reference_tile'])
                    atomic_write_json(output / filename, dict(comparison=name, rows=result))
                    seal(filename)
                payload = read(output / filename)
                if payload['comparison'] != name or [r['source_index'] for r in payload['rows']] != [r['source_index'] for r in batch]:
                    raise ValueError('Saved chunk population differs from planned queries')
                rows.extend(payload['rows'])
                print(f"{name}: {min(start + len(batch), len(queries))}/{len(queries)} queries "
                      f"({time.perf_counter() - tick:.1f}s this invocation)", flush=True)
            comparisons[name] = rows
        summary = summarize(comparisons, benchmark_indices, plan['identity'], plan['stage'])
        atomic_write_json(output / 'summary.json', summary); seal('summary.json')
        # An interrupted report write is unsealed and safe to replace on resume.
        (output / 'summary.md').write_text(report(summary)); seal('summary.md')
        atomic_write_json(output / 'complete.json', dict(finished_at=datetime.now(timezone.utc).isoformat(),
            plan_sha256=sha256_file(output / 'plan.json'), artifacts=progress))
        print(report(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'smoke', 'run', 'status', 'resume'])
    parser.add_argument('output_path', nargs='?', help='Existing audit directory for status/resume')
    parser.add_argument('--recipe', type=Path, default=ROOT / 'benchmark/recipes/oas4_h3_leakage.json')
    parser.add_argument('--stage', choices=['full', 'smoke'], default='full', help='Stage to inspect with plan')
    parser.add_argument('--output', type=Path, help='New output directory for smoke/run')
    args = parser.parse_args(argv)
    if args.command in {'status', 'resume'}:
        if not args.output_path or args.output:
            parser.error('status/resume require an existing positional output directory')
        output = Path(args.output_path).resolve()
        if args.command == 'status':
            status(output)
        else:
            execute(read(output / 'plan.json'), output, resume=True)
    else:
        if args.output_path:
            parser.error('Use --output for a new smoke/run directory')
        stage = args.stage if args.command == 'plan' else ('smoke' if args.command == 'smoke' else 'full')
        plan = build_plan(args.recipe, stage)
        if args.command == 'plan':
            print(json.dumps(plan, indent=2))
        else:
            if not args.output:
                parser.error('smoke/run require --output to avoid accidental output reuse')
            execute(plan, args.output.resolve())
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'Audit failed: {exc}') from exc
