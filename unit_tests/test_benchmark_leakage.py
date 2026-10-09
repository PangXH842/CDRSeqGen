"""Synthetic CPU tests for exact neighbours, cohort decisions and safe resume."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from benchmark._common import sha256_file
from benchmark.empirical import representative_clusters, verify_splits
from benchmark import leakage


def record(index, loop, split='train'):
    chain = dict(fwr1='ACD', cdr1='EFG', fwr2='HIK', cdr2='LMN',
                 fwr3='PQR', cdr3=loop, fwr4='STV')
    light = {**chain, 'cdr3': 'WYAC'}
    return dict(sequence_id=str(index), source=dict(run=1), split=split, heavy=chain, light=light)


def normalized(raw):
    return leakage.normalize_corpus(raw, dict(sources=[dict(run=1, subject='donor')]))


def fixture(directory):
    """Ten disjoint synthetic H3 groups; no project corpus or models involved."""
    raw = [record(i, amino * 6) for i, amino in enumerate('ACDEFGHIKL')]
    labels = list(range(len(raw))); shuffled = labels.copy(); random.Random(42).shuffle(shuffled)
    for i in shuffled[:2]: raw[i]['split'] = 'test'
    for i in shuffled[2:4]: raw[i]['split'] = 'validation'
    settings = dict(test_fraction=.2, validation_fraction=.2, seed=42, cluster_identity=.85)
    corpus_path = directory / 'corpus.json'; corpus_path.write_text(json.dumps(raw))
    digest = sha256_file(corpus_path)
    metadata = dict(records=len(raw), processed_file_sha256=digest, preprocessing=settings,
                    sources=[dict(run=1, subject='donor')])
    metadata_path = directory / 'metadata.json'; metadata_path.write_text(json.dumps(metadata))
    groups = dict(dataset_sha256=digest, settings=settings, cluster_ids=labels,
        verification=verify_splits(raw, labels, .2, .2, 42),
        source=dict(files={'benchmark/empirical.py': leakage.source_identity()['benchmark/empirical.py']}))
    clusters_path = directory / 'clusters.json'; clusters_path.write_text(json.dumps(groups))
    rows = []
    for r in normalized(raw):
        if r['split'] == 'validation':
            rows.append(dict(source_index=r['source_index'], example_id=r['example_id'],
                native=r['h3'], pt1=r['heavy_flanks'][0], pt2=r['heavy_flanks'][1]))
    manifest = dict(selection=dict(split='validation'), source=dict(processed_file_sha256=digest), records=rows)
    manifest_path = directory / 'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    recipe = dict(name='fixture', corpus=str(corpus_path), corpus_sha256=digest,
        metadata=str(metadata_path), clusters=str(clusters_path), manifest=str(manifest_path),
        expected_hashes={name: sha256_file(path) for name, path in
            [('metadata', metadata_path), ('clusters', clusters_path), ('manifest', manifest_path)]},
        identity=.85, expected_splits=dict(train=6, validation=2, test=2), benchmark_frameworks=2,
        chunk_size=1, query_tile=1, reference_tile=2, smoke_queries_per_split=1)
    recipe_path = directory / 'recipe.json'; recipe_path.write_text(json.dumps(recipe))
    return recipe_path


class NeighbourTests(unittest.TestCase):
    def test_cutoff_inclusive_and_decimal_rounding(self):
        self.assertEqual(leakage.required_matches(20, .85), 17)
        self.assertEqual(leakage.required_matches(100, .07), 7)
        for invalid in [0, -1, 1.1]:
            with self.assertRaises(ValueError): leakage.required_matches(20, invalid)
        records = normalized([record(0, 'C' * 3 + 'A' * 17), record(1, 'C' * 4 + 'A' * 16),
                              record(2, 'A' * 20, 'test')])
        result = leakage.nearest_h3([records[2]], leakage.reference_index(records[:2]))[2]
        self.assertEqual(result['neighbours_at_cutoff'], 1)
        self.assertEqual(result['closest_matches'], 17)

    def test_nearest_below_cutoff_and_tie_source_order_across_tiles(self):
        records = normalized([record(0, 'AAAC'), record(1, 'AACA'), record(2, 'CCCC'), record(3, 'AAAA', 'test')])
        result = leakage.nearest_h3([records[3]], leakage.reference_index(list(reversed(records[:3]))),
                                   reference_tile=1)[3]
        self.assertEqual(result['closest_source_index'], 0)
        self.assertEqual(result['closest_identity'], .75)
        self.assertEqual(result['neighbours_at_cutoff'], 0)

    def test_no_same_length_reference_and_duplicate_counts(self):
        records = normalized([record(0, 'AAAA'), record(1, 'AAAA'), record(2, 'AAAA', 'test'),
                              record(3, 'AAAAA', 'test')])
        result = leakage.nearest_h3(records[2:], leakage.reference_index(records[:2]))
        self.assertEqual(result[2]['exact_h3_references'], 2)
        self.assertEqual(result[2]['neighbours_at_cutoff'], 2)
        self.assertIsNone(result[3]['closest_identity'])
        self.assertEqual(result[3]['same_length_references'], 0)

    def test_matches_independent_bruteforce_for_many_lengths_tiles_and_thresholds(self):
        rng = random.Random(33)
        refs = [record(i, ''.join(rng.choice('ACDE') for _ in range(rng.choice([4, 7, 20])))) for i in range(31)]
        queries = [record(i + 31, ''.join(rng.choice('ACDE') for _ in range(rng.choice([4, 7, 20, 49]))), 'test')
                   for i in range(15)]
        records = normalized(refs + queries); references = records[:31]; queries = records[31:]
        for cutoff in [.5, .85, 1.]:
            for tile in [1, 3, 4096]:
                actual = leakage.nearest_h3(queries, leakage.reference_index(references), cutoff, 2, tile)
                for query in queries:
                    same = [r for r in references if len(r['h3']) == len(query['h3'])]
                    counts = [(sum(a == b for a, b in zip(query['h3'], r['h3'])), r['source_index']) for r in same]
                    closest = sorted(counts, key=lambda x: (-x[0], x[1]))[0] if counts else (None, None)
                    row = actual[query['source_index']]
                    self.assertEqual(row['closest_matches'], closest[0])
                    self.assertEqual(row['closest_source_index'], closest[1])
                    self.assertEqual(row['neighbours_at_cutoff'], sum(c / len(query['h3']) >= cutoff for c, _ in counts))

    def test_representative_disjointness_does_not_imply_pairwise_separation(self):
        loops = ['A' * 20, 'C' * 3 + 'A' * 17, 'C' * 4 + 'A' * 16]
        labels = representative_clusters(loops)
        self.assertEqual(labels, [0, 0, 1])
        records = normalized([record(i, s, 'test' if i == 2 else 'train') for i, s in enumerate(loops)])
        result = leakage.nearest_h3([records[2]], leakage.reference_index(records[:2]))[2]
        self.assertEqual(result['closest_identity'], .95)
        self.assertEqual(result['closest_source_index'], 1)
        self.assertEqual(result['neighbours_at_cutoff'], 1)

    def test_no_silent_sequence_normalization_or_provenance_guess(self):
        for loop in ['aaac', 'AA X', '', 'AAAX']:
            with self.assertRaises(ValueError): normalized([record(0, loop)])
        with self.assertRaises(ValueError):
            leakage.normalize_corpus([record(0, 'AAAA')], dict(sources=[]))

    def test_empty_frameworks_allowed_and_non_h3_ambiguity_is_literal_and_flagged(self):
        raw = record(0, 'AAAA'); raw['heavy']['fwr4'] = ''; raw['light']['cdr1'] = 'AXA'
        result = normalized([raw])[0]
        self.assertEqual(result['heavy_flanks'][1], '')
        self.assertEqual(result['l1'], 'AXA')
        self.assertEqual(result['noncanonical_regions'], ['light.cdr1'])


class CohortTests(unittest.TestCase):
    def test_validation_exposure_excludes_test_but_shared_flanks_do_not(self):
        records = normalized([record(0, 'CCCC', 'train'), record(1, 'AAAA', 'validation'),
                              record(2, 'AAAA', 'test'), record(3, 'GGGG', 'test')])
        labels = [0, 1, 2, 3]
        comparisons = {}
        for query, reference in leakage.COMPARISONS:
            comparisons[f'{query}_to_{reference}'] = leakage.compare_chunk(
                [r for r in records if r['split'] == query],
                leakage.reference_index([r for r in records if r['split'] == reference]),
                records, labels, .85, 2, 1)
        summary = leakage.summarize(comparisons, {1}, .85, 'full')
        self.assertEqual(summary['candidate_test']['eligible_source_indices'], [3])
        self.assertEqual(summary['candidate_test']['by']['donor']['donor'],
                         dict(audited_records=2, eligible_records=1, excluded_records=1))
        self.assertFalse(summary['candidate_test']['is_final_manifest'])
        self.assertEqual(summary['comparisons']['test_to_train']['all']['with_exact_overlap']['heavy_flanks'], 2)
        self.assertEqual(summary['comparisons']['test_to_validation']['all']['with_exact_overlap']['h3'], 1)
        self.assertEqual(summary['benchmark_validation']['audited_records'], 1)
        self.assertIn('No final manifest', leakage.report(summary))


class RunnerTests(unittest.TestCase):
    def test_plan_does_not_normalize_search_or_run_models(self):
        with tempfile.TemporaryDirectory() as name:
            recipe = fixture(Path(name))
            with patch.object(leakage, 'nearest_h3', side_effect=AssertionError('must not search')), \
                 patch.object(leakage, 'normalize_corpus', side_effect=AssertionError('must not parse')):
                plan = leakage.build_plan(recipe, 'full')
                smoke = leakage.build_plan(recipe, 'smoke')
            self.assertEqual([c['queries'] for c in plan['comparisons']], [2, 2, 2])
            self.assertEqual([c['queries'] for c in smoke['comparisons']], [1, 1, 1])
            self.assertEqual([c['references'] for c in smoke['comparisons']], [6, 6, 2])

    def test_changed_pinned_groups_fail_at_plan(self):
        with tempfile.TemporaryDirectory() as name:
            recipe = fixture(Path(name))
            groups_path = Path(leakage.read(recipe)['clusters'])
            groups = leakage.read(groups_path); groups['cluster_ids'].reverse()
            groups_path.write_text(json.dumps(groups))
            with self.assertRaisesRegex(ValueError, 'pinned recipe: clusters'):
                leakage.build_plan(recipe, 'full')

    def test_end_to_end_fixture_and_completed_resume_no_search(self):
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            directory = Path(name); plan = leakage.build_plan(fixture(directory), 'full'); output = directory / 'out'
            leakage.execute(plan, output)
            summary = leakage.read(output / 'summary.json')
            self.assertEqual(summary['benchmark_validation']['audited_records'], 2)
            self.assertEqual(summary['candidate_test']['audited_records'], 2)
            with patch.object(leakage, 'compare_chunk', side_effect=AssertionError('must not search')):
                leakage.execute(plan, output, resume=True)
                leakage.status(output)
            with self.assertRaises(FileExistsError): leakage.execute(plan, output)

    def test_interruption_resume_reuses_sealed_chunks(self):
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            directory = Path(name); plan = leakage.build_plan(fixture(directory), 'full'); output = directory / 'out'
            original = leakage.compare_chunk; calls = []
            def interrupt(*args, **kwargs):
                calls.append(args[0][0]['source_index'])
                if len(calls) == 2: raise KeyboardInterrupt()
                return original(*args, **kwargs)
            with patch.object(leakage, 'compare_chunk', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt): leakage.execute(plan, output)
            progress = leakage.read(output / 'progress.json')
            first = 'comparisons/validation_to_train/00000.json'; digest = progress[first]
            with patch.object(leakage, 'compare_chunk', wraps=original) as spy:
                leakage.execute(plan, output, resume=True)
            self.assertEqual(spy.call_count, 5)
            self.assertEqual(sha256_file(output / first), digest)

    def test_changed_sealed_output_fails_before_search(self):
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            directory = Path(name); plan = leakage.build_plan(fixture(directory), 'full'); output = directory / 'out'
            leakage.execute(plan, output)
            (output / 'comparisons/test_to_train/00000.json').write_text('{}')
            with patch.object(leakage, 'compare_chunk', side_effect=AssertionError('must not search')):
                with self.assertRaisesRegex(ValueError, 'Sealed audit artifact changed'):
                    leakage.execute(plan, output, resume=True)
                with self.assertRaises(ValueError): leakage.status(output)

    def test_changed_input_or_source_prevents_partial_resume(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name); plan = leakage.build_plan(fixture(directory), 'full')
            changed = copy.deepcopy(plan); changed['source']['benchmark/leakage.py'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'source/environment changed'): leakage.validate_plan(changed)
            Path(plan['corpus']).write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'): leakage.validate_plan(plan)

    def test_incomplete_completion_marker_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            directory = Path(name); plan = leakage.build_plan(fixture(directory), 'full'); output = directory / 'out'
            leakage.execute(plan, output)
            complete = leakage.read(output / 'complete.json'); complete['artifacts'].pop('summary.json')
            (output / 'complete.json').write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, 'Completion marker'): leakage.status(output)


if __name__ == '__main__':
    unittest.main()
