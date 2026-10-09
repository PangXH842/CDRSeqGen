"""Merge, paired analysis and capacity for generators added to a finished comparison, on synthetic batches."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmark._common import sha256_file
from benchmark.extend import build_plan, execute, summary, verify_complete

IDS = ['a', 'b', 'c', 'd']
SAMPLES = 2


def metrics(edit, valid=SAMPLES):
    return dict(n_samples=valid, judge_candidates_scored=valid, edit_similarity_mean=edit, judge_perplexity_mean=3.0,
                length_absolute_error_mean=2.0, length_match_rate=0.1, core_aar_mean=0.2, diversity_hamming=0.5)


def run(label, edit, valid=SAMPLES, path='unused'):
    return dict(label=label, input=dict(path=path, sha256='unused'),
                sampling=dict(track='length_free', num_samples=SAMPLES, base_seed=42),
                records=[dict(example_id=i, native='AAA', native_length=3, metrics=metrics(edit + k / 100, valid))
                         for k, i in enumerate(IDS)])


def scores(*runs):
    return dict(manifest=dict(sha256='manifest-hash'), judge=dict(name='judge', model_commit='v1'),
                scoring=dict(aggregation_unit='example'), runs=list(runs))


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return path


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = write(self.root / 'manifest.json', dict(records=[
            dict(example_id=i, source_index=k) for k, i in enumerate(IDS)]))
        original = self.root / 'original'
        write(original / 'plan.json', dict(arms=[dict(label='mlm')], iglm_python='p', iglm={}))
        self.base = self.root / 'base'
        write(self.base / 'plan.json', dict(manifest=str(self.manifest), source_batch=str(original),
                                            analysis=dict(iterations=1000, seed=42)))
        write(self.base / 'scores/comparison.json', scores(run('mlm', .30), run('iglm', .35), run('empirical', .20)))
        write(self.base / 'clusters.json', dict(cluster_ids=[0, 1, 2, 3]))
        write(self.base / 'capacity.json', dict(rows=[dict(arm='mlm'), dict(arm='iglm'), dict(arm='empirical')], limitations=['x']))
        write(self.base / 'complete.json', dict(artifacts={n: sha256_file(self.base / n) for n in
                                                           ['scores/comparison.json', 'capacity.json', 'clusters.json']}))
        self.added = self.root / 'added'
        self.generations = write(self.added / 'generations/igcraft.json', dict(
            finished_at='done', model=dict(parameters_network=300, parameters_by_module=dict(backbone=250, structure_encoder=50))))
        self.write_added(run('igcraft', .40))

    def write_added(self, added_run, **plan_changes):
        added_run = dict(added_run, input=dict(path=str(self.generations), sha256=sha256_file(self.generations)))
        plan = dict(arms=[], stage='full', externals=[dict(label='igcraft', adapter='igcraft')])
        write(self.added / 'plan.json', {**plan, **plan_changes})
        write(self.added / 'scores/comparison.json', scores(added_run))

    def extend(self, **options):
        plan = build_plan(self.base, self.added, **options)
        output = self.root / 'extension'
        execute(plan, output)
        return plan, output

    def test_added_generator_joins_a_wider_family_and_the_capacity_table(self):
        plan, output = self.extend()
        analysis = json.loads((output / 'analysis.json').read_text())
        self.assertEqual(plan['externals'], ['iglm', 'igcraft'])
        # mlm vs each external (2), then mlm / iglm / igcraft vs the empirical pool (3)
        self.assertEqual(analysis['primary_comparison_family'], 5)
        self.assertIn(['mlm', 'igcraft'], analysis['pairs'])
        self.assertIn(['igcraft', 'empirical'], analysis['pairs'])
        self.assertNotIn(['iglm', 'igcraft'], analysis['pairs'])
        merged = json.loads((output / 'scores/comparison.json').read_text())
        self.assertEqual([r['label'] for r in merged['runs']], ['mlm', 'iglm', 'empirical', 'igcraft'])
        capacity = json.loads((output / 'capacity.json').read_text())
        self.assertEqual([r['arm'] for r in capacity['rows']], ['mlm', 'iglm', 'empirical', 'igcraft'])
        self.assertEqual(capacity['rows'][-1]['system_parameters'], 300)
        self.assertEqual(capacity['rows'][-1]['parameters_by_module']['structure_encoder'], 50)
        self.assertTrue((output / 'analysis.png').exists())

    def test_completed_extension_reports_its_hashes_and_base_is_untouched(self):
        before = {p: sha256_file(p) for p in self.base.rglob('*.json')}
        _, output = self.extend()
        self.assertEqual(before, {p: sha256_file(p) for p in self.base.rglob('*.json')})
        complete = json.loads((output / 'complete.json').read_text())
        self.assertEqual(complete['added'], ['igcraft'])
        summary(output)
        (output / 'analysis.md').write_text('tampered')
        self.assertNotEqual(sha256_file(output / 'analysis.md'), complete['artifacts']['analysis.md'])
        summary(output)  # reports the mismatch instead of passing silently

    def test_changed_base_artifact_is_refused(self):
        (self.base / 'capacity.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Base artifact changed'):
            verify_complete(self.base)
        with self.assertRaisesRegex(ValueError, 'Base artifact changed'):
            build_plan(self.base, self.added)

    def test_added_batch_must_be_externals_only_and_match_its_scores(self):
        self.write_added(run('igcraft', .40), arms=[dict(label='mlm_s42')])
        with self.assertRaisesRegex(ValueError, 'external generators only'):
            build_plan(self.base, self.added)
        self.write_added(run('other', .40))
        with self.assertRaisesRegex(ValueError, 'do not match the externals'):
            build_plan(self.base, self.added)

    def test_scored_generations_must_be_the_finished_file(self):
        self.generations.write_text(json.dumps(dict(finished_at=None, model={})))
        self.write_added(run('igcraft', .40))
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            build_plan(self.base, self.added)

    def test_changed_input_after_planning_is_refused(self):
        plan = build_plan(self.base, self.added)
        (self.base / 'clusters.json').write_text(json.dumps(dict(cluster_ids=[9, 9, 9, 9])))
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            execute(plan, self.root / 'late')

    def test_short_candidate_coverage_needs_an_explicit_waiver(self):
        self.write_added(run('igcraft', .40, valid=SAMPLES - 1))
        with self.assertRaisesRegex(ValueError, 'Missing candidates'):
            self.extend()
        plan = build_plan(self.base, self.added, allow_incomplete=True)
        execute(plan, self.root / 'waived')
        merged = json.loads((self.root / 'waived/scores/comparison.json').read_text())
        self.assertEqual(merged['runs'][-1]['records'][0]['metrics']['n_samples'], SAMPLES - 1)


if __name__ == '__main__':
    unittest.main()
