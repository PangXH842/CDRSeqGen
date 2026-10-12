"""Seed variability, determinism, capacity and sensitivity analysis on a synthetic world with real scoring.

Candidates are generated per arm, scored by the real scorer with a stand-in judge, sealed into batches shaped like the
real ones, and given tiny checkpoints, so the module's verification, numbers and refusals are all exercised.
"""
import contextlib
import io
import json
from pathlib import Path
import random
import statistics
import tempfile
import unittest
import zlib

import numpy as np
from safetensors.numpy import save_file

from benchmark import variability as V
from benchmark._common import sha256_file
from benchmark.analysis import crossed_interval
from benchmark.extend import EXTENSION
from benchmark.scorer import score_payload

AA = 'ACDEFGHIKLMNPQRSTVWY'
SAMPLES = 6
FRAMEWORKS = [(f'f{i}', ''.join(random.Random(i).choice(AA) for _ in range(9 + i))) for i in range(6)]
METHODS = V.METHODS


class Judge:
    def score(self, record, samples):
        return [3.0] * len(samples)


def mutate(native, rng, rate):
    out = [c if rng.random() > rate else rng.choice(AA) for c in native]
    if rng.random() < .3:
        out = out[:-1]
    if rng.random() < .2:
        out.append(rng.choice(AA))
    return ''.join(out)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return path


class World:
    """Everything the analysis reads, built under one temporary directory."""

    def __init__(self, root, references=False):
        self.root = Path(root)
        records = [dict(example_id=i, source_index=k, native=n, native_length=len(n), pt1='QQQ', pt2='WWW', parent_sequence='QQQ' + n + 'WWW',
                        infill_range=[3, 3 + len(n)]) for k, (i, n) in enumerate(FRAMEWORKS)]
        self.manifest = write(self.root / 'manifest.json', dict(benchmark_id='b', schema_version='m', task=dict(track='length_free', anchor_trim_each_end=2),
                                                                source={}, records=records))
        self.clusters = write(self.root / 'clusters.json', dict(cluster_ids=[0, 0, 1, 1, 2, 2]))
        self.by_id = {r['example_id']: r for r in records}
        self.digest = sha256_file(self.manifest)
        self.runs, self.arms, self.samples = {}, {}, {}
        # Original six, IgLM and the empirical pool, IgCraft: one generation batch each, as in the real project.
        for method in METHODS:
            for seed in V.ORIGINAL_SEEDS:
                self.arm('original', f'{method}_s{seed}', .30 if method == 'mlm' else .35, checkpoint=(method, 'S', seed))
        self.arm('baselines', 'iglm', .33)
        self.arm('baselines', 'empirical', .7)
        self.arm('igcraft_batch', 'igcraft', .45)
        for method in METHODS:
            for seed in V.NEW_SEEDS:
                self.arm('seeds', f'{method}_s{seed}', .30 if method == 'mlm' else .35)
            for k in (1, 2):
                self.arm('repeats', f'{method}_s42_r{k}', .30 if method == 'mlm' else .35, checkpoint=(method, f'r{k}', 42),
                         copy_of=f'{method}_s42' if method == 'ar' else None)
            for size in V.SIZES:
                for seed in V.SCAN_SEEDS:
                    self.arm('capacity', f'{method}_{size}_s{seed}', .30 if method == 'mlm' else .35, checkpoint=(method, size, seed))
        self.arm('t1', 'igcraft_t1', .6, interior_gaps=True)
        if references:
            for label, rate in (('retrieval', .5), ('retrieval_j', .55), ('retrieval_fine', .45)):
                self.arm('references', label, rate)
        self.build_main()
        for batch in ('seeds', 'repeats', 'capacity', 't1', 'references'):
            if batch in {a for a, _ in self.arms.values()}:
                self.batch(batch)

    def arm(self, batch, label, rate, checkpoint=None, copy_of=None, interior_gaps=False):
        path = self.root / batch / 'generations' / f'{label}.json'
        rng = random.Random(label)
        records = []
        for example_id, native in FRAMEWORKS:
            samples = ([s for s in self.samples[copy_of][example_id]] if copy_of else [mutate(native, rng, rate) for _ in range(SAMPLES)])
            raw = [dict(attempt=k + 1, loop=s, length=len(s), valid_generation=True, interior_gap=interior_gaps and k == 0) for k, s in enumerate(samples)]
            matches = sum(len(s) == len(native) for s in samples)
            records.append(dict(example_id=example_id, native=native, native_length=len(native), samples=samples, attempts=SAMPLES,
                                valid_attempts=SAMPLES, invalid_attempts=0, length_matches=matches, length_mismatches=SAMPLES - matches,
                                runtime_seconds=0.1, complete=True, raw_generations=raw))
        write(path, dict(schema_version='cdrseqgen.adapter_output.v1', adapter='x', benchmark_id='b', manifest=dict(sha256=self.digest),
                         sampling=dict(track='length_free', num_samples=SAMPLES), model={}, finished_at='done', records=records))
        with contextlib.redirect_stdout(io.StringIO()):  # the scorer reports progress per framework
            run = score_payload(label, path, json.loads(path.read_text()), self.by_id, 2, Judge())
        self.runs[label] = run
        self.samples[label] = {r['example_id']: r['samples'] for r in records}
        self.arms[label] = (batch, checkpoint)

    def checkpoint(self, label):
        """A tiny checkpoint whose size encodes the model size and whose values encode the run."""
        batch, key = self.arms[label]
        method, tag, seed = key
        sized = dict(S=40, M=120, L=300, XL=700)
        n = sized.get(tag, 40)
        directory = self.root / 'runs' / label
        base = np.random.default_rng(zlib.crc32(f'{method}:{seed}'.encode())).standard_normal(n).astype(np.float32)
        identical = method == 'ar'
        shift = 0 if (identical or tag == 'S') else (.05 if method == 's2s' else .3) * (1 if tag == 'r1' else 2 if tag == 'r2' else 0)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({'weight': base + np.float32(shift), 'bias': np.ones(3, dtype=np.float32)}, str(directory / 'model.safetensors'))
        epochs = 20 if (method == 'mlm' and tag == 'r2') else 11
        for step in (100, 200):
            write(directory / f'checkpoint-{step}' / 'trainer_state.json', dict(global_step=step, epoch=epochs * step / 200,
                  best_model_checkpoint=f'{directory}/checkpoint-100', best_metric=.3 + shift / 100))
        return directory

    def batch(self, name):
        runs = [self.runs[l] for l, (b, _) in self.arms.items() if b == name]
        arms = [dict(label=r['label'], checkpoint=str(self.checkpoint(r['label']))) for r in runs if self.arms[r['label']][1]]
        write(self.root / name / 'plan.json', dict(arms=arms, stage='full', externals=[dict(label=r['label'], adapter='retrieval') for r in runs
                                                                                    if name in ('references', 't1')]))
        write(self.root / name / 'scores/comparison.json', dict(manifest=dict(sha256=self.digest), runs=runs, judge={}, scoring={}))

    def build_main(self):
        """The merged comparison and its sealed files, shaped like an extension of the original follow-up."""
        original = [self.runs[l] for l, (b, _) in self.arms.items() if b in ('original', 'baselines', 'igcraft_batch')]
        for name in ('original', 'baselines', 'igcraft_batch'):
            arms = [dict(label=l, checkpoint=str(self.checkpoint(l))) for l, (b, c) in self.arms.items() if b == name and c]
            write(self.root / name / 'plan.json', dict(arms=arms, stage='full'))
        main = self.root / 'main'
        write(main / 'scores/comparison.json', dict(manifest=dict(sha256=self.digest), runs=original, judge={}, scoring={}))
        write(main / 'capacity.json', dict(rows=[dict(arm='iglm', system_parameters=1000), dict(arm='igcraft', system_parameters=5000, parameters_by_module=dict(a=1))]))
        write(main / 'plan.json', dict(schema_version=EXTENSION, manifest=str(self.manifest),
                                      clusters=str(self.clusters), externals=['iglm', 'igcraft'], analysis=dict(iterations=1000, seed=42)))
        write(main / 'complete.json', dict(artifacts={n: sha256_file(main / n) for n in ('scores/comparison.json', 'capacity.json')}))
        recipe = dict(name='test_variability', main=str(main), analysis=dict(iterations=1000, seed=42, anchor_trim=2),
                      references=None, **{k: str(self.root / d) for k, d in (('seeds', 'seeds'), ('repeats', 'repeats'), ('capacity', 'capacity'), ('igcraft_t1', 't1'))})
        self.recipe_path = write(self.root / 'recipe.json', recipe)


class VariabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.world = World(Path(cls.temp.name) / 'plain')
        cls.reference_world = World(Path(cls.temp.name) / 'with_references', references=True)
        cls.plan = V.build_plan(cls.world.recipe_path)
        cls.result = V.compute(cls.plan)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def edit(self, label):
        return V.aggregate(self.world.runs[label], 'edit_similarity_mean')

    def test_core_definition(self):
        self.assertEqual((V.core('ABCDEFG', 2), V.core('ABCD', 2), V.core('', 2)), ('CDE', '', ''))
        self.assertEqual((V.core_similarity('', ''), V.core_similarity('', 'ABC'), V.core_similarity('ABC', '')), (1.0, 0.0, 0.0))
        self.assertEqual(V.core_similarity('ABC', 'ABC'), 1.0)
        self.assertAlmostEqual(V.core_similarity('ABC', 'ABD'), 1 - 1 / 3)

    def test_the_plan_freezes_inputs_and_covers_every_generation_file(self):
        frozen = set(self.plan['frozen_files'])
        for label, run in self.world.runs.items():
            self.assertIn(str(Path(run['input']['path']).resolve()), frozen, label)
        self.assertIsNone(self.plan['references'])

    def test_seed_spread_matches_the_runs(self):
        for method in METHODS:
            values = [self.edit(f'{method}_s{s}') for s in V.ALL_SEEDS]
            block = self.result['seeds']['methods'][method]['metrics']['edit']
            self.assertAlmostEqual(block['mean'], statistics.fmean(values))
            self.assertAlmostEqual(block['sd'], statistics.stdev(values))
            self.assertEqual((block['n'], block['min'], block['max']), (7, min(values), max(values)))
            new = self.result['seeds']['methods'][method]['new_seed_edit']
            self.assertEqual(new['n'], 5)
        self.assertEqual(set(self.result['seeds']['pools']), {'iglm', 'empirical', 'igcraft', 'igcraft_t1'})

    def test_contrasts_use_both_endpoints_and_the_planned_family(self):
        contrasts = self.result['seeds']['contrasts']
        self.assertEqual(len(contrasts), 20)
        self.assertEqual({c['endpoint'] for c in contrasts}, {'edit', 'core'})
        mlm_ar = next(c for c in contrasts if (c['a'], c['b'], c['endpoint']) == ('mlm', 'ar', 'edit'))
        seven = lambda m: statistics.fmean(self.edit(f'{m}_s{s}') for s in V.ALL_SEEDS)
        self.assertAlmostEqual(mlm_ar['difference'], seven('mlm') - seven('ar'))
        self.assertEqual((mlm_ar['n_seeds_a'], mlm_ar['n_seeds_b'], mlm_ar['n_groups']), (7, 7, 3))
        vs_iglm = next(c for c in contrasts if (c['a'], c['b'], c['endpoint']) == ('mlm', 'iglm', 'edit'))
        self.assertEqual(vs_iglm['n_seeds_b'], 1)
        self.assertIn('ci_low', mlm_ar['frameworks_only'])
        self.assertEqual(len(self.result['seeds']['new_seed_contrasts']), 3)

    def test_contrast_matches_the_statistic_it_wraps(self):
        study = V.Study(json.loads(self.world.manifest.read_text()), json.loads(self.world.clusters.read_text()), self.plan['analysis'])
        runs = {m: [self.world.runs[f'{m}_s{s}'] for s in V.ALL_SEEDS] for m in ('mlm', 'ar')}
        direct = crossed_interval(study.matrix(runs['mlm'], 'edit'), study.matrix(runs['ar'], 'edit'), study.groups, 1000,
                                  V.stable_example_seed(42, 'mlm:ar:edit'), 10)
        reported = next(c for c in self.result['seeds']['contrasts'] if (c['a'], c['b'], c['endpoint']) == ('mlm', 'ar', 'edit'))
        self.assertEqual((reported['ci_low'], reported['ci_high']), (direct['ci_low'], direct['ci_high']))

    def test_core_vectors_come_from_the_saved_candidates(self):
        study = V.Study(json.loads(self.world.manifest.read_text()), json.loads(self.world.clusters.read_text()), self.plan['analysis'])
        run = self.world.runs['ar_s42']
        expected = [statistics.fmean(V.core_similarity(V.core(s, 2), V.core(n, 2)) for s in self.world.samples['ar_s42'][i]) for i, n in FRAMEWORKS]
        self.assertTrue(np.allclose(study.vector(run, 'core'), expected))

    def test_determinism_separates_identical_candidates_from_drifting_ones(self):
        rows = self.result['determinism']['methods']
        self.assertEqual(rows['ar']['identical_frameworks'], {'s42_r1': 6, 'r1_r2': 6, 's42_r2': 6})
        self.assertEqual(rows['ar']['weights']['s42_r1']['tensors_differing'], 0)
        self.assertEqual(rows['mlm']['identical_frameworks']['r1_r2'], 0)
        self.assertGreater(rows['mlm']['weights']['r1_r2']['max_abs_difference'], 0.2)
        self.assertGreater(rows['s2s']['weights']['s42_r1']['max_abs_difference'], 0)
        self.assertEqual([rows['mlm']['runs'][k]['trajectory']['epochs'] for k in ('s42', 'r1', 'r2')], [11.0, 11.0, 20.0])
        self.assertEqual(rows['ar']['edit_range'], 0.0)

    def test_capacity_rows_report_parameters_and_the_change_in_seed_sd(self):
        rows = {(r['method'], r['size']): r for r in self.result['capacity']['rows']}
        self.assertEqual(len(rows), 12)
        self.assertLess(rows[('ar', 'S')]['parameters'], rows[('ar', 'M')]['parameters'])
        self.assertLess(rows[('ar', 'L')]['parameters'], rows[('ar', 'XL')]['parameters'])
        small = statistics.fmean(self.edit(f'mlm_s{s}') for s in V.SCAN_SEEDS)
        large = statistics.fmean(self.edit(f'mlm_L_s{s}') for s in V.SCAN_SEEDS)
        self.assertAlmostEqual(rows[('mlm', 'L')]['edit_minus_small'], large - small)
        sd = self.result['capacity']['small_model_seed_sd']['mlm']
        self.assertAlmostEqual(rows[('mlm', 'L')]['edit_minus_small_in_seed_sd'], (large - small) / sd)
        self.assertEqual(rows[('mlm', 'S')]['edit_minus_small'], 0.0)
        self.assertEqual(self.result['external_parameters']['igcraft']['by_module'], {'a': 1})

    def test_sensitivity_compares_lengths_gaps_and_scores(self):
        sensitivity = self.result['sensitivity']
        self.assertEqual(sensitivity['arms']['default']['label'], 'igcraft')
        self.assertEqual(sensitivity['arms']['sensitivity']['lengths']['interior_gap_slots'], 6)  # one per framework
        self.assertEqual(sensitivity['arms']['default']['lengths']['interior_gap_slots'], 0)
        self.assertAlmostEqual(sensitivity['edit_difference']['difference'], self.edit('igcraft_t1') - self.edit('igcraft'))
        self.assertEqual(sensitivity['arms']['default']['lengths']['attempts'], 36)

    def test_the_ladder_needs_references_and_reports_the_share_of_learned_gain(self):
        self.assertIsNone(self.result['ladder'])
        world = self.reference_world
        plan = V.build_plan(world.recipe_path, str(world.root / 'references'))
        result = V.compute(plan)
        ladder = result['ladder']['edit']
        floor = V.aggregate(world.runs['empirical'], 'edit_similarity_mean')
        learned = ladder['learned']['mlm']
        retrieval = V.aggregate(world.runs['retrieval'], 'edit_similarity_mean')
        self.assertAlmostEqual(ladder['captured']['retrieval']['mlm'], (retrieval - floor) / (learned - floor))
        self.assertEqual(set(ladder['references']), {'retrieval', 'retrieval_j', 'retrieval_fine'})
        self.assertEqual(len(result['seeds']['contrasts']), 2 * 16)  # six more reference contrasts per endpoint

    def test_report_and_figures_render(self):
        text = V.report(self.result)
        for heading in ('## Training seeds', 'Seed-aware contrasts', '## Same seed, identical config', '## Capacity', '## IgCraft temperature', '## Limitations'):
            self.assertIn(heading, text)
        with tempfile.TemporaryDirectory() as directory:
            V.capacity_figure(self.result, Path(directory) / 'c.png')
            V.seed_figure(self.result, Path(directory) / 's.png')
            for name in ('c.png', 'c.pdf', 's.png', 's.pdf'):
                self.assertGreater((Path(directory) / name).stat().st_size, 1000, name)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.world = World(Path(temp.name))

    def test_execute_seals_its_outputs_and_refuses_later_changes(self):
        plan = V.build_plan(self.world.recipe_path)
        output = self.world.root / 'out'
        V.execute(plan, output)
        complete = json.loads((output / 'complete.json').read_text())
        self.assertEqual(set(complete['artifacts']), set(V.ARTIFACTS))
        self.assertTrue(all(sha256_file(output / n) == d for n, d in complete['artifacts'].items()))
        generation = Path(self.world.runs['ar_s44']['input']['path'])
        generation.write_text(generation.read_text().replace('"complete": true', '"complete": true '))
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            V.validate_plan(plan)

    def test_generations_that_differ_from_what_was_scored_are_refused(self):
        path = Path(self.world.runs['mlm_s44']['input']['path'])
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'differ from what was scored'):
            V.build_plan(self.world.recipe_path)

    def test_unfinished_generations_are_refused(self):
        path = Path(self.world.runs['mlm_s44']['input']['path'])
        payload = json.loads(path.read_text())
        payload['finished_at'] = None
        path.write_text(json.dumps(payload))
        scores = self.world.root / 'seeds/scores/comparison.json'
        data = json.loads(scores.read_text())
        next(r for r in data['runs'] if r['label'] == 'mlm_s44')['input']['sha256'] = sha256_file(path)
        scores.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            V.build_plan(self.world.recipe_path)

    def test_missing_arms_and_mismatched_manifests_are_refused(self):
        scores = self.world.root / 'capacity/scores/comparison.json'
        data = json.loads(scores.read_text())
        kept = [r for r in data['runs'] if r['label'] != 'ar_XL_s43']
        scores.write_text(json.dumps(dict(data, runs=kept)))
        with self.assertRaisesRegex(ValueError, "capacity batch lacks \\['ar_XL_s43'\\]"):
            V.build_plan(self.world.recipe_path)
        scores.write_text(json.dumps(dict(data, manifest=dict(sha256='other'))))
        with self.assertRaisesRegex(ValueError, 'different manifests'):
            V.build_plan(self.world.recipe_path)

    def test_a_batch_that_was_never_run_is_explained(self):
        with self.assertRaisesRegex(ValueError, 'No scored comparison in .*never_run.*must run to completion first'):
            V.build_plan(self.world.recipe_path, str(self.world.root / 'never_run'))
        shutil_free = self.world.root / 'seeds/scores/comparison.json'
        shutil_free.rename(shutil_free.with_suffix('.moved'))  # a batch whose scoring stage did not finish
        with self.assertRaisesRegex(ValueError, 'No scored comparison'):
            V.build_plan(self.world.recipe_path)

    def test_a_changed_main_comparison_fails_its_seal(self):
        main = self.world.root / 'main/capacity.json'
        main.write_text(main.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Base artifact changed'):
            V.build_plan(self.world.recipe_path)


if __name__ == '__main__':
    unittest.main()
