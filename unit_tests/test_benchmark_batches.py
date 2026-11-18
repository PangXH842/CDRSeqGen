"""Tests merged from: test_benchmark_extend.py, test_benchmark_followup.py, test_benchmark_leakage.py, test_export_manifest.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from benchmark._common import sha256_file
from benchmark.extend import build_plan, execute, summary, verify_complete
import random
import struct
from unittest.mock import patch
import numpy as np
from benchmark.analysis import analyze, cluster_interval, crossed_interval, planned_pairs
from benchmark.capacity import gpt2_parameters, tensor_parameters
from benchmark.empirical import (
    draw_training_loops,
    generate,
    representative_clusters,
    training_pool,
    verify_splits,
)
from benchmark.followup import merge_scores, validate_plan
from benchmark._common import execution_source_identity
from benchmark.scorer import score_payload, validate_payload
from contextlib import redirect_stdout
import io
from benchmark.empirical import representative_clusters, verify_splits
from benchmark import leakage
import hashlib
from benchmark.export_manifest import build_manifest


# ====================================================================================================
# from test_benchmark_extend.py
# Merge, paired analysis and capacity for generators added to a finished comparison, on synthetic batches.
# ====================================================================================================

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

    def references_batch(self):
        """A second added batch: two parameter-free retrieval pools, as the retrieval recipe produces."""
        self.added2 = self.root / 'added2'
        generations = {label: write(self.added2 / f'generations/{label}.json', dict(finished_at='done', model=dict(parameters_network=0)))
                       for label in ('retrieval', 'retrieval_j')}
        runs = [dict(run(label, edit), input=dict(path=str(path), sha256=sha256_file(path)))
                for (label, path), edit in zip(generations.items(), (.33, .32))]
        write(self.added2 / 'plan.json', dict(arms=[], stage='full', externals=[dict(label=l, adapter='retrieval') for l in generations]))
        write(self.added2 / 'scores/comparison.json', scores(*runs))

    def test_extensions_chain_and_assign_reference_and_excluded_roles(self):
        first_plan, first = self.extend()
        self.references_batch()
        plan = build_plan(first, self.added2, references=['retrieval'], exclude=['retrieval_j'])
        self.assertEqual((plan['externals'], plan['references'], plan['excluded']),
                         (['iglm', 'igcraft'], ['empirical', 'retrieval'], ['retrieval_j']))
        self.assertEqual(plan['clusters'], str((self.base / 'clusters.json').resolve()))
        second = self.root / 'extension2'
        execute(plan, second)
        analysis = json.loads((second / 'analysis.json').read_text())
        expected = {('mlm', 'iglm'), ('mlm', 'igcraft'), ('mlm', 'empirical'), ('iglm', 'empirical'), ('igcraft', 'empirical'),
                    ('retrieval', 'empirical'), ('mlm', 'retrieval'), ('iglm', 'retrieval'), ('igcraft', 'retrieval')}
        self.assertEqual({tuple(pair) for pair in analysis['pairs']}, expected)
        self.assertEqual(analysis['primary_comparison_family'], 9)
        self.assertEqual(analysis['roles'], dict(externals=['iglm', 'igcraft'], references=['empirical', 'retrieval'], excluded=['retrieval_j']))
        merged = json.loads((second / 'scores/comparison.json').read_text())
        self.assertEqual([r['label'] for r in merged['runs']], ['mlm', 'iglm', 'empirical', 'igcraft', 'retrieval', 'retrieval_j'])
        self.assertEqual(len(merged['source_scoring_provenance']), 3)
        capacity = json.loads((second / 'capacity.json').read_text())['rows']
        self.assertEqual([r['arm'] for r in capacity][-3:], ['igcraft', 'retrieval', 'retrieval_j'])
        self.assertEqual((capacity[-1]['system_parameters'], capacity[-1]['timing_scope']), (0, 'No neural training'))
        self.assertEqual(json.loads((first / 'complete.json').read_text())['added'], ['igcraft'])  # the earlier extension is untouched

    def test_an_older_extension_plan_without_roles_still_chains(self):
        _, first = self.extend()
        old = json.loads((first / 'plan.json').read_text())
        for key in ('clusters', 'references', 'excluded'):
            old.pop(key)
        (first / 'plan.json').write_text(json.dumps(old))
        self.references_batch()
        plan = build_plan(first, self.added2, references=['retrieval'], exclude=['retrieval_j'])
        self.assertEqual(plan['references'], ['empirical', 'retrieval'])
        self.assertEqual(plan['clusters'], str((self.base / 'clusters.json').resolve()))

    def test_roles_must_name_added_arms_once(self):
        self.references_batch()
        for options in (dict(references=['mlm']), dict(exclude=['nope']), dict(references=['retrieval'], exclude=['retrieval'])):
            with self.subTest(options), self.assertRaisesRegex(ValueError, 'must name added arms'):
                build_plan(self.base, self.added2, **options)

    def test_short_candidate_coverage_needs_an_explicit_waiver(self):
        self.write_added(run('igcraft', .40, valid=SAMPLES - 1))
        with self.assertRaisesRegex(ValueError, 'Missing candidates'):
            self.extend()
        plan = build_plan(self.base, self.added, allow_incomplete=True)
        execute(plan, self.root / 'waived')
        merged = json.loads((self.root / 'waived/scores/comparison.json').read_text())
        self.assertEqual(merged['runs'][-1]['records'][0]['metrics']['n_samples'], SAMPLES - 1)


# ====================================================================================================
# from test_benchmark_followup.py
# Synthetic CPU checks for empirical sampling, verified groups and paired inference.
# ====================================================================================================

def record(loop, split='train'):
    return dict(heavy=dict(cdr3=loop),split=split)


def manifest():
    return dict(schema_version='cdrseqgen.benchmark_manifest.v1',benchmark_id='fixture',
                task=dict(track='length_free',anchor_trim_each_end=2),records=[dict(example_id='a',native='AAAAAA',
                native_length=6,pt1='CCC',pt2='DDD',parent_sequence='CCCAAAAAADDD',infill_range=[3,9])])


def naive_clusters(sequences,identity):
    from collections import defaultdict
    buckets=defaultdict(list)
    for i,s in enumerate(sequences):buckets[len(s)].append(i)
    labels=[-1]*len(sequences);next_id=0
    for length,indices in buckets.items():
        representatives=[]
        for i in indices:
            selected=next((group for seq,group in representatives if sum(a==b for a,b in zip(seq,sequences[i]))>=identity*length),None)
            if selected is None:
                selected=next_id;next_id+=1;representatives.append((sequences[i],selected))
            labels[i]=selected
    return labels


class EmpiricalTests(unittest.TestCase):
    def test_training_only_and_duplicate_frequencies(self):
        pool=training_pool([record('AAA'),record('CCC','test'),record('AAA'),record('DDD','validation')])
        self.assertEqual(pool,[(0,'AAA'),(2,'AAA')])

    def test_draws_keep_source_indices_and_seed(self):
        pool=[(2,'AAA'),(8,'CCC')]
        a=draw_training_loops(pool,'framework',20,42)
        self.assertEqual(a,draw_training_loops(pool,'framework',20,42))
        self.assertTrue(all(draw in pool for draw in a))

    def test_native_and_flanks_do_not_affect_draws(self):
        a=manifest();b=copy.deepcopy(a)
        b['records'][0].update(native='YYYYYYYY',native_length=8,pt1='EEE',pt2='FFF')
        corpus=[record('AAAAAA'),record('CCCCCC')]
        first=generate(a,corpus,20,42,{},{});second=generate(b,corpus,20,42,{},{})
        self.assertEqual(first['records'][0]['samples'],second['records'][0]['samples'])

    def test_invalid_training_loop_and_missing_split_fail(self):
        for corpus in [[record('AXA')],[dict(heavy=dict(cdr3='AAA'))]]:
            with self.assertRaises(ValueError):training_pool(corpus)

    def test_common_scorer_accepts_baseline_contract(self):
        m=manifest();payload=generate(m,[record('AAAAAA'),record('CCCCCCC')],20,42,
                                    dict(sha256='digest'),{})
        validate_payload(payload,label='empirical',manifest=m,manifest_hash='digest',allow_partial=False)
        class FakeJudge:
            def score(self,record,samples):return [2.0]*len(samples)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'draws.json';path.write_text(json.dumps(payload))
            scored=score_payload('empirical',path,payload,{'a':m['records'][0]},2,FakeJudge())
        self.assertEqual(scored['aggregate']['micro_accounting']['attempts'],20)
        self.assertEqual(scored['records'][0]['metrics']['judge_candidates_scored'],20)


class ClusterTests(unittest.TestCase):
    def test_accelerated_groups_equal_original_algorithm(self):
        rng=random.Random(8);sequences=[]
        for length in [16,6,26,12,49,20]:
            for _ in range(22):
                s=''.join(rng.choice('ACDEFGHIKLMNPQRSTVWY') for _ in range(length))
                sequences.extend([s,s,'Y'+s[1:],s[:-2]+'YY'])
        rng.shuffle(sequences)
        for identity in [.85,1,.7]:
            self.assertEqual(representative_clusters(sequences,identity),naive_clusters(sequences,identity))

    def test_first_matching_representative_is_preserved(self):
        sequences=['AAAAAACCCC','CCCCAAAAAA','AAAACCAAAA','AAAAAACCCC']
        self.assertEqual(representative_clusters(sequences,.5),naive_clusters(sequences,.5))

    def test_exact_split_reconstruction_and_mismatch_rejection(self):
        corpus=[record('AAAA'),record('AAAA','test'),record('CCCC','validation'),record('DDDD')]
        groups=[0,0,1,2]
        # Recover an expected allocation independently from cluster ordering/shuffle.
        keys=[0,1,2];random.Random(42).shuffle(keys)
        assigned=['train']*4
        for group,label in [(keys[0],'test'),(keys[1],'validation')]:
            for i,key in enumerate(groups):
                if key==group:assigned[i]=label
        for item,label in zip(corpus,assigned):item['split']=label
        self.assertEqual(verify_splits(corpus,groups,.25,.25,42)['split_labels_verified'],4)
        corpus[0]['split']='changed'
        with self.assertRaises(ValueError):verify_splits(corpus,groups,.25,.25,42)


class IntervalTests(unittest.TestCase):
    def test_pairing_cancels_constant_offset_with_unequal_groups(self):
        a=[.1,.5,.9,.3];b=[v-.1 for v in a]
        result=cluster_interval(a,b,['x','x','x','y'],2000,2,13)
        self.assertAlmostEqual(result['ci_low'],.1)
        self.assertAlmostEqual(result['family_ci_high'],.1)

    def test_framework_weighting_is_retained(self):
        result=cluster_interval([1,1,1,0],[0]*4,['x','x','x','y'],2000,42)
        self.assertEqual(result['difference'],.75)
        self.assertEqual(result['n_groups'],2)

    def test_grouping_prevents_candidate_like_independence(self):
        a=[0]*30+[1]*30;b=[0]*60
        grouped=cluster_interval(a,b,['a']*30+['b']*30,2000,42)
        singleton=cluster_interval(a,b,list(range(60)),2000,42)
        self.assertGreater(grouped['ci_high']-grouped['ci_low'],singleton['ci_high']-singleton['ci_low'])

    def test_family_interval_is_at_least_as_wide(self):
        result=cluster_interval([.1,.3,.5,.9],[0]*4,list(range(4)),2000,42,13)
        self.assertLessEqual(result['family_ci_low'],result['ci_low'])
        self.assertGreaterEqual(result['family_ci_high'],result['ci_high'])

    def test_undefined_single_group_does_not_claim_precision(self):
        result=cluster_interval([.1,.2],[0,0],['x','x'],1000)
        self.assertIsNone(result['ci_low'])

    def test_analysis_uses_paired_conditional_population(self):
        runs=[]
        for label,values in [('local',[.2,None,.4]),('iglm',[.1,.8,None]),('empirical',[.1,.1,.1])]:
            runs.append(dict(label=label,records=[dict(example_id=str(i),metrics=dict(edit_similarity_mean=.2,core_aar_mean=v)) for i,v in enumerate(values)]))
        result=analyze(dict(runs=runs),{'0':0,'1':1,'2':2},[('local','iglm')],1000)
        row=next(r for r in result['results'] if r['metric']=='core_aar_mean')
        self.assertEqual(row['example_ids'],['0']);self.assertIsNone(row['ci_low'])

    def test_pair_family_keeps_six_seeds_and_one_baseline_pool(self):
        labels=['mlm_s42','mlm_s43','ar_s42','ar_s43','s2s_s42','s2s_s43','iglm','empirical']
        self.assertEqual(len(planned_pairs(labels)),13)
        self.assertEqual(planned_pairs(labels).count(('iglm','empirical')),1)


class CrossedIntervalTests(unittest.TestCase):
    def data(self, seeds=1, spread=0.0, frameworks=60, shift=0.05):
        rng = random.Random(7)
        base = [rng.gauss(.4, .08) for _ in range(frameworks)]
        a = [[x + shift + rng.gauss(0, spread) for x in base] for _ in range(seeds)]
        b = [[x + rng.gauss(0, spread) for x in base] for _ in range(seeds)]
        return a, b, list(range(frameworks))

    def test_one_row_per_arm_reduces_to_the_cluster_interval(self):
        a, b, groups = self.data()
        crossed = crossed_interval(a, b, groups, iterations=20000)
        plain = cluster_interval(a[0], b[0], groups, iterations=20000)
        self.assertAlmostEqual(crossed['difference'], plain['difference'])
        self.assertAlmostEqual(crossed['ci_low'], plain['ci_low'], delta=.004)
        self.assertAlmostEqual(crossed['ci_high'], plain['ci_high'], delta=.004)
        self.assertEqual((crossed['n_seeds_a'], crossed['n_seeds_b'], crossed['n_groups']), (1, 1, 60))

    def test_training_variability_widens_the_interval(self):
        a, b, groups = self.data(seeds=7, spread=.03)
        crossed = crossed_interval(a, b, groups, iterations=20000)
        frameworks_only = cluster_interval(np.mean(a, axis=0), np.mean(b, axis=0), groups, iterations=20000)
        # About 1.7x with seed resampling and about 1.0x without it, so this cannot pass by random noise.
        self.assertGreater(crossed['ci_high'] - crossed['ci_low'], 1.4 * (frameworks_only['ci_high'] - frameworks_only['ci_low']))
        self.assertAlmostEqual(crossed['difference'], .05, delta=.02)

    def test_a_fixed_pool_has_no_seed_variance_and_groups_are_resampled_whole(self):
        a, b, _ = self.data(seeds=5, spread=.02, frameworks=6)
        fixed = np.mean(b, axis=0)
        result = crossed_interval(a, fixed, [0, 0, 1, 1, 2, 2], iterations=2000)
        self.assertEqual((result['n_groups'], result['n_seeds_a'], result['n_seeds_b']), (3, 5, 1))

    def test_degenerate_and_invalid_inputs(self):
        a, b, groups = self.data(frameworks=6)
        self.assertIsNone(crossed_interval(a, b, [0] * 6, iterations=1000)['ci_low'])
        for args in [dict(a=a, b=b, groups=groups[:5]), dict(a=[], b=b, groups=groups), dict(a=[[float('nan')] * 6], b=b, groups=groups)]:
            with self.assertRaises(ValueError):
                crossed_interval(**args, iterations=1000)
        with self.assertRaises(ValueError):
            crossed_interval(a, b, groups, iterations=10)


class AccountingTests(unittest.TestCase):
    def test_gpt2_count_matches_tiny_tied_model(self):
        from transformers import GPT2Config,GPT2LMHeadModel
        model=GPT2LMHeadModel(GPT2Config(n_embd=16,n_layer=2,n_head=2,n_positions=32,n_inner=48,vocab_size=30,bos_token_id=0,eos_token_id=1))
        self.assertEqual(gpt2_parameters(model.config.to_dict()),sum(p.numel() for p in model.parameters()))

    def test_header_count_includes_learned_position_tensor(self):
        header={'pos_embedding':dict(shape=[1,4,8]),'weight':dict(shape=[2,8])}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'model.safetensors';data=json.dumps(header).encode();p.write_bytes(struct.pack('<Q',len(data))+data)
            self.assertEqual(tensor_parameters(p),48)

    def test_resume_rejects_modified_frozen_input(self):
        from benchmark.followup import environment
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'data';p.write_text('changed')
            plan=dict(source=execution_source_identity(),environment=environment(),frozen_files={str(p):'wrong'})
            with self.assertRaises(ValueError):validate_plan(plan)

    def test_merge_preserves_original_rows_and_refuses_judge_drift(self):
        record=dict(example_id='a',native='AAA',native_length=3,
                    metrics=dict(n_samples=2,judge_candidates_scored=2,edit_similarity_mean=.5,judge_perplexity_mean=2.))
        original=dict(manifest=dict(sha256='same'),judge=dict(name='judge',model_commit='v1'),scoring=dict(aggregation_unit='example'),
                      runs=[dict(label='iglm',sampling=dict(track='length_free',num_samples=2,base_seed=42),records=[record])])
        added=copy.deepcopy(original);added['runs'][0]['label']='empirical'
        self.assertEqual(merge_scores(original,added)['runs'][0],original['runs'][0])
        added['judge']['model_commit']='v2'
        with self.assertRaises(ValueError):merge_scores(original,added)

    def test_merge_refuses_truth_budget_or_primary_population_drift(self):
        record=dict(example_id='a',native='AAA',native_length=3,
                    metrics=dict(n_samples=2,judge_candidates_scored=2,edit_similarity_mean=.5,judge_perplexity_mean=2.))
        original=dict(manifest=dict(sha256='same'),judge={},scoring={},runs=[
            dict(label='iglm',sampling=dict(track='length_free',num_samples=2,base_seed=42),records=[record])])
        for change in [dict(native='CCC'),dict(metrics={**record['metrics'],'n_samples':1,'judge_candidates_scored':1}),
                       dict(metrics={**record['metrics'],'edit_similarity_mean':float('nan')})]:
            added=copy.deepcopy(original);added['runs'][0]['label']='empirical'
            added['runs'][0]['records'][0].update(change)
            with self.assertRaises(ValueError):merge_scores(original,added)

    def test_resume_refuses_changed_completed_output_before_gpu_work(self):
        from benchmark.followup import execute
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);(output/'clusters.json').write_text('modified')
            (output/'progress.json').write_text(json.dumps({'clusters.json':'original-hash'}))
            with patch('benchmark.followup.validate_plan'),patch('benchmark.followup.subprocess.run') as child:
                with self.assertRaisesRegex(ValueError,'artifact changed'):
                    execute({},output,'0',resume=True)
                child.assert_not_called()

    def test_complete_fixture_followup_resumes_without_rejudging(self):
        """Exercise orchestration on four synthetic sequences and a mocked judge process."""
        from benchmark.followup import execute, summary
        corpus=[record(loop) for loop in ['AAAAAA','CCCCCC','DDDDDD','EEEEEE']]
        order=list(range(4));random.Random(42).shuffle(order)
        for index in order[:2]:corpus[index]['split']='validation'
        m=manifest();m['records']=[]
        for index in order[:2]:
            item=manifest()['records'][0].copy()
            item.update(example_id=str(index),source_index=index,native=corpus[index]['heavy']['cdr3'])
            item['parent_sequence']=item['pt1']+item['native']+item['pt2']
            m['records'].append(item)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=root/'original';batch.mkdir();output=root/'followup'
            for name,data in [('corpus.json',corpus),('manifest.json',m),('iglm.json',{})]:
                (root/name).write_text(json.dumps(data))
            (batch/'plan.json').write_text('{}')
            baseline=generate(m,corpus,2,42,dict(sha256=sha256_file(root/'manifest.json')),{})
            rows=[dict(example_id=r['example_id'],native=r['native'],native_length=6,
                       metrics=dict(n_samples=2,judge_candidates_scored=2,edit_similarity_mean=.5,judge_perplexity_mean=2.)) for r in m['records']]
            original=dict(manifest=baseline['manifest'],judge=dict(name='fixture',max_len=384),scoring={},
                          runs=[dict(label='iglm',sampling=baseline['sampling'],records=rows)])
            (root/'scores.json').write_text(json.dumps(original))
            plan=dict(source={},manifest=str(root/'manifest.json'),corpus=str(root/'corpus.json'),
                      source_scores=str(root/'scores.json'),source_batch=str(batch),iglm_config=str(root/'iglm.json'),
                      cluster_settings=dict(cluster_identity=.85,test_fraction=0,validation_fraction=.5,seed=42),
                      frozen_files={str(root/'corpus.json'):sha256_file(root/'corpus.json')},clusters_input=None,
                      num_samples=2,sampling_seed=42,baseline_attempts=4,judge=original['judge'],
                      pairs=[('iglm','empirical')],analysis=dict(iterations=1000,seed=42))
            def fake_judge_process(command,**kwargs):
                self.assertEqual(command[1],str(Path(__file__).resolve().parents[1]/'benchmark/scorer.py'))
                payload=json.loads((output/'generations/empirical.json').read_text())
                class FakeJudge:
                    def score(self,item,samples):return [2.]*len(samples)
                scored=score_payload('empirical',output/'generations/empirical.json',payload,
                                     {r['example_id']:r for r in m['records']},2,FakeJudge())
                added={**original,'runs':[scored]}
                destination=Path(command[command.index('--output')+1]);destination.parent.mkdir()
                destination.write_text(json.dumps(added))
            with patch('benchmark.followup.validate_plan'),patch('benchmark.followup.extract',return_value=dict(rows=[])),\
                 patch('benchmark.followup.plot',side_effect=lambda data,path:path.write_bytes(b'fixture-plot')),\
                 patch('benchmark.followup.subprocess.run',side_effect=fake_judge_process) as child:
                execute(plan,output,'0')
                completion=json.loads((output/'complete.json').read_text())
                for filename,digest in completion['artifacts'].items():
                    self.assertEqual(sha256_file(output/filename),digest)
                execute(plan,output,'0',resume=True)
                child.assert_called_once()
            self.assertEqual(json.loads((output/'scores/comparison.json').read_text())['runs'][0],original['runs'][0])
            (output/'analysis.json').write_text('changed')
            with patch('builtins.print') as printed:
                summary(output)
                self.assertIn('complete.json: changed',[call.args[0] for call in printed.call_args_list])


# ====================================================================================================
# from test_benchmark_leakage.py
# Synthetic CPU tests for exact neighbours, cohort decisions and safe resume.
# ====================================================================================================

def benchmark_leakage_record(index, loop, split='train'):
    chain = dict(fwr1='ACD', cdr1='EFG', fwr2='HIK', cdr2='LMN',
                 fwr3='PQR', cdr3=loop, fwr4='STV')
    light = {**chain, 'cdr3': 'WYAC'}
    return dict(sequence_id=str(index), source=dict(run=1), split=split, heavy=chain, light=light)


def normalized(raw):
    return leakage.normalize_corpus(raw, dict(sources=[dict(run=1, subject='donor')]))


def fixture(directory):
    """Ten disjoint synthetic H3 groups; no project corpus or models involved."""
    raw = [benchmark_leakage_record(i, amino * 6) for i, amino in enumerate('ACDEFGHIKL')]
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
        records = normalized([benchmark_leakage_record(0, 'C' * 3 + 'A' * 17), benchmark_leakage_record(1, 'C' * 4 + 'A' * 16),
                              benchmark_leakage_record(2, 'A' * 20, 'test')])
        result = leakage.nearest_h3([records[2]], leakage.reference_index(records[:2]))[2]
        self.assertEqual(result['neighbours_at_cutoff'], 1)
        self.assertEqual(result['closest_matches'], 17)

    def test_nearest_below_cutoff_and_tie_source_order_across_tiles(self):
        records = normalized([benchmark_leakage_record(0, 'AAAC'), benchmark_leakage_record(1, 'AACA'), benchmark_leakage_record(2, 'CCCC'), benchmark_leakage_record(3, 'AAAA', 'test')])
        result = leakage.nearest_h3([records[3]], leakage.reference_index(list(reversed(records[:3]))),
                                   reference_tile=1)[3]
        self.assertEqual(result['closest_source_index'], 0)
        self.assertEqual(result['closest_identity'], .75)
        self.assertEqual(result['neighbours_at_cutoff'], 0)

    def test_no_same_length_reference_and_duplicate_counts(self):
        records = normalized([benchmark_leakage_record(0, 'AAAA'), benchmark_leakage_record(1, 'AAAA'), benchmark_leakage_record(2, 'AAAA', 'test'),
                              benchmark_leakage_record(3, 'AAAAA', 'test')])
        result = leakage.nearest_h3(records[2:], leakage.reference_index(records[:2]))
        self.assertEqual(result[2]['exact_h3_references'], 2)
        self.assertEqual(result[2]['neighbours_at_cutoff'], 2)
        self.assertIsNone(result[3]['closest_identity'])
        self.assertEqual(result[3]['same_length_references'], 0)

    def test_matches_independent_bruteforce_for_many_lengths_tiles_and_thresholds(self):
        rng = random.Random(33)
        refs = [benchmark_leakage_record(i, ''.join(rng.choice('ACDE') for _ in range(rng.choice([4, 7, 20])))) for i in range(31)]
        queries = [benchmark_leakage_record(i + 31, ''.join(rng.choice('ACDE') for _ in range(rng.choice([4, 7, 20, 49]))), 'test')
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
        records = normalized([benchmark_leakage_record(i, s, 'test' if i == 2 else 'train') for i, s in enumerate(loops)])
        result = leakage.nearest_h3([records[2]], leakage.reference_index(records[:2]))[2]
        self.assertEqual(result['closest_identity'], .95)
        self.assertEqual(result['closest_source_index'], 1)
        self.assertEqual(result['neighbours_at_cutoff'], 1)

    def test_no_silent_sequence_normalization_or_provenance_guess(self):
        for loop in ['aaac', 'AA X', '', 'AAAX']:
            with self.assertRaises(ValueError): normalized([benchmark_leakage_record(0, loop)])
        with self.assertRaises(ValueError):
            leakage.normalize_corpus([benchmark_leakage_record(0, 'AAAA')], dict(sources=[]))

    def test_empty_frameworks_allowed_and_non_h3_ambiguity_is_literal_and_flagged(self):
        raw = benchmark_leakage_record(0, 'AAAA'); raw['heavy']['fwr4'] = ''; raw['light']['cdr1'] = 'AXA'
        result = normalized([raw])[0]
        self.assertEqual(result['heavy_flanks'][1], '')
        self.assertEqual(result['l1'], 'AXA')
        self.assertEqual(result['noncanonical_regions'], ['light.cdr1'])


class CohortTests(unittest.TestCase):
    def test_validation_exposure_excludes_test_but_shared_flanks_do_not(self):
        records = normalized([benchmark_leakage_record(0, 'CCCC', 'train'), benchmark_leakage_record(1, 'AAAA', 'validation'),
                              benchmark_leakage_record(2, 'AAAA', 'test'), benchmark_leakage_record(3, 'GGGG', 'test')])
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


# ====================================================================================================
# from test_export_manifest.py
# ====================================================================================================

def _entry():
    regions = {
        "fwr1": "Q",
        "cdr1": "A",
        "fwr2": "W",
        "cdr2": "C",
        "fwr3": "R",
        "cdr3": "DDD",
        "fwr4": "G",
    }
    return {
        "sequence_id": "example",
        "species": "human",
        "split": "test",
        "source": {"dataset": "OAS", "run": 7},
        "heavy": {**regions, "full_aa": "QAWCRDDDG"},
    }


class ExportManifestTests(unittest.TestCase):
    def test_length_free_track_is_written_to_task_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "processed.json"
            path.write_text(json.dumps([_entry()]))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".metadata.json").write_text(json.dumps({
                "dataset": "OAS",
                "processed_file_sha256": digest,
            }))

            manifest = build_manifest(
                path,
                split="test",
                sample_size=None,
                selection_seed="test",
                benchmark_id="length_free_test",
                track="length_free",
            )

        self.assertEqual(manifest["task"]["track"], "length_free")
        self.assertEqual(len(manifest["records"]), 1)

    def test_unknown_track_is_rejected_before_reading_input(self):
        with self.assertRaisesRegex(ValueError, "unsupported benchmark track"):
            build_manifest(
                Path("does-not-exist.json"),
                split="test",
                sample_size=None,
                selection_seed="test",
                benchmark_id="bad",
                track="unknown",
            )


if __name__ == "__main__":
    unittest.main()
