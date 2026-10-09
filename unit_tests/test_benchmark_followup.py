"""Synthetic CPU checks for empirical sampling, verified groups and paired inference."""
import copy
import json
import random
import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from benchmark.analysis import analyze, cluster_interval, planned_pairs
from benchmark.capacity import gpt2_parameters, tensor_parameters
from benchmark.empirical import draw_training_loops, generate, representative_clusters, training_pool, verify_splits
from benchmark.followup import merge_scores, validate_plan
from benchmark._common import execution_source_identity
from benchmark._common import sha256_file
from benchmark.scorer import score_payload, validate_payload


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


if __name__=='__main__':unittest.main()
