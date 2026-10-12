"""Prepare/run the empirical baseline, verified-group analysis and compute export.

Commands: plan, smoke, run, status, resume. Plan reads hashes/metadata only.
User-run stages reconstruct clusters on CPU and judge only the new baseline
draws on one GPU. Saved neural candidates and scores remain read-only.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark._common import atomic_write_json, execution_source_identity, sha256_file, validate_manifest_record
from benchmark.analysis import analyze, planned_pairs, plot, report
from benchmark.capacity import extract
from benchmark.empirical import generate, representative_clusters, verify_splits


def read(path):
    return json.loads(Path(path).read_text())


def environment():
    return dict(python=sys.version, executable=sys.executable,
                packages={n:importlib.metadata.version(n) for n in ['numpy','torch','transformers','matplotlib']})


def build_plan(recipe_path, stage, clusters_input=None):
    """Validate frozen inputs and reused score membership without running models."""
    recipe=read(recipe_path)
    if stage not in {'full','smoke'}:raise ValueError('Unsupported stage')
    iterations=recipe['analysis']['iterations']
    if not isinstance(iterations,int) or isinstance(iterations,bool) or iterations<1000:
        raise ValueError('Analysis requires at least 1,000 integer resamples')
    batch=ROOT/recipe['smoke_source_batch' if stage=='smoke' else 'source_batch']
    source_plan=read(batch/'plan.json');scores=read(batch/'scores/comparison.json')
    manifest_path=ROOT/source_plan['manifest'];manifest=read(manifest_path)
    expected=5 if stage=='smoke' else 150
    if len(manifest['records'])!=expected or scores['manifest']['sha256']!=sha256_file(manifest_path):
        raise ValueError('Source manifest/budget mismatch')
    if manifest['task']['track']!='length_free' or manifest['task']['partner_chain_context'] or manifest['task']['antigen_context']:
        raise ValueError('Follow-up requires length-free heavy-only inputs')
    ids={r['example_id'] for r in manifest['records']}
    if len(ids)!=expected or manifest['selection']['split']!='validation':
        raise ValueError('Expected unique validation examples')
    for record in manifest['records']:validate_manifest_record(record)
    labels=[r['label'] for r in scores['runs']]
    expected_labels=[a['label'] for a in source_plan['arms']]+['iglm']
    if labels!=expected_labels or len(set(labels))!=len(labels):
        raise ValueError('Scored arm labels differ from source plan')
    files={}
    def freeze(path, expected_hash=None):
        path=Path(os.path.abspath(path));actual=sha256_file(path)
        if expected_hash and actual!=expected_hash:raise ValueError(f'Input changed: {path}')
        files[str(path)]=actual
    # New orchestration may differ, but reused scores require the same metric/judge implementation.
    scoring_sources=['benchmark/scorer.py','benchmark/_common.py','cdrseqgen/eval/metrics.py',
                     'cdrseqgen/datasets/dataset_common.py','cdrseqgen/datasets/dataset_mlm.py',
                     'cdrseqgen/methods/common/backbones.py','cdrseqgen/methods/common/sampling.py',
                     'cdrseqgen/methods/mlm/backbones.py','cdrseqgen/methods/mlm/sampling.py']
    for filename in scoring_sources:
        freeze(ROOT/filename,source_plan['execution_source']['files'][filename])
    runtime=environment()
    for package in ['torch','transformers']:
        if runtime['packages'][package]!=scores['judge'][package+'_version']:
            raise ValueError(f'Judge runtime differs from reused scores: {package}')
    freeze(recipe_path);freeze(batch/'plan.json');freeze(batch/'scores/comparison.json')
    freeze(manifest_path);freeze(ROOT/recipe['corpus'],manifest['source']['processed_file_sha256'])
    freeze(ROOT/recipe['metadata'])
    metadata=read(ROOT/recipe['metadata'])['preprocessing']
    if metadata['split']!='cluster':raise ValueError('Corpus was not split by H3 clusters')
    for run in scores['runs']:
        if {r['example_id'] for r in run['records']}!=ids or len(run['records'])!=expected:
            raise ValueError('Unequal reused framework membership')
        freeze(run['input']['path'],run['input']['sha256'])
        payload=read(run['input']['path'])
        if not payload.get('finished_at') or any(not r['complete'] for r in payload['records']):
            raise ValueError('Source generation is unfinished')
        if payload['sampling']['num_samples']!=source_plan['sampling']['num_samples']:
            raise ValueError('Source attempt budget differs')
        for record in run['records']:
            if record['metrics']['n_samples']!=source_plan['sampling']['num_samples'] or record['metrics']['judge_candidates_scored']!=record['metrics']['n_samples']:
                raise ValueError('Source scored population is incomplete')
    for filename,digest in source_plan['frozen_files'].items():freeze(ROOT/filename,digest)
    for arm in source_plan['arms']:
        run=ROOT/arm['checkpoint'];freeze(run/'train_results.json')
        for state in run.glob('checkpoint-*/trainer_state.json'):freeze(state)
        if (run/'trainer_state.json').exists():freeze(run/'trainer_state.json')
    iglm=read(batch/'generations/iglm.json')
    iglm_config=Path(iglm['model']['checkpoint_file']).parent/'config.json';freeze(iglm_config)
    # Freeze the cached judge revision and main ref: offline loading must resolve the original weights/tokenizer.
    from huggingface_hub.constants import HF_HUB_CACHE
    judge=scores['judge'];revision=judge['model_commit']
    cache=Path(HF_HUB_CACHE)/('models--'+judge['name'].replace('/','--'))
    ref=cache/'refs/main'
    if not ref.exists() or ref.read_text().strip()!=revision:
        raise ValueError('Cached judge main revision differs from source scores; resolve the pinned judge before running')
    freeze(ref)
    snapshot=cache/'snapshots'/revision
    judge_files=[p for p in snapshot.iterdir() if p.is_file()]
    if not any(p.suffix in {'.safetensors','.bin'} for p in judge_files):
        raise ValueError('Original judge weights are not cached')
    for path in judge_files:freeze(path)
    if clusters_input:freeze(clusters_input)
    pairs=planned_pairs(labels+['empirical'])
    return dict(schema_version='cdrseqgen.benchmark_followup.v1',name=recipe['name'],stage=stage,
                recipe=str(recipe_path.resolve()),source_batch=str(batch),manifest=str(manifest_path),
                corpus=str((ROOT/recipe['corpus']).resolve()),metadata=str((ROOT/recipe['metadata']).resolve()),
                source_scores=str(batch/'scores/comparison.json'),iglm_config=str(iglm_config),
                clusters_input=str(Path(clusters_input).resolve()) if clusters_input else None,
                frameworks=expected,reused_arms=len(labels),num_samples=source_plan['sampling']['num_samples'],
                baseline_attempts=expected*source_plan['sampling']['num_samples'],
                sampling_seed=source_plan['sampling']['seed'],cluster_settings=metadata,judge=judge,
                pairs=pairs,analysis=recipe['analysis'],limits=recipe['limits'],
                source=execution_source_identity(),environment=runtime,frozen_files=files)


def validate_plan(plan):
    if plan['source']!=execution_source_identity() or plan['environment']!=environment():
        raise ValueError('Follow-up source/environment changed; preserve this batch and prepare a new version')
    for filename,digest in plan['frozen_files'].items():
        if sha256_file(filename)!=digest:raise ValueError(f'Frozen input changed: {filename}')


def merge_scores(original, added, labels=('empirical',), require_full=True):
    """Append the `labels` runs only after verifying common policy, judge and populations.

    `require_full` demands every attempted candidate be retained and judged, as for the empirical sampler.
    A generator that can return invalid attempts passes False: each framework still needs retained candidates,
    all of them judged, and the shortfall stays visible in the accounting rather than being averaged away.
    """
    if original['manifest']['sha256']!=added['manifest']['sha256']:
        raise ValueError('Scored manifest mismatch')
    for key in ['name','backbone','model_commit','max_len','torch_version','transformers_version']:
        if original['judge'].get(key)!=added['judge'].get(key):raise ValueError(f'Judge mismatch: {key}')
    for key,value in original['scoring'].items():
        if not key.startswith('repository_') and added['scoring'].get(key)!=value:
            raise ValueError(f'Scoring policy mismatch: {key}')
    runs=original['runs']+added['runs']
    if len({r['label'] for r in runs})!=len(runs):raise ValueError('Duplicate arm label')
    if [r['label'] for r in added['runs']]!=list(labels):raise ValueError(f'Expected only {list(labels)} added scores')
    membership={r['example_id'] for r in runs[0]['records']}
    natives={r['example_id']:(r['native'],r['native_length']) for r in runs[0]['records']}
    for run in runs:
        for field in ['track','num_samples','base_seed']:
            if run['sampling'].get(field)!=runs[0]['sampling'].get(field):
                raise ValueError(f'Unmatched sampling: {field}')
        if len(run['records'])!=len(membership) or {r['example_id'] for r in run['records']}!=membership:
            raise ValueError('Unequal scored framework population')
        for record in run['records']:
            if (record['native'],record['native_length'])!=natives[record['example_id']]:
                raise ValueError('Unmatched native truth')
            metrics=record['metrics']
            full=metrics['n_samples']==run['sampling']['num_samples']
            if (require_full and not full) or metrics['judge_candidates_scored']!=metrics['n_samples'] or metrics['n_samples']<=0:
                raise ValueError('Missing candidates or judge coverage')
            for metric in ['edit_similarity_mean','judge_perplexity_mean']:
                value=metrics.get(metric)
                if not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError(f'Missing finite scored metric: {metric}')
    return {**original,'runs':runs,'source_scoring_provenance':original.get('source_scoring_provenance',[original['scoring']])+[added['scoring']],
            'created_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat()}


def summary(output):
    """Read stage completion; no model or analysis execution."""
    if not (output/'plan.json').is_file():
        raise ValueError('No saved follow-up plan in this directory')
    progress=read(output/'progress.json') if (output/'progress.json').exists() else {}
    for name in ['clusters.json','generations/empirical.json','scores/empirical.json','scores/comparison.json','capacity.json','analysis.json','analysis.png','complete.json']:
        path=output/name
        status='pending'
        if path.exists():
            if name=='complete.json':
                artifacts=read(path)['artifacts']
                status='complete' if all((output/item).is_file() and sha256_file(output/item)==digest for item,digest in artifacts.items()) else 'changed'
            elif name in progress:
                status='complete' if sha256_file(path)==progress[name] else 'changed'
            else:
                status='present; not yet sealed'
        print(f'{name}: {status}')


def execute(plan, output, gpu, resume=False):
    output.mkdir(parents=True,exist_ok=resume)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not resume:atomic_write_json(output/'plan.json',plan)
        validate_plan(plan)
        progress_path=output/'progress.json'
        progress=read(progress_path) if progress_path.exists() else {}
        for name,digest in progress.items():
            if not (output/name).exists() or sha256_file(output/name)!=digest:
                raise ValueError(f'Completed follow-up artifact changed: {name}')
        def seal(name):
            progress[name]=sha256_file(output/name)
            atomic_write_json(progress_path,progress)
        manifest=read(plan['manifest']);settings=plan['cluster_settings']
        cluster_path=output/'clusters.json'
        generation_path=output/'generations/empirical.json'
        corpus=None
        if not cluster_path.exists() or not generation_path.exists():
            print('Read frozen corpus; no model execution',flush=True);corpus=read(plan['corpus'])
        if not cluster_path.exists():
            if plan['clusters_input']:
                clusters=read(plan['clusters_input'])
                if clusters['dataset_sha256']!=plan['frozen_files'][plan['corpus']] or clusters['settings']!=settings:
                    raise ValueError('Supplied groups have different corpus/settings')
                if clusters['source']['files']['benchmark/empirical.py']!=plan['source']['files']['benchmark/empirical.py']:
                    raise ValueError('Supplied groups use a different clustering source')
                labels=clusters['cluster_ids']
            else:
                print('Reconstruct ordered H3 representative clusters on CPU',flush=True)
                labels=representative_clusters([r['heavy']['cdr3'] for r in corpus],settings['cluster_identity'],lambda s:print(s,flush=True))
            verification=verify_splits(corpus,labels,settings['test_fraction'],settings['validation_fraction'],settings['seed'])
            clusters=dict(dataset_sha256=plan['frozen_files'][plan['corpus']],settings=settings,
                          cluster_ids=labels,verification=verification,source=plan['source'])
            atomic_write_json(cluster_path,clusters)
        else:
            clusters=read(cluster_path)
            if clusters['source']!=plan['source'] or clusters['dataset_sha256']!=plan['frozen_files'][plan['corpus']] or clusters['settings']!=settings:
                raise ValueError('Saved groups are incompatible with this batch')
        seal('clusters.json')
        if not generation_path.exists():
            print(f'Draw {plan["baseline_attempts"]} training-only empirical loops',flush=True)
            baseline=generate(manifest,corpus,plan['num_samples'],plan['sampling_seed'],
                              dict(path=plan['manifest'],sha256=sha256_file(plan['manifest']),schema_version=manifest['schema_version']),plan['source'])
            atomic_write_json(generation_path,baseline)
        else:
            baseline=read(generation_path)
            if baseline['model']['execution_source']!=plan['source'] or baseline['manifest']['sha256']!=sha256_file(plan['manifest']) or baseline['sampling']['num_samples']!=plan['num_samples']:
                raise ValueError('Saved empirical draws are incompatible')
        seal('generations/empirical.json')
        del corpus
        validate_plan(plan)
        baseline_scores=output/'scores/empirical.json'
        if not baseline_scores.exists():
            command=[sys.executable,str(ROOT/'benchmark/scorer.py'),'--manifest',plan['manifest'],
                     '--input',f'empirical={generation_path}','--output',str(baseline_scores),
                     '--judge',plan['judge']['name'],'--judge-max-len',str(plan['judge']['max_len']),'--device','cuda']
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
            print('Judge only the new empirical candidates; existing neural scores are reused',flush=True)
            subprocess.run(command,cwd=ROOT,env=env,check=True)
        combined_path=output/'scores/comparison.json'
        combined=merge_scores(read(plan['source_scores']),read(baseline_scores))
        seal('scores/empirical.json')
        if not combined_path.exists():atomic_write_json(combined_path,combined)
        else:
            saved=read(combined_path)
            if saved['runs']!=combined['runs'] or saved['manifest']!=combined['manifest']:
                raise ValueError('Saved combined scores changed')
            combined=saved
        seal('scores/comparison.json')
        if not (output/'capacity.json').exists():
            capacity=extract(ROOT,read(Path(plan['source_batch'])/'plan.json'),read(plan['iglm_config']))
            atomic_write_json(output/'capacity.json',capacity)
        seal('capacity.json')
        group_by_id={r['example_id']:clusters['cluster_ids'][r['source_index']] for r in manifest['records']}
        if not (output/'analysis.json').exists():
            print('Compute paired group intervals on CPU',flush=True)
            analysis=analyze(combined,group_by_id,[tuple(pair) for pair in plan['pairs']],plan['analysis']['iterations'],plan['analysis']['seed'])
            analysis['inputs']=dict(scores_sha256=sha256_file(combined_path),clusters_sha256=sha256_file(cluster_path),manifest_sha256=sha256_file(plan['manifest']))
            atomic_write_json(output/'analysis.json',analysis)
        else:
            analysis=read(output/'analysis.json')
            inputs=dict(scores_sha256=sha256_file(combined_path),clusters_sha256=sha256_file(cluster_path),manifest_sha256=sha256_file(plan['manifest']))
            if analysis['inputs']!=inputs:raise ValueError('Saved analysis inputs changed')
        seal('analysis.json')
        (output/'analysis.md').write_text(report(analysis))
        if not (output/'analysis.png').exists():plot(analysis,output/'analysis.png')
        seal('analysis.md');seal('analysis.png')
        validate_plan(plan)
        artifacts=['clusters.json','generations/empirical.json','scores/empirical.json','scores/comparison.json','capacity.json','analysis.json','analysis.md','analysis.png']
        atomic_write_json(output/'complete.json',dict(completed_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
            artifacts={name:sha256_file(output/name) for name in artifacts},baseline_attempts=plan['baseline_attempts']))
        summary(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','smoke','run','status','resume'])
    parser.add_argument('batch',nargs='?',type=Path)
    parser.add_argument('--recipe',type=Path,default=ROOT/'benchmark/recipes/oas4_h3_followup.json')
    parser.add_argument('--stage',choices=['full','smoke'],default='full')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--clusters',type=Path,help='Reuse a completed full-corpus clusters.json; still verify saved split labels')
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    if not args.gpu.isdigit():parser.error('--gpu must be one GPU index')
    if args.command in ['resume','status']:
        if args.batch is None:parser.error('resume/status require a batch directory')
        if args.command=='status':summary(args.batch.resolve());return
        execute(read(args.batch/'plan.json'),args.batch.resolve(),args.gpu,True);return
    if args.batch is not None:parser.error('Use --output for a new batch')
    stage=args.stage if args.command=='plan' else 'smoke' if args.command=='smoke' else 'full'
    plan=build_plan(args.recipe.resolve(),stage,args.clusters)
    print(f'{stage}: reuse {plan["reused_arms"]} scored arms; prepare {len(plan["pairs"])} primary contrasts; generate/judge {plan["baseline_attempts"]} empirical candidates')
    print('CPU: verify original groups, paired intervals and actual capacity/compute; GPU: empirical judge only')
    if args.command=='plan':return
    stamp=datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')
    output=args.output.resolve() if args.output else ROOT/f'results/benchmark/{stamp}_{plan["name"]}_{stage}'
    execute(plan,output,args.gpu)


if __name__=='__main__':
    os.chdir(ROOT)
    main()
