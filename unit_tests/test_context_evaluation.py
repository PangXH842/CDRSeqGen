"""Context dropout must affect training only; paired evals share examples and judge context."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from transformers import TrainingArguments
from cdrseqgen.data.dataset_common import evaluation_context_view, framework_for_dataset
from cdrseqgen.methods.registry import resolve
from cdrseqgen.methods.common.evaluation import post_training_generation
from cdrseqgen.methods.common.arguments import BaseGenerationArguments
from cdrseqgen.experiments import completed_artifacts, spec_trials
from cdrseqgen.artifacts import GENERATION_EVAL_FILENAME, CONTEXT_ABLATION_FILENAME
from evaluate import default_output_path

CHAIN = dict(fwr1='AAAA', cdr1='CC', fwr2='DDDD', cdr2='EE', fwr3='FFFF', cdr3='GGGGGG', fwr4='HHHH', full_aa='AAAACCDDDDEEFFFFGGGGGGHHHH')

class ContextEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'data.json'
        self.path.write_text(json.dumps([dict(heavy=CHAIN,light=CHAIN,split=split)
            for split in ['train']*12+['validation']*8+['test']]))

    def datasets(self,method):
        m=resolve(method);args=m.module('arguments')
        data=args.DataArguments(data_path=str(self.path),target_region='h3',max_len=384,
            include_partner_chain=True,context_dropout=1.0,eval_split='validation')
        training=TrainingArguments(output_dir=str(self.root),use_cpu=True,report_to='none',seed=42)
        train,evaluation,tok=m.module('training').get_datasets(args.ModelArguments(),data,training,m.module('backbones').resolve('scratch'))
        return train,evaluation,tok,data,training

    def test_dropout_only_in_training_for_all_methods(self):
        for method in ('mlm','ar','s2s'):
            with self.subTest(method=method):
                train,evaluation,*_=self.datasets(method)
                self.assertIsNone(framework_for_dataset(train,0)['partner_chain'])
                self.assertTrue(all(framework_for_dataset(evaluation,i)['partner_chain'] for i in range(len(evaluation))))
                self.assertIs(train.dataset.data,evaluation.dataset.data)
                self.assertIsNot(train.dataset,evaluation.dataset)
                self.assertEqual(train.dataset.context_dropout,1.0)

    def test_nested_views_preserve_identity_and_do_not_mutate(self):
        _,evaluation,*_=self.datasets('s2s')
        subset=torch.utils.data.Subset(evaluation,[2,0])
        without=evaluation_context_view(subset,include_partner_chain=False)
        self.assertEqual(without.indices,[2,0])
        self.assertEqual(without.dataset.indices,evaluation.indices)
        self.assertIsNone(framework_for_dataset(without,0)['partner_chain'])
        self.assertIsNotNone(framework_for_dataset(subset,0)['partner_chain'])
        self.assertEqual(framework_for_dataset(without,0)['target'],framework_for_dataset(subset,0)['target'])

    def test_paired_post_training_eval_uses_fixed_judge_and_seed(self):
        _,evaluation,tok,data,training=self.datasets('s2s')
        gen=BaseGenerationArguments(gen_eval_context_ablation=True)
        judge=object();calls=[]
        def evaluate(model,tokenizer,dataset,**kwargs):
            calls.append((dataset,kwargs));return 'primary' if len(calls)==1 else 'ablation'
        with patch('cdrseqgen.methods.common.evaluation.build_judge',return_value=(object(),object(),judge)) as build, patch('cdrseqgen.methods.common.evaluation.resolve',return_value=NS(evaluate=evaluate)):
            result=post_training_generation('s2s',NS(model=object()),tok,evaluation,data,gen,training,anchor_trim=2)
        self.assertEqual(result,'primary');self.assertEqual(len(calls),2);build.assert_called_once()
        self.assertTrue(build.call_args.kwargs['include_partner_chain'])
        for index,(ds,kws) in enumerate(calls):
            self.assertEqual(ds.indices,evaluation.indices)
            self.assertEqual(bool(framework_for_dataset(ds,0)['partner_chain']),index==0)
            self.assertIs(kws['judge_dataset'],judge);self.assertEqual(kws['seed'],42)
            self.assertTrue(kws['meta']['judge_include_partner_chain'])
            self.assertEqual(kws['meta']['eval_context_dropout'],0)
        self.assertNotEqual(calls[0][1]['output_path'],calls[1][1]['output_path'])

    def test_completion_requires_both_context_outputs(self):
        config=dict(gen_eval_context_ablation=True,run_generative_eval=True)
        (self.root/'run_config.json').write_text(json.dumps(config))
        (self.root/'epoch_metrics.csv').write_text('epoch,eval_loss\n1,2\n')
        payload=json.dumps(dict(records=[dict(eval_index=0)],summary=dict(aar_mean=.2)))
        (self.root/GENERATION_EVAL_FILENAME).write_text(payload)
        self.assertFalse(completed_artifacts(self.root,config))
        (self.root/CONTEXT_ABLATION_FILENAME).write_text(payload)
        self.assertTrue(completed_artifacts(self.root,config))

    def test_output_identity_includes_both_context_choices(self):
        args=NS(model_dir='x',length_mode='inferred',min_length=None,max_length=None,length_temperature=1.,decoding='causal',num_examples=150,stratify=False,num_samples=20,temperature=1.,top_k=0,top_p=.9,seed=42,skip_perplexity=False,judge='IgBert',judge_path=None,partner_context='with',judge_partner_context='with')
        first=default_output_path(args);args.partner_context='without';second=default_output_path(args)
        args.judge_partner_context='without';third=default_output_path(args)
        self.assertEqual(len({first,second,third}),3)

    def test_spec_pairs_differ_only_in_training_dropout(self):
        trials,_=spec_trials('configs/experiments/20260915_context_dropout.json',{})
        self.assertEqual(len(trials),12)
        for i in range(0,len(trials),2):
            a,b=(dict(t['config']) for t in trials[i:i+2])
            self.assertEqual(a.pop('context_dropout'),0.)
            self.assertEqual(b.pop('context_dropout'),.5)
            self.assertEqual(a,b)

if __name__=='__main__':unittest.main()
