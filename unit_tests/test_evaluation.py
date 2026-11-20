"""Tests merged from: test_context_evaluation.py, test_eval_split_selection.py, test_select_indices.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from transformers import TrainingArguments
from cdrseqgen.datasets.dataset_common import evaluation_context_view, framework_for_dataset
from cdrseqgen.methods.registry import resolve
from cdrseqgen.methods.common.evaluation import post_training_generation
from cdrseqgen.methods.common.arguments import BaseGenerationArguments, validate_common_args
from experiments.core import completed_artifacts
from experiments.spec import spec_trials
from cdrseqgen.artifacts import GENERATION_EVAL_FILENAME, CONTEXT_ABLATION_FILENAME
from evaluate import default_output_path
from types import SimpleNamespace
from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.methods.mlm.training import get_datasets
import random
from cdrseqgen.eval.generative import bucket_of, select_indices


# ====================================================================================================
# from test_context_evaluation.py
# Context dropout must affect training only; paired evals share examples and judge context.
# ====================================================================================================

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

    def test_ar_and_s2s_finish_forward_decode_cap_for_evaluation_only(self):
        trainer = NS(model=object(), eval_dataset=object())
        tokenizer, data = object(), object()
        training = NS(do_train=False)
        for method in ('ar', 's2s'):
            with self.subTest(method=method):
                gen = NS(run_generative_eval=False, gen_max_new_tokens=37)
                finish = resolve(method).module('training').finish_training
                with patch('cdrseqgen.methods.common.evaluation.post_training_generation') as generate:
                    finish(trainer, tokenizer, data, gen, training, anchor_trim=3)
                    generate.assert_not_called()
                    gen.run_generative_eval = True
                    finish(trainer, tokenizer, data, gen, training, anchor_trim=3)
                generate.assert_called_once_with(
                    method, trainer, tokenizer, trainer.eval_dataset, data, gen, training,
                    anchor_trim=3, method_options={'max_new_tokens': 37})

    def test_completion_requires_both_context_outputs(self):
        config=dict(gen_eval_context_ablation=True,run_generative_eval=True)
        (self.root/'run_config.json').write_text(json.dumps(config))
        (self.root/'epoch_metrics.csv').write_text('epoch,eval_loss\n1,2\n')
        payload=json.dumps(dict(records=[dict(eval_index=0)],summary=dict(aar_mean=.2)))
        (self.root/GENERATION_EVAL_FILENAME).write_text(payload)
        self.assertFalse(completed_artifacts(self.root,config))
        (self.root/CONTEXT_ABLATION_FILENAME).write_text(payload)
        self.assertTrue(completed_artifacts(self.root,config))

    def test_no_partner_generators_can_keep_partner_context_for_the_judge(self):
        for method in ('mlm', 'ar', 's2s'):
            with self.subTest(method=method):
                _, evaluation, tok, data, training = self.datasets(method)
                data.include_partner_chain = False
                evaluation = evaluation_context_view(evaluation, include_partner_chain=False)
                gen = BaseGenerationArguments(gen_judge_partner_context='with')
                calls = []
                def evaluate(model, tokenizer, dataset, **kwargs):
                    calls.append((dataset, kwargs))
                    return 'control'
                with patch('cdrseqgen.methods.common.evaluation.build_judge', return_value=(object(), object(), object())) as build, patch('cdrseqgen.methods.common.evaluation.resolve', return_value=NS(evaluate=evaluate)):
                    result = post_training_generation(method, NS(model=object()), tok, evaluation,
                        data, gen, training, anchor_trim=2)
                self.assertEqual(result, 'control')
                build.assert_called_once()
                self.assertTrue(build.call_args.kwargs['include_partner_chain'])
                self.assertEqual(len(calls), 1)
                dataset, kwargs = calls[0]
                self.assertIsNone(framework_for_dataset(dataset, 0)['partner_chain'])
                self.assertFalse(kwargs['meta']['training_include_partner_chain'])
                self.assertFalse(kwargs['meta']['include_partner_chain'])
                self.assertTrue(kwargs['meta']['judge_include_partner_chain'])

    def test_judge_override_does_not_change_generator_ablation(self):
        _, evaluation, tok, data, training = self.datasets('s2s')
        gen = BaseGenerationArguments(gen_eval_context_ablation=True,
                                      gen_judge_partner_context='without')
        calls = []
        with patch('cdrseqgen.methods.common.evaluation.build_judge', return_value=(object(), object(), object())) as build, patch('cdrseqgen.methods.common.evaluation.resolve', return_value=NS(evaluate=lambda *args, **kwargs: calls.append(kwargs))):
            post_training_generation('s2s', NS(model=object()), tok, evaluation,
                data, gen, training, anchor_trim=2)
        self.assertFalse(build.call_args.kwargs['include_partner_chain'])
        self.assertEqual([call['meta']['include_partner_chain'] for call in calls], [True, False])
        self.assertTrue(all(not call['meta']['judge_include_partner_chain'] for call in calls))

    def test_invalid_judge_context_fails_before_training(self):
        args = resolve('s2s').module('arguments')
        with self.assertRaisesRegex(ValueError, 'gen_judge_partner_context'):
            validate_common_args(args.ModelArguments(), args.DataArguments(),
                BaseGenerationArguments(gen_judge_partner_context='typo'), expected_method='s2s')

    def test_completion_accepts_new_warmup_storage_and_rejects_changed_schedule(self):
        config = dict(warmup_ratio=.06, warmup_steps=0, run_generative_eval=False)
        saved = dict(warmup_steps=.06, run_generative_eval=False)
        (self.root/'run_config.json').write_text(json.dumps(saved))
        (self.root/'epoch_metrics.csv').write_text('epoch,eval_loss\n1,2\n')
        self.assertTrue(completed_artifacts(self.root, config))
        saved['warmup_steps'] = .02
        (self.root/'run_config.json').write_text(json.dumps(saved))
        self.assertFalse(completed_artifacts(self.root, config))
        config['warmup_steps'] = 12
        saved['warmup_steps'] = 12
        (self.root/'run_config.json').write_text(json.dumps(saved))
        self.assertTrue(completed_artifacts(self.root, config))

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


# ====================================================================================================
# from test_eval_split_selection.py
# ====================================================================================================

def _chain(cdr1="CDEF"):
    regions = {
        "fwr1": "AA",
        "cdr1": cdr1,
        "fwr2": "GG",
        "cdr2": "HI",
        "fwr3": "KK",
        "cdr3": "LMN",
        "fwr4": "PP",
    }
    regions["full_aa"] = "".join(regions.values())
    return regions


class _Backbone:
    def build_tokenizer(self, _model_type, checkpoint_path=None):
        return TokenizerAdapter.from_custom(AntibodyTokenizer())


class EvalSplitSelectionTests(unittest.TestCase):
    """The eval split drives early stopping and load_best_model_at_end, so leaving it implicit
    silently selects checkpoints on whatever non-train partition the dataset happens to carry."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _data_args(self, splits, eval_split):
        path = self.dir / f"{'_'.join(sorted(set(splits)))}.json"
        path.write_text(json.dumps([
            {"heavy": _chain(), "light": _chain("QRST"), "split": split} for split in splits
        ]))
        return SimpleNamespace(
            data_path=str(path),
            max_len=64,
            test_size=0.5,
            eval_split=eval_split,
            target_region="h1",
            include_partner_chain=False,
            include_antigen=False,
            context_dropout=0.0,
            masking_policy="full_span",
            mixed_full_span_probability=0.5,
        )

    def _call(self, splits, eval_split):
        return get_datasets(
            SimpleNamespace(model_type="scratch"),
            self._data_args(splits, eval_split),
            SimpleNamespace(seed=42),
            _Backbone(),
        )

    def test_unset_eval_split_is_refused_when_a_test_partition_exists(self):
        with self.assertRaisesRegex(ValueError, "eval_split is unset"):
            self._call(["train", "train", "test"], "")

    def test_unset_eval_split_is_refused_on_a_three_way_split(self):
        with self.assertRaisesRegex(ValueError, "eval_split is unset"):
            self._call(["train", "validation", "test"], "")

    def test_validation_is_selected_without_touching_test(self):
        _, evaluation, _ = self._call(["train", "validation", "test"], "validation")
        self.assertEqual(len(evaluation), 1)
        self.assertEqual(evaluation.dataset.data[evaluation.indices[0]]["split"], "validation")

    def test_test_split_remains_available_once_declared(self):
        _, evaluation, _ = self._call(["train", "validation", "test"], "test")
        self.assertEqual(evaluation.dataset.data[evaluation.indices[0]]["split"], "test")

    def test_fallback_survives_for_datasets_with_no_test_partition(self):
        train, evaluation, _ = self._call(["train", "train", "validation"], "")
        self.assertEqual(len(train), 2)
        self.assertEqual(len(evaluation), 1)


# ====================================================================================================
# from test_select_indices.py
# ====================================================================================================

class SelectIndicesTests(unittest.TestCase):
    """Guards the draw that decides which frameworks every generative number is computed on.

    An under-populated length bucket cannot fill its quota, and without redistribution the draw
    silently returns fewer than k examples. That is not hypothetical: eight runs of the 07-29 batch
    had their generative metrics computed on ~half the intended sample before it was fixed, and the
    failure is invisible in the output because the summary just reports a smaller n.
    """

    def _skewed_lengths(self):
        # 749 short loops, 1 medium, 0 long — the SD2 H1 shape that caused the original incident
        return [5] * 749 + [14]

    def test_a_starved_bucket_does_not_shrink_the_draw(self):
        picked = select_indices(self._skewed_lengths(), 150, True, random.Random(42))
        self.assertEqual(len(picked), 150)

    def test_stratified_draw_is_still_without_replacement(self):
        picked = select_indices(self._skewed_lengths(), 150, True, random.Random(42))
        self.assertEqual(len(set(picked)), len(picked))
        self.assertTrue(all(0 <= i < 750 for i in picked))

    def test_stratification_still_over_represents_the_rare_bucket(self):
        """The point of stratifying: the lone medium loop must survive a 150-of-750 draw."""
        lengths = self._skewed_lengths()
        picked = select_indices(lengths, 150, True, random.Random(42))
        self.assertIn("medium", {bucket_of(lengths[i]) for i in picked})

    def test_balanced_data_fills_every_bucket(self):
        lengths = [5] * 100 + [14] * 100 + [25] * 100
        picked = select_indices(lengths, 30, True, random.Random(0))
        self.assertEqual(len(picked), 30)
        self.assertEqual({bucket_of(lengths[i]) for i in picked}, {"short", "medium", "long"})

    def test_unstratified_draw_returns_exactly_k(self):
        picked = select_indices([5] * 749 + [14], 150, False, random.Random(42))
        self.assertEqual(len(picked), 150)
        self.assertEqual(len(set(picked)), 150)

    def test_requesting_everything_returns_every_index_in_order(self):
        lengths = [5, 14, 25, 9]
        for k in (-1, 4, 99):
            with self.subTest(k=k):
                self.assertEqual(select_indices(lengths, k, True, random.Random(1)), [0, 1, 2, 3])

    def test_selection_is_reproducible_for_a_seed(self):
        lengths = self._skewed_lengths()
        first = select_indices(lengths, 150, True, random.Random(7))
        second = select_indices(lengths, 150, True, random.Random(7))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
