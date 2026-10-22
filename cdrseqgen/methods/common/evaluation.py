"""Set up post-training generated-loop evaluation for MLM, AR and S2S.

Caller:
  A method's training workflow calls post_training_generation after training/saving.

Workflow:
  optional judge + evaluation context views -> registry-selected method evaluator
                                           -> payloads in the run directory

This module chooses context regimes and handles evaluation failures. Method
evaluators produce candidates; cdrseqgen.eval supplies shared measurements and reports.
"""
import logging
import os

from cdrseqgen import artifacts
from cdrseqgen.datasets.dataset_common import evaluation_context_view
from cdrseqgen.eval.generative import build_judge
from cdrseqgen.methods.registry import resolve

logger = logging.getLogger(__name__)


def post_training_generation(method, trainer, tokenizer, eval_dataset, data_args, gen_args,
                             training_args, *, anchor_trim, method_options=None):
    """Evaluate trainer.model and return the primary payload, or None on failure.

    Steps:
      1. Load the optional judge with its requested partner-context setting.
      2. Evaluate with the training partner-context setting and dropout disabled.
      3. If requested, also evaluate with partner context removed from the generator.

    The same judge is reused across context regimes. Method-specific options are
    forwarded to the evaluator, which receives trainer.model without reloading a
    checkpoint. Evaluation errors are logged; previously saved training artifacts remain.
    """
    try:
        judge_context = getattr(gen_args, 'gen_judge_partner_context', 'trained')
        judge_partner = (bool(data_args.include_partner_chain) if judge_context == 'trained'
                         else judge_context == 'with')
        judge_model = judge_tokenizer = judge_dataset = None
        if gen_args.gen_judge:
            judge_model, judge_tokenizer, judge_dataset = build_judge(
                gen_args.gen_judge, gen_args.gen_judge_path,
                data_path=data_args.data_path, max_len=data_args.max_len,
                target_region=data_args.target_region,
                include_partner_chain=judge_partner,
                include_antigen=data_args.include_antigen, device=training_args.device)
        regimes = [(bool(data_args.include_partner_chain), artifacts.GENERATION_EVAL_FILENAME)]
        if getattr(gen_args, 'gen_eval_context_ablation', False):
            regimes.append((False, artifacts.CONTEXT_ABLATION_FILENAME))
        primary = None
        for partner, filename in regimes:
            context_dataset = evaluation_context_view(eval_dataset, include_partner_chain=partner)
            result = resolve(method).evaluate(
                trainer.model, tokenizer, context_dataset, device=training_args.device,
                num_examples=gen_args.gen_num_examples, num_samples=gen_args.gen_num_samples,
                temperature=gen_args.gen_temperature, top_k=gen_args.gen_top_k, top_p=gen_args.gen_top_p,
                stratify=gen_args.gen_stratify, seed=gen_args.gen_seed,
                skip_perplexity=gen_args.gen_skip_perplexity,
                judge_model=judge_model, judge_tokenizer=judge_tokenizer, judge_dataset=judge_dataset,
                judge_name=gen_args.gen_judge, anchor_trim=anchor_trim,
                output_path=os.path.join(training_args.output_dir, filename),
                title=f'{os.path.basename(training_args.output_dir)}   target={data_args.target_region}',
                meta={'model_dir': training_args.output_dir, 'method': method,
                      'target_region': data_args.target_region,
                      'include_partner_chain': partner,
                      'training_include_partner_chain': bool(data_args.include_partner_chain),
                      'training_context_dropout': data_args.context_dropout,
                      'eval_context_dropout': 0.0,
                      'judge_include_partner_chain': judge_partner,
                      'include_antigen': bool(data_args.include_antigen)},
                **(method_options or {}))
            if filename == artifacts.GENERATION_EVAL_FILENAME:
                primary = result
        return primary
    except Exception:
        logger.exception('Generative evaluation failed; the trained model remains saved.')
        return None
