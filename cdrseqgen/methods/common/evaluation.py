"""Post-training evaluation setup, shared across methods without sharing decoding assumptions."""
import logging
import os

from cdrseqgen import artifacts
from cdrseqgen.eval.generative import build_judge
from cdrseqgen.methods.registry import resolve

logger = logging.getLogger(__name__)


def post_training_generation(method, trainer, tokenizer, eval_dataset, data_args, gen_args,
                             training_args, *, anchor_trim, method_options=None):
    """Evaluate the saved model; preserve training artifacts and log any evaluation failure."""
    try:
        judge_model = judge_tokenizer = judge_dataset = None
        if gen_args.gen_judge:
            judge_model, judge_tokenizer, judge_dataset = build_judge(
                gen_args.gen_judge, gen_args.gen_judge_path,
                data_path=data_args.data_path, max_len=data_args.max_len,
                target_region=data_args.target_region,
                include_partner_chain=data_args.include_partner_chain,
                include_antigen=data_args.include_antigen, device=training_args.device)
        return resolve(method).evaluate(
            trainer.model, tokenizer, eval_dataset, device=training_args.device,
            num_examples=gen_args.gen_num_examples, num_samples=gen_args.gen_num_samples,
            temperature=gen_args.gen_temperature, top_k=gen_args.gen_top_k, top_p=gen_args.gen_top_p,
            stratify=gen_args.gen_stratify, seed=gen_args.gen_seed,
            skip_perplexity=gen_args.gen_skip_perplexity,
            judge_model=judge_model, judge_tokenizer=judge_tokenizer, judge_dataset=judge_dataset,
            judge_name=gen_args.gen_judge, anchor_trim=anchor_trim,
            output_path=os.path.join(training_args.output_dir, artifacts.GENERATION_EVAL_FILENAME),
            title=f'{os.path.basename(training_args.output_dir)}   target={data_args.target_region}',
            meta={'model_dir': training_args.output_dir, 'method': method,
                  'target_region': data_args.target_region,
                  'include_partner_chain': bool(data_args.include_partner_chain),
                  'include_antigen': bool(data_args.include_antigen)},
            **(method_options or {}))
    except Exception:
        logger.exception('Generative evaluation failed; the trained model remains saved.')
        return None
