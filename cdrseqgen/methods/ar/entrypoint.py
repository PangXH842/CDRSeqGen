"""Autoregressive training entrypoint. Invoked by the top-level ``train.py`` dispatcher.

Kept in the method package for the same reason as the masked one: a plain ``Trainer`` (the model
returns its own loss), no length head, and a decode-to-``<eos>`` generative evaluation.
"""
import logging

from transformers import set_seed

from cdrseqgen.methods.ar.arguments import setup_args
from cdrseqgen.methods.ar.backbones import resolve
from cdrseqgen.methods.ar.training import build_causal_trainer, get_datasets
from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.train.logging import configure_logging, update_latest_link
from cdrseqgen.train.record_results import plot_training_curves, save_training_csv

logger = logging.getLogger(__name__)


def post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                             anchor_trim):
    """Sample from the finished best model and write generation_eval.json.

    Exception-guarded, like the masked side: a generation failure must never damage a training run
    that has already finished and saved.
    """
    from cdrseqgen.eval.generative import build_judge
    from cdrseqgen.methods.ar.evaluation import run_causal_generative_eval
    import os

    judge_model = judge_tokenizer = judge_dataset = None
    if gen_args.gen_judge:
        judge_model, judge_tokenizer, judge_dataset = build_judge(
            gen_args.gen_judge, gen_args.gen_judge_weights,
            data_path=data_args.data_path, max_len=data_args.max_len,
            target_region=data_args.target_region,
            include_partner_chain=data_args.include_partner_chain,
            include_antigen=data_args.include_antigen, device=training_args.device)

    run_causal_generative_eval(
        trainer.model, tokenizer, eval_dataset, device=training_args.device,
        judge_dataset=judge_dataset, judge_model=judge_model, judge_tokenizer=judge_tokenizer,
        num_examples=gen_args.gen_num_examples, num_samples=gen_args.gen_num_samples,
        temperature=gen_args.gen_temperature, top_k=gen_args.gen_top_k, top_p=gen_args.gen_top_p,
        max_new_tokens=gen_args.gen_max_new_tokens, stratify=gen_args.gen_stratify,
        seed=training_args.seed, skip_perplexity=gen_args.gen_skip_perplexity,
        output_path=os.path.join(training_args.output_dir, "generation_eval.json"),
        title=training_args.run_name, anchor_trim=anchor_trim)


def main():
    configure_logging()
    model_args, data_args, gen_args, training_args = setup_args()
    set_seed(training_args.seed)

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(
        model_args, data_args, training_args, backbone)
    # Only now: everything above can succeed for a config get_datasets then rejects.
    update_latest_link(training_args.output_dir)

    model = backbone.load_training_model(
        model_args.model_type,
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_args.hidden_size,
        num_layers=model_args.num_layers,
        num_attention_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        max_len=data_args.max_len,
    )
    trainer = build_causal_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,
                                   callbacks=build_callbacks(model_args, training_args, logger),
                                   trim=model_args.anchor_trim)

    if training_args.do_train:
        trainer.train()
        trainer.save_model(training_args.output_dir)
    if training_args.do_eval:
        trainer.evaluate()

    save_training_csv(trainer, training_args.output_dir)
    plot_training_curves(training_args.output_dir)

    if gen_args.run_generative_eval:
        try:
            post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args,
                                     training_args, anchor_trim=model_args.anchor_trim)
        except Exception:
            logger.warning("⚠️ Generative evaluation failed; the trained run is unaffected.",
                           exc_info=True)
