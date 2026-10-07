"""Autoregressive training command. Invoked by the top-level ``train.py`` dispatcher.

Kept in the method package for the same reason as the masked one: a plain ``Trainer`` (the model
returns its own loss), no length head, and a decode-to-``<eos>`` generative evaluation.
"""
import logging

from transformers import set_seed

from cdrseqgen.methods.ar.arguments import setup_args
from cdrseqgen.methods.ar.backbones import resolve, record_training_layout
from cdrseqgen.methods.ar.training import build_causal_trainer, get_datasets
from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.train.logging import configure_logging, update_latest_link
from cdrseqgen.methods.common.training import run_training_pipeline

logger = logging.getLogger(__name__)


def post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                             anchor_trim):
    from cdrseqgen.methods.common.evaluation import post_training_generation as evaluate
    return evaluate('ar', trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                    anchor_trim=anchor_trim,
                    method_options={'max_new_tokens': gen_args.gen_max_new_tokens})


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
    record_training_layout(model, tokenizer, training_args.output_dir)
    trainer = build_causal_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,
                                   callbacks=build_callbacks(model_args, training_args, logger),
                                   trim=model_args.anchor_trim)

    run_training_pipeline(trainer, training_args, tokenizer=tokenizer)

    if gen_args.run_generative_eval:
        post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args,
                                 training_args, anchor_trim=model_args.anchor_trim)
