"""Encoder-decoder training command. Invoked by the top-level ``train.py`` dispatcher.

Kept in the method package for the same reason as its siblings: a plain ``Trainer`` (the model
returns its own loss), no length head, and a decode-to-``<eos>`` generative evaluation.
"""
import logging

from transformers import set_seed

from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.methods.common.training import run_training_pipeline
from cdrseqgen.methods.s2s.arguments import setup_args
from cdrseqgen.methods.s2s.backbones import resolve
from cdrseqgen.methods.s2s.training import build_seq2seq_trainer, get_datasets
from cdrseqgen.methods.common.logging import configure_logging
from cdrseqgen.methods.common.run_setup import update_latest_link

logger = logging.getLogger(__name__)


def post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                             anchor_trim):
    from cdrseqgen.methods.common.evaluation import post_training_generation as evaluate
    return evaluate('s2s', trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
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
        num_encoder_layers=model_args.num_encoder_layers,
        num_decoder_layers=model_args.num_decoder_layers,
        encoder_init=model_args.encoder_init,
        freeze_encoder=model_args.freeze_encoder,
    )
    trainer = build_seq2seq_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,
                                   callbacks=build_callbacks(model_args, training_args, logger),
                                   trim=model_args.anchor_trim)

    if model_args.conditioning_probe_steps or model_args.diagnostic_stop_epoch:
        from cdrseqgen.methods.s2s.diagnostics import ConditioningProbe
        trainer.add_callback(ConditioningProbe(
            eval_dataset, trainer.data_collator, model_args.conditioning_probe_steps,
            model_args.diagnostic_stop_epoch))

    run_training_pipeline(trainer, training_args, tokenizer=tokenizer)

    if gen_args.run_generative_eval:
        post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args,
                                 training_args, anchor_trim=model_args.anchor_trim)
