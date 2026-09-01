"""Masked-LM training entrypoint. Invoked by the top-level ``train.py`` dispatcher.

Kept in the method package because it is the part that genuinely diverges: it builds an
``MLMLossTrainer`` around an external loss registry, fits the gap-sentinel length head, and runs
the fixed-native generative evaluation. None of that exists on the causal side.
"""
import functools

from transformers import set_seed

from cdrseqgen.data.dataset_mlm import collate_fn
from cdrseqgen.methods.mlm.arguments import setup_args
from cdrseqgen.methods.mlm.backbones import resolve
from cdrseqgen.methods.mlm.losses import MLMLossTrainer, loss_config_from_args
from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.methods.mlm.recovery import compute_metrics
from cdrseqgen.methods.mlm.training import (
    get_datasets,
    post_training_generation,
    run_training_pipeline,
)
from cdrseqgen.train.logging import LogMetricsCallback, configure_logging, update_latest_link

configure_logging()


def main():
    import logging
    logger = logging.getLogger(__name__)

    model_args, data_args, gen_args, training_args = setup_args()
    set_seed(training_args.seed)
    loss_config = loss_config_from_args(model_args, training_args)
    logger.info(
        "🎯 Loss: %s (core=%g, anchor=%g, trim=%d, label_smoothing=%g)",
        loss_config.loss_type,
        loss_config.core_weight,
        loss_config.anchor_weight,
        loss_config.anchor_trim,
        loss_config.label_smoothing,
    )

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(model_args, data_args, training_args, backbone)
    # Only now is this a real run: a rejected config (an undeclared eval_split, say) must not take
    # results/runs/latest away from the last batch that actually produced results.
    update_latest_link(training_args.output_dir)

    # load_training_model ignores the scratch-only kwargs for pretrained backbones (**kw)
    model = backbone.load_training_model(
        model_args.model_type,
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_args.hidden_size,
        num_layers=model_args.num_layers,
        num_attention_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        max_len=data_args.max_len,
    )

    trainer = MLMLossTrainer(
        model=model,
        args=training_args,
        loss_config=loss_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=functools.partial(collate_fn, pad_token_id=tokenizer.pad_id),
        compute_metrics=functools.partial(compute_metrics, trim=loss_config.anchor_trim),
        callbacks=build_callbacks(model_args, training_args, logger),
    )

    run_training_pipeline(trainer, training_args)

    length_predictor = None
    if training_args.do_train and data_args.train_length_predictor:
        from cdrseqgen.methods.mlm.length_predictor import fit_length_predictor
        length_predictor = fit_length_predictor(
            trainer.model,
            tokenizer,
            train_dataset,
            eval_dataset,
            output_dir=training_args.output_dir,
            device=training_args.device,
            epochs=data_args.length_predictor_epochs,
            learning_rate=data_args.length_predictor_learning_rate,
            batch_size=data_args.length_predictor_batch_size,
            seed=training_args.seed,
        )

    if gen_args.run_generative_eval:
        post_training_generation(
            trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
            anchor_trim=loss_config.anchor_trim,
            length_predictor=length_predictor,
        )
