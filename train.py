"""Training entrypoint, processes configs from batch jsons/CLI, loads functions from `cdrseqgen/train/`.

Usage (in CLI):
  * python train.py configs/h3_scratch.json
  * python train.py configs/h3_scratch.json --learning_rate 3e-4 --num_train_epochs 20

Trailing flags override the JSON rather than replacing it, so any TrainingArguments field is tunable
without writing a new config. Batch rounds go through scripts/run.sh, which exports
CDRSEQGEN_RESULTS_ROOT so every run of the round shares one batch folder.
"""
import functools

from transformers import Trainer, set_seed

from cdrseqgen.data.dataset import collate_fn
from cdrseqgen.models.registry import resolve
from cdrseqgen.eval.recovery import compute_metrics
from cdrseqgen.train.arguments import setup_args
from cdrseqgen.train.pipeline import get_datasets, post_training_generation, run_training_pipeline
from cdrseqgen.train.logging import LogMetricsCallback, configure_logging

configure_logging()


def build_callbacks(model_args, training_args, logger):
    """Trainer callbacks: metric logging always, early stopping only when a config asks for it."""
    callbacks = [LogMetricsCallback()]  # Required for train.log to record training process

    if model_args.early_stopping_patience and model_args.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=model_args.early_stopping_patience,
            early_stopping_threshold=model_args.early_stopping_threshold,
        ))
        logger.info(f"⏱️  Early stopping ON: patience={model_args.early_stopping_patience} evals on "
                    f"'{training_args.metric_for_best_model}' (threshold {model_args.early_stopping_threshold}).")
    return callbacks


def main():
    import logging
    logger = logging.getLogger(__name__)

    model_args, data_args, gen_args, training_args = setup_args()
    set_seed(training_args.seed)

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(model_args, data_args, training_args, backbone)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=functools.partial(collate_fn, pad_token_id=tokenizer.pad_id),
        compute_metrics=compute_metrics,
        callbacks=build_callbacks(model_args, training_args, logger),
    )

    run_training_pipeline(trainer, training_args)

    if gen_args.run_generative_eval:
        post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args)


if __name__ == "__main__":
    main()
