"""Prepare MLM training resources and run method-specific finishing stages.

Entry path:
  train_cli.main -> common.training.run_training_pipeline
                 -> prepare_training -> common.training.train_and_save
                 -> finish_training

Functions:
  * prepare_training         -- seed, load data/model, resolve loss and build Trainer
  * finish_training          -- optionally fit the length head and generate candidates
  * get_datasets             -- train/evaluation subsets and their tokenizer
  * compute_metrics          -- masked-position recovery, imported from mlm.recovery
  * build_trainer            -- MLMLossTrainer, collator, metrics and callbacks

Notes:
  * Training uses its configured masking policy; evaluation masks the full span
    and disables context dropout. Named splits use the shared selection guard.
  * MLMLossTrainer owns the configured external objective. The selected model
    may also train a length head; length fitting requires do_train.
  * train_cli owns setup; common.training owns train/evaluate/save. Recovery
    selects checkpoints; generated candidates are evaluated in a separate stage.
"""
import functools
import logging
from typing import Tuple

import torch
from transformers import set_seed

from cdrseqgen.datasets.dataset_common import (
    TokenizerAdapter, select_split_indices, evaluation_context_view,
)
from cdrseqgen.datasets.dataset_mlm import MaskedInfillingDataset, mlm_collate_fn
from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.methods.common import evaluation as shared_evaluation
from cdrseqgen.methods.mlm.backbones import resolve
from cdrseqgen.methods.mlm.losses import MLMLossTrainer, loss_config_from_args
from cdrseqgen.methods.mlm.recovery import compute_metrics
from cdrseqgen.methods.common.run_setup import update_latest_link

logger = logging.getLogger(__name__)


def prepare_training(model_args, data_args, training_args):
    """Return (Trainer, tokenizer) ready for the shared training lifecycle.

    Steps:
      seed -> backbone -> datasets/tokenizer -> update latest link
           -> model -> loss -> Trainer/callbacks

    Dataset/split validation precedes the latest-link update. The Trainer keeps
    the datasets and loss configuration for finish_training(). Training starts
    when the shared workflow calls train_and_save().
    """
    set_seed(training_args.seed)

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(
        model_args, data_args, training_args, backbone)
    # Validate the data and split before publishing this run as latest.
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

    loss_config = loss_config_from_args(model_args, training_args)
    logger.info(
        "🎯 Loss: %s (core=%g, anchor=%g, trim=%d, label_smoothing=%g)",
        loss_config.loss_type,
        loss_config.core_weight,
        loss_config.anchor_weight,
        loss_config.anchor_trim,
        loss_config.label_smoothing,
    )

    trainer = build_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,
                            callbacks=build_callbacks(model_args, training_args, logger),
                            loss_config=loss_config)
    return trainer, tokenizer


def finish_training(trainer, tokenizer, data_args, gen_args, training_args):
    """Run optional MLM stages on the selected or loaded model after the shared lifecycle.

    do_train + train_length_predictor -> fit the length head
    run_generative_eval              -> evaluate generated loops with that head

    These switches are independent. Evaluation-only commands may still generate
    candidates, using no newly fitted head. This function passes decoder/length
    options directly to common.evaluation.post_training_generation(). Length-fitting
    errors propagate; shared candidate evaluation handles its own generation errors.
    """
    length_predictor = None
    if training_args.do_train and data_args.train_length_predictor:
        from cdrseqgen.methods.mlm.length import fit_length_predictor
        length_predictor = fit_length_predictor(
            trainer.model,
            tokenizer,
            trainer.train_dataset,
            trainer.eval_dataset,
            output_dir=training_args.output_dir,
            device=training_args.device,
            epochs=data_args.length_predictor_epochs,
            learning_rate=data_args.length_predictor_learning_rate,
            batch_size=data_args.length_predictor_batch_size,
            seed=training_args.seed,
        )

    if gen_args.run_generative_eval:
        shared_evaluation.post_training_generation(
            "mlm", trainer, tokenizer, trainer.eval_dataset,
            data_args, gen_args, training_args,
            anchor_trim=trainer.loss_config.anchor_trim,
            method_options={
                "decoding": gen_args.gen_decoding,
                "length_mode": gen_args.gen_length_mode,
                "length_predictor": length_predictor,
                "min_length": gen_args.gen_min_length,
                "max_length": gen_args.gen_max_length,
                "length_temperature": gen_args.gen_length_temperature,
            },
        )


def get_datasets(
    model_args, data_args, training_args, backbone, *, checkpoint_path=None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, TokenizerAdapter]:
    """Return train/evaluation subsets and a tokenizer, sharing parsed records.

    Views:
      training:   configured masking policy and context dropout
      evaluation: full-span masking, context dropout disabled

    Subsets select the configured split indices. This prepares datasets;
    the Trainer creates DataLoaders later. checkpoint_path lets saved-model
    evaluation reuse a checkpoint tokenizer.
    """
    tokenizer = backbone.build_tokenizer(model_args.model_type, checkpoint_path=checkpoint_path)
    masking_policy = getattr(data_args, "masking_policy", "full_span")
    mixed_full_span_probability = getattr(data_args, "mixed_full_span_probability", 0.5)

    full_dataset = MaskedInfillingDataset(
        data_args.data_path,
        tokenizer,
        max_len=data_args.max_len,
        target_region=data_args.target_region,
        include_partner_chain=data_args.include_partner_chain,
        include_antigen=data_args.include_antigen,
        masking_policy=masking_policy,
        mixed_full_span_probability=mixed_full_span_probability,
        context_dropout=data_args.context_dropout,
        context_dropout_seed=training_args.seed,
    )

    train_idx, val_idx, eval_label = select_split_indices(
        full_dataset.data, data_args.eval_split, data_path=data_args.data_path,
        seed=training_args.seed, test_size=data_args.test_size)

    eval_full_dataset = evaluation_context_view(full_dataset.masking_view("full_span"))
    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    eval_ds = torch.utils.data.Subset(eval_full_dataset, val_idx)

    masking_note = f", full-span probability={mixed_full_span_probability:.2f}" \
        if masking_policy == "mixed" else ""

    logger.info(f"📊 Data loaded: {len(train_ds)} train ({masking_policy}{masking_note}), "
                f"{len(eval_ds)} eval (full_span; split: {eval_label}).")
    return train_ds, eval_ds, tokenizer


def build_trainer(model, training_args, train_ds, eval_ds, tokenizer, callbacks=None,
                  *, loss_config):
    """Build the MLM Trainer with the already resolved loss configuration.

    mlm_collate_fn pads inputs and carries supervised target metadata.
    MLMLossTrainer applies the external objective; compute_metrics uses the
    same anchor trim. Trainer creates DataLoaders when training/evaluation starts.
    """
    return MLMLossTrainer(
        model=model,
        args=training_args,
        loss_config=loss_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=functools.partial(mlm_collate_fn, pad_token_id=tokenizer.pad_id),
        compute_metrics=functools.partial(compute_metrics, trim=loss_config.anchor_trim),
        callbacks=callbacks or [],
    )
