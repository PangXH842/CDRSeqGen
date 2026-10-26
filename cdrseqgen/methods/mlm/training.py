"""MLM training workflow and its dataset/evaluation helpers.

Caller:
  mlm.train_cli.main() -> run_training(model_args, data_args, gen_args, training_args)

Workflow:
  set seed -> resolve backbone -> get_datasets()
    -> load model -> configure loss -> construct Trainer/callbacks
    -> common.training.run_training_pipeline()
    -> fit length predictor when requested -> generate/evaluate when requested

Functions:
  * run_training -- execute the configured MLM workflow; return its Trainer
  * get_datasets -- return training/evaluation subsets and their tokenizer
  * post_training_generation -- pass MLM options into common.evaluation

Notes:
  * train_cli owns logging and argument setup, including run-directory creation.
  * Dataset setup respects named splits. Unlabelled data uses a random fallback;
    this helper does not create homology-controlled splits.
  * Evaluation uses full-span masking and disables training context dropout.
  * The common lifecycle handles do_train/do_eval, saving and metric exports.
    Length fitting requires do_train and train_length_predictor; generation
    requires run_generative_eval. Common evaluation logs generation failures.
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
from cdrseqgen.methods.common.training import run_training_pipeline
from cdrseqgen.methods.mlm.backbones import resolve
from cdrseqgen.methods.mlm.losses import MLMLossTrainer, loss_config_from_args
from cdrseqgen.methods.mlm.recovery import compute_metrics
from cdrseqgen.methods.common.run_setup import update_latest_link

logger = logging.getLogger(__name__)


def run_training(model_args, data_args, gen_args, training_args):
    """Build and run MLM training from the four resolved argument groups.

    Steps:
      * 1. Set the seed before constructing datasets or model weights.
      * 2. Resolve the backbone and call get_datasets(); update latest only
           after dataset/split validation succeeds.
      * 3. Load the model, resolve the loss and build the Trainer/callbacks.
      * 4. Delegate training, validation and saving to run_training_pipeline().
      * 4A. Optionally fit the length head on the resulting Trainer model.
      * 5. Optionally run generated-sequence evaluation with that length head.

    Returns the Trainer after all requested stages. Exceptions propagate from
    setup/training/length fitting; the shared generation wrapper handles its
    own failures after training artifacts have been saved.
    """
    set_seed(training_args.seed)

    # 2.1 Load model type
    backbone = resolve(model_args.model_type)

    # 2.2 Load datasets
    train_dataset, eval_dataset, tokenizer = get_datasets(
        model_args, data_args, training_args, backbone)

    # (1.1) Update latest link for logging
    update_latest_link(training_args.output_dir)

    # 3.1 Prepare model
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

    # 3.2 Configure loss
    loss_config = loss_config_from_args(model_args, training_args)
    logger.info(
        "🎯 Loss: %s (core=%g, anchor=%g, trim=%d, label_smoothing=%g)",
        loss_config.loss_type,
        loss_config.core_weight,
        loss_config.anchor_weight,
        loss_config.anchor_trim,
        loss_config.label_smoothing,
    )

    # 3.3 Prepare trainer
    trainer = MLMLossTrainer(
        model=model,
        args=training_args,
        loss_config=loss_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=functools.partial(mlm_collate_fn, pad_token_id=tokenizer.pad_id),
        compute_metrics=functools.partial(compute_metrics, trim=loss_config.anchor_trim),
        callbacks=build_callbacks(model_args, training_args, logger),
    )

    # 4. Run training/evaluation pipeline
    run_training_pipeline(trainer, training_args, tokenizer=tokenizer)

    # 4A. MLM extension: fit the length predictor
    length_predictor = None
    if training_args.do_train and data_args.train_length_predictor:
        from cdrseqgen.methods.mlm.length import fit_length_predictor
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

    # 5. Run generative evaluation
    if gen_args.run_generative_eval:
        post_training_generation(
            trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
            anchor_trim=loss_config.anchor_trim,
            length_predictor=length_predictor,
        )

    return trainer


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
    # --- Load MLM tokenizer ------
    tokenizer = backbone.build_tokenizer(model_args.model_type, checkpoint_path=checkpoint_path)
    masking_policy = getattr(data_args, "masking_policy", "full_span")
    mixed_full_span_probability = getattr(data_args, "mixed_full_span_probability", 0.5)

    # --- Load MLM dataset ------
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

    # --- Select train/evaluation indices ------
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


def post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                             *, anchor_trim, length_predictor=None):
    """Pass MLM decoding and length options into shared post-training evaluation.

    common.evaluation handles judge setup, context regimes, output paths and
    exception logging. The fitted length predictor is supplied when available.
    """
    from cdrseqgen.methods.common.evaluation import post_training_generation as evaluate
    return evaluate('mlm', trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                    anchor_trim=anchor_trim, method_options={
                        'decoding': gen_args.gen_decoding,
                        'length_mode': gen_args.gen_length_mode,
                        'length_predictor': length_predictor,
                        'min_length': gen_args.gen_min_length, 'max_length': gen_args.gen_max_length,
                        'length_temperature': gen_args.gen_length_temperature})
