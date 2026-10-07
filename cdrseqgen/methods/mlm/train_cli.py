"""MLM training command: root train.py calls main().

Caller:
  train.py -> methods.registry -> method.module('train_cli').main()

Steps:
  * 1.1. Set up logging                     via train.logging.configure_logging().
         (*) update latest after the datasets load successfully.
  * 1.2. Parse config/CLI                   via mlm.arguments.setup_args(); 
         and set the random seed            via transformers.set_seed().
  * 2.1. Resolve the backbone adapter       via mlm.backbones.resolve().
  * 2.2. Obtain train/eval datasets and tokenizer 
                                            from mlm.training.get_datasets().
  * 3.1. Create/load the model              via backbone.load_training_model().
  * 3.2. Configure and log the loss         via mlm.losses.loss_config_from_args().
  * 3.3. Construct mlm.losses.MLMLossTrainer 
                                            with the collator, recovery metrics and callbacks.
  * 4  . Run training and reconstruction evaluation 
                                            via common.training.run_training_pipeline(); 
         and save model/tokenizer, metrics, training CSV and plots.
  * 4A . Fit the MLM length predictor       via mlm.length_predictor.fit_length_predictor().
  * 5  . Generate and score candidates      via mlm.training.post_training_generation().

Notes:
  * configure_logging() runs at module import; main() obtains the logger.
  * Step 4 respects do_train/do_eval. Step 4A requires both do_train and
    train_length_predictor; step 5 requires run_generative_eval.
  * Step 4 measures masked-loop reconstruction. Step 5 evaluates generated
    candidate sequences under the chosen decoding and length policies.

"""
import functools

from transformers import set_seed

from cdrseqgen.datasets.dataset_mlm import mlm_collate_fn
from cdrseqgen.methods.mlm.arguments import setup_args
from cdrseqgen.methods.mlm.backbones import resolve
from cdrseqgen.methods.mlm.losses import MLMLossTrainer, loss_config_from_args
from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.methods.common.training import run_training_pipeline
from cdrseqgen.methods.mlm.recovery import compute_metrics
from cdrseqgen.methods.mlm.training import (
    get_datasets,
    post_training_generation,
)
from cdrseqgen.train.logging import configure_logging, update_latest_link

configure_logging()


def main():
    # 1.1 Set up logging
    import logging
    logger = logging.getLogger(__name__)

    # 1.2 Set up arguments
    model_args, data_args, gen_args, training_args = setup_args()
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

    # 5. Run generative evaluation
    if gen_args.run_generative_eval:
        post_training_generation(
            trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
            anchor_trim=loss_config.anchor_trim,
            length_predictor=length_predictor,
        )
