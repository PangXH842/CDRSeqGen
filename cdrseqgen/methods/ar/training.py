"""Prepare AR training resources and run method-specific finishing stages.

Entry path:
  train_cli.main -> common.training.run_training_pipeline
                 -> prepare_training -> common.training.train_and_save
                 -> finish_training

Functions:
  * prepare_training         -- seed, load data/model, record layout and build the Trainer
  * finish_training          -- optionally evaluate generated loops on the selected/loaded model
  * get_datasets             -- return training/evaluation subsets and their tokenizer
  * compute_metrics          -- align next-token predictions and calculate loop recovery
  * build_trainer            -- connect the model, collator, recovery metric and callbacks

  Notes:
  * train_cli owns logging and argument setup; common.training owns train/eval/save.
  * Named split selection is shared; evaluation disables training context dropout.
  * Only the answer and terminator receive labels. Recovery shifts predictions once
    and excludes the final terminator from residue scores.
  * The backbone supplies the model's loss and span layout. compute_loss_func is an
    optional Trainer loss override; generated-loop evaluation is a separate stage.
"""
import functools
import logging

from transformers import Trainer, set_seed

from cdrseqgen.datasets.dataset_ar import CausalInfillingDataset, causal_collate_fn
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.methods.ar.backbones import record_training_layout, resolve
from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.methods.common import evaluation as shared_evaluation
from cdrseqgen.methods.common.eos_decoding import eos_recovery_metrics, get_infilling_datasets
from cdrseqgen.methods.common.run_setup import update_latest_link

logger = logging.getLogger(__name__)


def prepare_training(model_args, data_args, training_args):
    """Return (Trainer, tokenizer) ready for the shared training lifecycle.

    Steps:
      seed -> backbone -> datasets/tokenizer -> update latest link
           -> model -> record span layout -> Trainer/callbacks

    Validate dataset/split selection before updating the latest link. The shared
    lifecycle starts training/evaluation after this function returns.
    """
    set_seed(training_args.seed)

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(
        model_args, data_args, training_args, backbone)
    # Keep dataset/split failures from changing the latest-run link.
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
    trainer = build_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,
                            callbacks=build_callbacks(model_args, training_args, logger),
                            trim=model_args.anchor_trim)
    return trainer, tokenizer


def finish_training(trainer, tokenizer, data_args, gen_args, training_args, *, anchor_trim):
    """Optionally evaluate generated loops on the selected or loaded model.

    Also runs for evaluation-only commands. train_cli binds anchor_trim from
    model_args; run_generative_eval controls this stage independently of do_train.
    Pass the AR decode cap directly to common.evaluation.post_training_generation().
    """
    if gen_args.run_generative_eval:
        shared_evaluation.post_training_generation(
            "ar", trainer, tokenizer, trainer.eval_dataset,
            data_args, gen_args, training_args,
            anchor_trim=anchor_trim,
            method_options={"max_new_tokens": gen_args.gen_max_new_tokens},
        )


def get_datasets(model_args, data_args, training_args, backbone, *, checkpoint_path=None):
    """Return (train subset, evaluation subset, tokenizer) through the shared data helper.

    Both views share parsed records; evaluation disables context dropout.
    checkpoint_path selects a saved tokenizer for standalone checkpoint evaluation.
    """
    return get_infilling_datasets(CausalInfillingDataset, "causal", model_args, data_args,
                                  training_args, backbone, checkpoint_path=checkpoint_path)


def compute_metrics(eval_pred, *, trim=ANCHOR_TRIM):
    """Return teacher-forced whole/core/anchor recovery after the next-token shift.

    Example: prompt P P, loop A C
      labels:       -100 -100 A C EOS
      predictions:   ?     A  C EOS ?
      comparison: predictions[:-1] against labels[1:]

    Accept logits or already-decoded token IDs. eos_recovery_metrics ignores
    prompt/padding labels and drops the final supervised terminator.
    """
    predictions, labels = eval_pred
    if predictions.ndim == 3:
        predictions = predictions.argmax(-1)
    return eos_recovery_metrics(predictions[:, :-1], labels[:, 1:], trim=trim)


def build_trainer(model, training_args, train_ds, eval_ds, tokenizer, callbacks=None,
                  trim=ANCHOR_TRIM, compute_loss_func=None):
    """Return an HF Trainer with causal collation, recovery metrics and callbacks.

    The model calculates loss by default. An optional compute_loss_func receives
    unshifted labels and must align them itself. Its contract is
    f(outputs, labels, num_items_in_batch=None) -> scalar: divide summed loss by
    the supplied accumulation-window count, or return a token mean when absent.
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=functools.partial(causal_collate_fn, pad_token_id=tokenizer.pad_id),
        compute_metrics=functools.partial(compute_metrics, trim=trim),
        compute_loss_func=compute_loss_func,
        callbacks=callbacks or [],
    )
