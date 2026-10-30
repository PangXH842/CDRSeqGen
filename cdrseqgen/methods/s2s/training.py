"""Encoder-decoder training pipeline: dataset -> Trainer -> metrics, and the hooks of the shared workflow.

Entry path:
  train_cli.main -> common.training.run_training_pipeline
                 -> prepare_training -> common.training.train_and_save
                 -> finish_training

  * prepare_training         -- seed, load data/model, build the Trainer, attach diagnostics
  * finish_training          -- evaluate generated loops after training
  * get_datasets             -- tokenizer + Seq2SeqInfillingDataset + the train/eval split
  * compute_metrics          -- teacher-forced recovery over the loop, split core/anchor
  * build_trainer            -- a plain HF Trainer; the model returns its own loss
  * post_training_generation -- pass the decode cap into common.evaluation

Mirrors ``methods/ar/training.py`` so the two are comparable. Two differences, both forced:

  * **No shift in the metrics.** The causal side receives predictions and labels aligned to the same
    sequence and must redo the shift itself. Here ``dataset_s2s`` already offset the decoder pair,
    so the prediction at t is scored against the label at t directly. Copying the causal version
    verbatim would offset every position by one and quietly report a different quantity — which is
    the whole reason this docstring exists rather than a "same as AR" comment.
  * **The loop is the entire supervised stream.** There is no prompt to mask out with -100, so
    ``core_coverage`` is bounded only by loop length, not by whether a prefix was labelled.

The dataset split and the recovery arithmetic are shared with the causal method in
:mod:`cdrseqgen.methods.common.eos_decoding`; this module keeps the encoder-decoder dataset and
collator, and the unshifted alignment in ``compute_metrics``.

Metrics are teacher-forced, like both siblings' ``compute_metrics``: argmax over the true prefix,
not free-running generation. Findings' "Teacher-forced accuracy is not generation accuracy" applies
here too, and the gap will be at least as large as the causal one, so these numbers select
checkpoints — they do not report design quality.
"""
import functools
import logging

from transformers import Trainer, set_seed

from cdrseqgen.datasets.dataset_s2s import Seq2SeqInfillingDataset, s2s_collate_fn
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.methods.common.eos_decoding import eos_recovery_metrics, get_infilling_datasets
from cdrseqgen.methods.common.run_setup import update_latest_link
from cdrseqgen.methods.s2s.backbones import resolve

logger = logging.getLogger(__name__)


def prepare_training(model_args, data_args, training_args):
    """Return (Trainer, tokenizer) ready for the shared training lifecycle.

    Steps:
      seed -> backbone -> datasets/tokenizer -> update latest link -> model -> Trainer/callbacks
           -> conditioning probe, when asked for

    Dataset/split validation precedes the latest-link update. Training starts when the shared
    workflow calls train_and_save().
    """
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
    trainer = build_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,
                            callbacks=build_callbacks(model_args, training_args, logger),
                            trim=model_args.anchor_trim)

    if model_args.conditioning_probe_steps or model_args.diagnostic_stop_epoch:
        from cdrseqgen.methods.s2s.diagnostics import ConditioningProbe
        trainer.add_callback(ConditioningProbe(
            eval_dataset, trainer.data_collator, model_args.conditioning_probe_steps,
            model_args.diagnostic_stop_epoch))
    return trainer, tokenizer


def finish_training(trainer, tokenizer, data_args, gen_args, training_args, *, anchor_trim):
    """Evaluate generated loops on the trained model when run_generative_eval is on.

    anchor_trim is model_args.anchor_trim; train_cli binds it because the shared workflow does not
    pass model_args to this hook.
    """
    if gen_args.run_generative_eval:
        post_training_generation(trainer, tokenizer, trainer.eval_dataset, data_args, gen_args,
                                 training_args, anchor_trim=anchor_trim)


def post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                             *, anchor_trim):
    """Pass the decode cap into shared post-training evaluation.

    common.evaluation handles judge setup, context regimes, output paths and exception logging.
    """
    from cdrseqgen.methods.common.evaluation import post_training_generation as evaluate
    return evaluate('s2s', trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                    anchor_trim=anchor_trim,
                    method_options={'max_new_tokens': gen_args.gen_max_new_tokens})


def get_datasets(model_args, data_args, training_args, backbone, *, checkpoint_path=None):
    """Tokenizer, train/eval subsets. Same signature as the masked and causal pipelines'."""
    return get_infilling_datasets(Seq2SeqInfillingDataset, "seq2seq", model_args, data_args,
                                  training_args, backbone, checkpoint_path=checkpoint_path)


def compute_metrics(eval_pred, *, trim=ANCHOR_TRIM):
    """Teacher-forced recovery over the loop: whole-loop, plus anchor-excluded core and anchor.

    Predictions and labels arrive ALIGNED — position t of the prediction answers position t of the
    label, because the dataset built the decoder pair offset by one. Do not re-shift here.
    The arithmetic itself is :func:`eos_recovery_metrics`, which also drops each row's final <eos>.
    """
    predictions, labels = eval_pred
    if predictions.ndim == 3:
        predictions = predictions.argmax(-1)
    return eos_recovery_metrics(predictions, labels, trim=trim)


def build_trainer(model, training_args, train_ds, eval_ds, tokenizer, callbacks=None,
                  trim=ANCHOR_TRIM):
    """A plain HF Trainer: like the causal side there is no external loss to bridge."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=functools.partial(
            s2s_collate_fn, pad_token_id=tokenizer.pad_id,
            encoder_pad_token_id=getattr(tokenizer, "encoder_tokenizer", tokenizer).pad_id),
        compute_metrics=functools.partial(compute_metrics, trim=trim),
        callbacks=callbacks or [],
    )
