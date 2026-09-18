"""Main training pipeline. Dataset -> training -> generative eval

  * get_datasets             -- tokenizer + MaskedInfillingDataset + the train/eval split
  * post_training_generation -- sample from the finished model into generation_eval.json

Notes:
  * get_datasets honours a 'split' field when the processed JSON carries one (OAS writes a
    leakage-safe clustered split; SAbDab2 ships its own), and only falls back to a random split when
    it does not. That fallback is NOT leakage-safe — see the preprocessing scripts.
  * post_training_generation exists because training optimises recovery, which rewards copying the
    native loop — the opposite of design. It measures the design side of the same checkpoint.
  * It is exception-guarded on purpose: a generation failure (an OOM loading the judge, say) must
    never destroy a training run that already finished and saved.
"""
import os
import logging
from typing import Tuple

import torch

from cdrseqgen.datasets.dataset_common import select_split_indices, evaluation_context_view
from cdrseqgen.datasets.dataset_mlm import MaskedInfillingDataset
from cdrseqgen.methods.common.training import run_training_pipeline  # re-export for old callers
from cdrseqgen import artifacts

logger = logging.getLogger(__name__)


def get_datasets(model_args, data_args, training_args, backbone, *, checkpoint_path=None) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, any]:
    """Handles tokenizer selection, dataset loading, and splitting."""
    # Load TokenizerAdapter from dataset_common.py as tokenizer.
    tokenizer = backbone.build_tokenizer(model_args.model_type, checkpoint_path=checkpoint_path)
    masking_policy = getattr(data_args, "masking_policy", "full_span")
    mixed_full_span_probability = getattr(data_args, "mixed_full_span_probability", 0.5)

    # Dataset Loading
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

    # Data split — shared guard; see dataset_common.select_split_indices for why it is not
    # reimplemented here.
    train_idx, val_idx, eval_label = select_split_indices(
        full_dataset.data, data_args.eval_split, data_path=data_args.data_path,
        seed=training_args.seed, test_size=data_args.test_size)

    # These lightweight views share parsed records/tokenizer but not masking behavior. Training may
    # reveal native residues; validation must always begin with the entire loop masked or recovery
    # would leak the answer.
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
    from cdrseqgen.methods.common.evaluation import post_training_generation as evaluate
    return evaluate('mlm', trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                    anchor_trim=anchor_trim, method_options={
                        'decoding': gen_args.gen_decoding,
                        'length_mode': gen_args.gen_length_mode,
                        'length_predictor': length_predictor,
                        'min_length': gen_args.gen_min_length, 'max_length': gen_args.gen_max_length,
                        'length_temperature': gen_args.gen_length_temperature})
