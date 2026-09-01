"""Main training pipeline. Dataset -> training -> generative eval

  * get_datasets             -- tokenizer + AntibodyInfillingDataset + the train/eval split
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

from cdrseqgen.data.dataset_common import select_split_indices
from cdrseqgen.data.dataset_mlm import AntibodyInfillingDataset
from cdrseqgen.methods.common.training import run_training_pipeline  # re-export for old callers

logger = logging.getLogger(__name__)


def get_datasets(model_args, data_args, training_args, backbone) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, any]:
    """Handles tokenizer selection, dataset loading, and splitting."""
    # Load TokenizerAdapter from dataset.py as tokenizer.
    tokenizer = backbone.build_tokenizer(model_args.model_type)
    masking_policy = getattr(data_args, "masking_policy", "full_span")
    mixed_full_span_probability = getattr(data_args, "mixed_full_span_probability", 0.5)

    # Dataset Loading
    full_dataset = AntibodyInfillingDataset(
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
    eval_full_dataset = full_dataset.masking_view("full_span")
    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    eval_ds = torch.utils.data.Subset(eval_full_dataset, val_idx)

    masking_note = f", full-span probability={mixed_full_span_probability:.2f}" \
        if masking_policy == "mixed" else ""
    logger.info(f"📊 Data loaded: {len(train_ds)} train ({masking_policy}{masking_note}), "
                f"{len(eval_ds)} eval (full_span; split: {eval_label}).")
    return train_ds, eval_ds, tokenizer




def post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                             *, anchor_trim, length_predictor=None):
    """Sample designs from the finished model, produces generation_eval.json."""
    from cdrseqgen.methods.mlm.evaluation import build_judge, run_generative_eval

    try:
        judge_model = judge_tokenizer = judge_dataset = None
        if gen_args.gen_judge:
            judge_model, judge_tokenizer, judge_dataset = build_judge(
                gen_args.gen_judge, gen_args.gen_judge_path,
                data_path=data_args.data_path, max_len=data_args.max_len,
                target_region=data_args.target_region,
                include_partner_chain=data_args.include_partner_chain,
                include_antigen=data_args.include_antigen,
                device=training_args.device,
            )

        logger.info("🎲 Running post-training generative evaluation...")
        run_generative_eval(
            trainer.model,
            tokenizer,
            eval_dataset,
            device=training_args.device,
            decoding=gen_args.gen_decoding,
            num_samples=gen_args.gen_num_samples,
            temperature=gen_args.gen_temperature,
            length_mode=gen_args.gen_length_mode,
            length_predictor=length_predictor,
            min_length=gen_args.gen_min_length,
            max_length=gen_args.gen_max_length,
            length_temperature=gen_args.gen_length_temperature,
            top_k=gen_args.gen_top_k,
            top_p=gen_args.gen_top_p,
            num_examples=gen_args.gen_num_examples,
            stratify=gen_args.gen_stratify,
            seed=gen_args.gen_seed,
            skip_perplexity=gen_args.gen_skip_perplexity,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            judge_dataset=judge_dataset,
            judge_name=gen_args.gen_judge,
            anchor_trim=anchor_trim,
            output_path=os.path.join(training_args.output_dir, "generation_eval.json"),
            title=f"{os.path.basename(training_args.output_dir)}   target={data_args.target_region}",
            meta={
                "model_dir": training_args.output_dir,
                "target_region": data_args.target_region,
                "include_partner_chain": bool(data_args.include_partner_chain),
                "include_antigen": bool(data_args.include_antigen),
            },
        )
    except (RuntimeError, OSError, ValueError, FileNotFoundError, ImportError) as e:
        # Guarded so a generation failure cannot destroy a finished training run, but logged with a
        # traceback: a missing length head raises FileNotFoundError here, and "{e}" alone left no
        # way to tell that from an OOM when reading train.log afterwards.
        logger.exception(f"⚠️ Generative eval failed ({type(e).__name__}); the trained model is saved: {e}")
