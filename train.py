import os
import logging
import sys
import functools
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Tuple

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    set_seed
)

from dataset import AntibodyInfillingDataset, collate_fn
from backbones import resolve
from linegraph_plotter import plot_linegraph

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Dataset registry: maps a substring found in data_args.data_path to the short tag used in
# run names. Every training data source gets an explicit tag so the dataset is never silently
# implied by its absence (an unlabelled "cdr3_scratch" is exactly what makes runs ambiguous).
# To add a new source, add one (substring, tag) pair here — matching is first-hit, so order
# more-specific substrings before more-general ones.
DATASET_TAGS = [
    ("sabdab2", "SD2"),  # SAbDab2 antibody-antigen structural data
    ("paired",  "OAS"),  # OAS paired heavy/light repertoire data
]
UNKNOWN_DATASET_TAG = "DS"  # fallback so an unrecognised source is still visibly tagged


def dataset_tag(data_path: str) -> str:
    """Resolve the short dataset tag for a data path via DATASET_TAGS (first match wins)."""
    lowered = data_path.lower()
    for substring, tag in DATASET_TAGS:
        if substring in lowered:
            return tag
    logger.warning(
        f"⚠️ Data path '{data_path}' matched no entry in DATASET_TAGS; "
        f"tagging run as '{UNKNOWN_DATASET_TAG}'. Add it to DATASET_TAGS for a clearer name."
    )
    return UNKNOWN_DATASET_TAG


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    model_type: str = field(
        default="scratch",
        metadata={"help": "Path to pretrained model or 'scratch' for our custom transformer."}
    )
    hidden_size: int = field(
        default=128,
        metadata={"help": "Hidden size for the scratch model."}
    )
    num_layers: int = field(
        default=4,
        metadata={"help": "Number of transformer layers for the scratch model."}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input."""
    data_path: str = field(
        default="dataset/data/processed/processed_paired_data.json",
        metadata={"help": "The path to the processed JSON data file."}
    )
    max_len: int = field(
        default=256, # Increased from 128 to fit both chains!
        metadata={"help": "Maximum sequence length for tokenization."}
    )
    test_size: float = field(
        default=0.1,
        metadata={"help": "Fraction of data to use for validation."}
    )
    target_region: str = field(
        default="h3",
        metadata={"help": "Which loop to mask: heavy h1/h2/h3, or light l1/l2/l3"}
    )
    include_partner_chain: bool = field(
        default=False,
        metadata={"help": "Whether to prepend the paired partner chain as context — the light chain "
                          "for a heavy (h*) target, the heavy chain for a light (l*) target."}
    )
    include_antigen: bool = field(
        default=False,
        metadata={"help": "Whether to prepend the bound antigen sequence as context (SAbDab2-derived data only)."}
    )


def setup_args() -> Tuple[ModelArguments, DataArguments, TrainingArguments]:
    """Parses CLI arguments and sets up the timestamped output directory."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        if len(sys.argv) > 2:
            remaining_args = sys.argv[2:]
            model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=remaining_args)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    target_loop = data_args.target_region
    # Dataset tag so otherwise-identical runs on different data don't collide in the run name,
    # and so every run name states its source explicitly (see DATASET_TAGS above).
    ds_tag = f"_{dataset_tag(data_args.data_path)}"
    # The partner chain differs by target, so the tag names the chain actually added: light chain
    # (_LC) for a heavy (h*) target, heavy chain (_HC) for a light (l*) target.
    if getattr(data_args, 'include_partner_chain', False):
        partner_tag = "_HC" if target_loop.lower().startswith("l") else "_LC"
    else:
        partner_tag = ""
    ag_tag = "_AG" if getattr(data_args, 'include_antigen', False) else ""
    model_short = model_args.model_type.split('/')[-1]

    # Combine: [DATETIME]_[LOOP][DS_TAG][PARTNER_TAG][AG_TAG]_[MODEL]
    # Example: 20260605_175022_h3_SD2_LC_AG_esm2_t12_35M_UR50D  (heavy target, light-chain context)
    #          20260605_175022_l3_OAS_HC_scratch                (light target, heavy-chain context)
    run_name = f"{timestamp}_{target_loop}{ds_tag}{partner_tag}{ag_tag}_{model_short}"
    
    # Update paths
    training_args.output_dir = os.path.join("./results", run_name)
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    training_args.run_name = run_name

    os.makedirs(training_args.output_dir, exist_ok=True)
    logger.info(f"📁 Workspace initialized at: {training_args.output_dir}")
    
    return model_args, data_args, training_args


def get_datasets(model_args, data_args, training_args, backbone) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, any]:
    """Handles tokenizer selection, dataset loading, and splitting."""
    # Tokenizer comes from the resolved backbone as a uniform TokenizerAdapter.
    tokenizer = backbone.build_tokenizer(model_args.model_type)

    # Dataset Loading
    full_dataset = AntibodyInfillingDataset(
        data_args.data_path,
        tokenizer,
        max_len=data_args.max_len,
        target_region=data_args.target_region,
        include_partner_chain=data_args.include_partner_chain,
        include_antigen=data_args.include_antigen
    )

    # Split
    # Honor a per-entry "split" field when every entry has one: SAbDab2 carries its own
    # leakage-safe split (from abag_split.csv), and OAS now can too via preprocess_paired.py
    # --split cluster (CDR3-identity clustered holdout). Only data preprocessed WITHOUT a split
    # field falls back to the fresh random split below, which leaks clonal near-duplicates.
    if len(full_dataset) > 0 and all('split' in entry for entry in full_dataset.data):
        train_idx = [i for i, entry in enumerate(full_dataset.data) if entry['split'] == 'train']
        val_idx = [i for i, entry in enumerate(full_dataset.data) if entry['split'] != 'train']
    else:
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)),
            test_size=data_args.test_size,
            random_state=training_args.seed
        )

    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    eval_ds = torch.utils.data.Subset(full_dataset, val_idx)
    
    logger.info(f"📊 Data loaded: {len(train_ds)} train, {len(eval_ds)} eval.")
    return train_ds, eval_ds, tokenizer


def save_training_csv(trainer, output_dir):
    """Extracts eval metrics, and saves them to a CSV."""
    history = trainer.state.log_history
    
    # Filter only the logs that contain evaluation metrics
    eval_logs = [log for log in history if 'eval_loss' in log]
    
    if not eval_logs:
        logger.warning("No evaluation logs found to plot.")
        return
    
    # Deduplicate Epochs to prevent post-training evaluation artifacts 
    seen_epochs = set()
    unique_eval_logs = []
    for log in eval_logs:
        rounded_epoch = round(log['epoch'], 4)
        if rounded_epoch not in seen_epochs:
            seen_epochs.add(rounded_epoch)
            unique_eval_logs.append(log)

    # ---> FIXED: Now extracting from unique_eval_logs! <---
    epochs = [log['epoch'] for log in unique_eval_logs]
    eval_loss = [log['eval_loss'] for log in unique_eval_logs]
    eval_acc = [log.get('eval_accuracy', 0.0) for log in unique_eval_logs]
    eval_core_acc = [log.get('eval_core_accuracy', 0.0) for log in unique_eval_logs]
    eval_anchor_acc = [log.get('eval_anchor_accuracy', 0.0) for log in unique_eval_logs]

    # Save data to a CSV for easy multi-line plotting later
    df = pd.DataFrame({
        "epoch": epochs,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "eval_core_accuracy": eval_core_acc,
        "eval_anchor_accuracy": eval_anchor_acc,
    })
    csv_path = os.path.join(output_dir, "epoch_metrics.csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"📈 Evaluation metrics saved to CSV at: {csv_path}")


def plot_training_curves(output_dir):
    """Uses the generic linegraph tool to plot the current run's CSV."""
    # We wrap the current run directory in a dictionary so plot_linegraph can read it
    current_run_dict = {"Current Run": output_dir}
    
    # Plot Loss (Saves to output_dir/eval_loss_curve.png)
    plot_linegraph(
        runs_dict=current_run_dict, 
        metric="eval_loss", 
        title="Evaluation Loss",
        save_dir=output_dir
    )
    
    # Plot Accuracy (Saves to output_dir/eval_accuracy_curve.png)
    plot_linegraph(
        runs_dict=current_run_dict, 
        metric="eval_accuracy", 
        title="Sequence Recovery (Accuracy)",
        save_dir=output_dir
    )
    
    logger.info(f"📊 Training curves successfully saved to {output_dir}")


def run_training_pipeline(trainer: Trainer, training_args: TrainingArguments):
    """Executes the train and eval loops based on the provided flags."""
    if training_args.do_train:
        logger.info("🚀 Starting Training...")
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    if training_args.do_eval:
        logger.info("🔍 Starting Evaluation...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
    save_training_csv(trainer, training_args.output_dir)
    plot_training_curves(training_args.output_dir)

# Residues trimmed from EACH end of the target loop to separate the V/J-gene-determined "anchor"
# residues (predicted ~80-100% regardless of model) from the hypervariable "core" that is actually
# being designed. The raw whole-span `accuracy` is dominated by the anchors and can mask a real
# effect on the core (see README Known Issues). Override with env ANCHOR_TRIM if needed.
ANCHOR_TRIM = int(os.environ.get("ANCHOR_TRIM", "2"))


def _core_anchor_masks(masked_positions: np.ndarray, trim: int = ANCHOR_TRIM):
    """Split each row's contiguous target span into core (middle) and anchor (trimmed ends) masks.

    The target loop is a single contiguous block of non-(-100) positions per row (see
    AntibodyInfillingDataset: [cls]+pt1+mask+pt2+[end]). Rows whose loop is too short to have a
    core after trimming `trim` from each end contribute only anchor tokens.
    """
    core = np.zeros_like(masked_positions)
    for r in range(masked_positions.shape[0]):
        cols = np.nonzero(masked_positions[r])[0]
        if cols.size > 2 * trim:
            core[r, cols[trim:cols.size - trim]] = True
    anchor = masked_positions & ~core
    return core, anchor


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Sequence recovery on the masked loop: whole-span, plus anchor-excluded core and anchor-only."""
    predictions, labels = eval_pred

    # If the model outputs a tuple (logits, hidden_states), extract the logits
    if isinstance(predictions, tuple):
        logits = predictions[0]
    else:
        logits = predictions

    # Take argmax over the vocab dimension (axis=-1) to get predicted token IDs
    preds = np.argmax(logits, axis=-1)

    # Create a boolean mask where labels are NOT -100 (i.e., the target loop only)
    masked_positions = (labels != -100)

    # Check if there are any masked tokens to evaluate (to avoid division by zero)
    total_masked = masked_positions.sum()
    if total_masked == 0:
        return {"accuracy": 0.0, "core_accuracy": 0.0, "anchor_accuracy": 0.0}

    correct = (preds == labels) & masked_positions

    # Split the span into hypervariable core vs. V/J-determined anchor ends.
    core_mask, anchor_mask = _core_anchor_masks(masked_positions)
    core_total = core_mask.sum()
    anchor_total = anchor_mask.sum()

    return {
        "accuracy": float(correct.sum() / total_masked),
        "core_accuracy": float((correct & core_mask).sum() / core_total) if core_total else 0.0,
        "anchor_accuracy": float((correct & anchor_mask).sum() / anchor_total) if anchor_total else 0.0,
    }


def main():
    model_args, data_args, training_args = setup_args()
    set_seed(training_args.seed)

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(model_args, data_args, training_args, backbone)

    # load_training_model ignores the scratch-only kwargs for pretrained backbones (**kw).
    model = backbone.load_training_model(
        model_args.model_type,
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_args.hidden_size,
        num_layers=model_args.num_layers,
        max_len=data_args.max_len,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=functools.partial(collate_fn, pad_token_id=tokenizer.pad_id),
        compute_metrics=compute_metrics,
    )

    run_training_pipeline(trainer, training_args)

if __name__ == "__main__":
    main()