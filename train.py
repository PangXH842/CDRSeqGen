import os
import json
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

from cdrseqgen.data.dataset import AntibodyInfillingDataset, collate_fn
from cdrseqgen.models.backbones import resolve
from cdrseqgen.eval.linegraph_plotter import plot_linegraph, plot_metric_breakdown

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILENAME = "train.log"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def attach_file_logging(output_dir: str) -> str:
    """Tee the log stream into <output_dir>/train.log, so every run keeps its own record.

    Until this existed, logger output went ONLY to stdout: a run launched by scripts/run.sh or bare
    `python train.py` left no readable log at all — the run dir held numbers (epoch_metrics.csv,
    *_results.json) and binary TensorBoard events, but nothing you could open to see what happened.

    Attached to the ROOT logger so the transformers/Trainer loggers are captured too (they propagate
    to root), and called after output_dir exists — the module-level basicConfig above runs at import,
    long before the run directory is known. Re-attaching is guarded so repeated calls in one process
    (tests, notebooks) don't duplicate every line.
    """
    log_path = os.path.join(output_dir, LOG_FILENAME)
    root = logging.getLogger()

    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == os.path.abspath(log_path):
            return log_path

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(file_handler)

    # A crash otherwise goes to stderr only — and scripts/run.sh deliberately continues past a failed
    # run ("|| true"), so the run dir would be left with no explanation of what went wrong. Route
    # uncaught exceptions through the logger first so the traceback lands in train.log too.
    def log_uncaught(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            logger.error("💥 Run failed with an uncaught exception:", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = log_uncaught
    return log_path

DATASET_TAGS = [
    ("sabdab2", "SD2"),  
    ("oas",     "OAS"),  
]
UNKNOWN_DATASET_TAG = "DS"

# Every run lives at results/<batch>/<run>/ — always two levels, so there is one shape to reason
# about and results/ never collects loose run folders again (the grouping previously done by hand:
# results/20260710/, results/20260708/, ...).
#
# Batch runners (scripts/run.sh, scripts/sweep_scratch.sh) export CDRSEQGEN_RESULTS_ROOT so every
# run of one round shares a batch folder; a standalone `python train.py ...` gets a batch folder of
# its own. An env var rather than a CLI flag, matching the ANCHOR_TRIM precedent below: it keeps the
# runners passing only the config path, so setup_args stays on the plain parse_json_file branch.
#
# The run folder itself is named from the RESOLVED arguments, not the config filename — a CLI
# override such as `--target_region l3` on h3_scratch.json produces an "..._l3_..." folder, and its
# timestamp is what keeps runs unique when one config is reused (sweep_scratch.sh reuses a single
# base config for its whole grid).
RESULTS_ROOT_ENV = "CDRSEQGEN_RESULTS_ROOT"
RUN_SUFFIX_ENV = "CDRSEQGEN_RUN_SUFFIX"
DEFAULT_RESULTS_ROOT = "./results"
LATEST_LINK = "latest"


def unique_run_dir(root: str, run_name: str) -> str:
    """<root>/<run_name>, suffixed with _2, _3, ... if that path is already taken.

    Run folders carry no timestamp (the batch folder has one), which is only safe if a name clash
    can never silently overwrite an earlier run. Two cases genuinely clash:
      * sweep_scratch.sh reuses ONE base config for its whole grid — every grid point resolves to
        the same name (mitigated further by CDRSEQGEN_RUN_SUFFIX, which names the grid point);
      * re-running a config into an existing batch folder, e.g. retrying a failed run.
    Both would otherwise land on an existing directory and overwrite real results, so the clash is
    resolved by renaming and logged loudly rather than passing silently.
    """
    candidate = os.path.join(root, run_name)
    if not os.path.exists(candidate):
        return candidate
    for n in range(2, 1000):
        alternative = os.path.join(root, f"{run_name}_{n}")
        if not os.path.exists(alternative):
            logger.warning(f"⚠️ {candidate} already exists — writing this run to {alternative} instead.")
            return alternative
    raise RuntimeError(f"Could not find a free run directory for {candidate!r} (tried _2.._999).")


def results_root() -> str:
    """Batch directory that this run's folder is created inside (see RESULTS_ROOT_ENV)."""
    from_env = os.environ.get(RESULTS_ROOT_ENV)
    if from_env:
        return from_env
    # Standalone run: give it its own batch folder so every run sits at the same depth.
    return os.path.join(DEFAULT_RESULTS_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))


def update_latest_link(run_dir: str) -> None:
    """Point results/latest at the newest run directory.

    Kept as a *run* pointer (not a batch pointer) so the documented shortcut works verbatim:
        python predict.py --model_type scratch --model_path ./results/latest ...
    The link is relative, so moving/renaming the results tree doesn't break it.

    Refuses to touch results/latest if it is a real directory rather than a symlink — that would
    mean deleting real results, which is never worth a convenience shortcut.
    """
    link_path = os.path.join(DEFAULT_RESULTS_ROOT, LATEST_LINK)
    try:
        os.makedirs(DEFAULT_RESULTS_ROOT, exist_ok=True)
        if os.path.islink(link_path):
            os.remove(link_path)
        elif os.path.exists(link_path):
            logger.warning(f"⚠️ {link_path} exists and is not a symlink — leaving it alone.")
            return
        target = os.path.relpath(os.path.abspath(run_dir), os.path.abspath(DEFAULT_RESULTS_ROOT))
        os.symlink(target, link_path)
    except OSError as e:
        logger.warning(f"⚠️ Could not update {link_path}: {e}")


def dataset_tag(data_path: str) -> str:
    """Resolve the short dataset tag for a data path via DATASET_TAGS (first match wins)."""
    lowered = data_path.lower()
    for substring, tag in DATASET_TAGS:
        if substring in lowered:
            return tag
    logger.warning(
        f"⚠️ Data path '{data_path}' matched no entry in DATASET_TAGS. "
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
        default="dataset/data/processed/processed_oas_data.json",
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


@dataclass
class GenerationArguments:
    """Optional post-training generative evaluation (see cdrseqgen/eval/generation.py).

    NOTE on gen_judge: a judge is only meaningful *across* models (the same judge applied to each),
    which a single training run cannot do by construction — and loading one alongside the trainer's
    optimizer state risks OOM. So it stays off here; use eval_generative.py for judged comparisons.
    """
    run_generative_eval: bool = field(
        default=False,
        metadata={"help": "After training, sample designs from the best model and write generation_eval.json."}
    )
    gen_num_examples: int = field(default=150, metadata={"help": "Eval frameworks to sample (-1 = all)."})
    gen_num_samples: int = field(default=20, metadata={"help": "Candidates per framework."})
    gen_decoding: str = field(default="iterative", metadata={"help": "greedy | independent | iterative."})
    gen_temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})
    gen_top_k: int = field(default=0, metadata={"help": "Top-k filter (0 = disabled)."})
    gen_top_p: float = field(default=0.9, metadata={"help": "Nucleus cutoff (1.0 = disabled)."})
    gen_stratify: bool = field(default=True, metadata={"help": "Balance the draw across H3 length buckets."})
    gen_skip_perplexity: bool = field(default=False, metadata={"help": "Skip self pseudo-perplexity."})
    gen_seed: int = field(default=42, metadata={"help": "Seed for reproducible sampling."})
    gen_judge: str = field(
        default=None,
        metadata={"help": "Independent judge LM. Normally leave unset — see class docstring; prefer eval_generative.py."}
    )
    gen_judge_path: str = field(default=None, metadata={"help": "Weights for the judge (default: the gen_judge id)."})

def setup_args() -> Tuple[ModelArguments, DataArguments, GenerationArguments, TrainingArguments]:
    """Parses CLI arguments and sets up the timestamped output directory."""
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))

    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        # Read the JSON config provided
        config_path = os.path.abspath(sys.argv[1])
        if len(sys.argv) > 2:
            # Case: extra args to override given JSON
            with open(config_path) as f:
                json_cfg = json.load(f)
            argv = []
            for key, value in json_cfg.items():
                argv += [f"--{key}", str(value)]
            argv += sys.argv[2:]
            model_args, data_args, gen_args, training_args = parser.parse_args_into_dataclasses(args=argv)
        else:
            # Case: follow JSON config 
            model_args, data_args, gen_args, training_args = parser.parse_json_file(json_file=config_path)
    else:
        # Case: parse CLI args directly
        model_args, data_args, gen_args, training_args = parser.parse_args_into_dataclasses()

    # Combine: [LOOP][DS_TAG][PARTNER_TAG][AG_TAG]_[MODEL][_SUFFIX]
    # Example: h3_SD2_LC_AG_esm2_t12_35M_UR50D  (heavy target, light-chain + antigen context)
    #          l3_OAS_HC_scratch                (light target, heavy-chain context)
    # The date/time lives on the parent batch folder, so it isn't repeated here. Uniqueness is
    # guaranteed by unique_run_dir() instead — see the note there on why that guard is required.
    target_loop = data_args.target_region
    ds_tag = f"_{dataset_tag(data_args.data_path)}"
    if getattr(data_args, 'include_partner_chain', False):
        partner_tag = "_HC" if target_loop.lower().startswith("l") else "_LC"
    else:
        partner_tag = ""
    ag_tag = "_AG" if getattr(data_args, 'include_antigen', False) else ""
    model_name = model_args.model_type.split('/')[-1]

    # Optional tag from a batch runner to distinguish runs of the SAME config — sweep_scratch.sh
    # sets e.g. "lr1e-3_ep20_wu0.06", which is far more useful on a folder than a timestamp.
    suffix = os.environ.get(RUN_SUFFIX_ENV, "").strip()
    suffix = f"_{suffix}" if suffix else ""

    run_name = f"{target_loop}{ds_tag}{partner_tag}{ag_tag}_{model_name}{suffix}"

    # Update paths.
    # NOTE: there used to be a `logging_dir = output_dir/"logs"` assignment here. It was dead —
    # the TensorBoard callback writes event files to output_dir/runs/<datetime>_<host>/ regardless,
    # and no logs/ directory was ever created — while implying a log folder that never existed.
    # The human-readable log is written by attach_file_logging() below, as output_dir/train.log.
    training_args.output_dir = unique_run_dir(results_root(), run_name)
    training_args.run_name = os.path.basename(training_args.output_dir)

    os.makedirs(training_args.output_dir, exist_ok=True)
    log_path = attach_file_logging(training_args.output_dir)
    update_latest_link(training_args.output_dir)
    logger.info(f"📁 Workspace initialized at: {training_args.output_dir}")
    logger.info(f"📝 Logging this run to: {log_path}")

    return model_args, data_args, gen_args, training_args


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

    # Data split
    if len(full_dataset) > 0 and all('split' in entry for entry in full_dataset.data):
        # If 'split' is given, split train and val accordingly
        train_idx = [i for i, entry in enumerate(full_dataset.data) if entry['split'] == 'train']
        val_idx = [i for i, entry in enumerate(full_dataset.data) if entry['split'] != 'train']
    else:
        # random train/val split, arg: data_args.test_size
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
    # Fraction of eval rows backing eval_core_accuracy. Recorded so a core score can always be
    # sanity-checked after the fact: well below 1.0 means the loop is too short for the anchor trim
    # and core is noise (see compute_metrics). Older runs lack this column.
    eval_core_cov = [log.get('eval_core_coverage', float('nan')) for log in unique_eval_logs]

    # Save data to a CSV for easy multi-line plotting later
    df = pd.DataFrame({
        "epoch": epochs,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_acc,
        "eval_core_accuracy": eval_core_acc,
        "eval_anchor_accuracy": eval_anchor_acc,
        "eval_core_coverage": eval_core_cov,
    })
    csv_path = os.path.join(output_dir, "epoch_metrics.csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"📈 Evaluation metrics saved to CSV at: {csv_path}")


def plot_training_curves(output_dir):
    """Auto-generate this run's curves from epoch_metrics.csv.

    All three recovery metrics are recorded per epoch, but only loss + whole-span were ever
    plotted — so every run's auto-plot showed the anchor-inflated metric the README says NOT to
    report, and hid the anchor-excluded CORE metric it says to use. Core is now plotted too, plus a
    combined whole-vs-core-vs-anchor chart that makes the anchor gap visible at a glance (on H3 the
    spread is ~0.46 / ~0.32 / ~0.90).
    """
    current_run_dict = {"Current Run": output_dir}

    # One PNG per metric: <metric>_curve.png
    for metric, title in [
        ("eval_loss", "Evaluation Loss"),
        ("eval_accuracy", "Sequence Recovery — whole-span (anchor-inflated)"),
        ("eval_core_accuracy", "Sequence Recovery — CORE (anchor-excluded)"),
    ]:
        plot_linegraph(runs_dict=current_run_dict, metric=metric, title=title, save_dir=output_dir)

    # Combined view: all three recovery metrics on one chart, so the anchor gap is one glance.
    plot_metric_breakdown(
        output_dir,
        metrics=[
            ("eval_accuracy", "whole-span (anchor-inflated)"),
            ("eval_core_accuracy", "core (anchor-excluded)"),
            ("eval_anchor_accuracy", "anchor only"),
        ],
        title="Recovery: whole-span vs core vs anchor",
        save_dir=output_dir,
        filename="recovery_breakdown",
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


MIN_CORE_COVERAGE = 0.5  # below this, core_accuracy is measuring too few rows to mean anything
_warned_low_core_coverage = False


def _warn_if_low_core_coverage(coverage: float, rows_with_core: int, n_rows: int) -> None:
    """Warn once per process when core_accuracy is backed by too few eval rows to be meaningful."""
    global _warned_low_core_coverage
    if coverage >= MIN_CORE_COVERAGE or _warned_low_core_coverage:
        return
    _warned_low_core_coverage = True
    logger.warning(
        f"⚠️ core_accuracy covers only {rows_with_core}/{n_rows} eval rows ({coverage:.1%}) — this "
        f"target's loops are mostly <= {2 * ANCHOR_TRIM} residues, so most have no anchor-excluded "
        f"core and the score is noise. Report WHOLE-SPAN accuracy for this loop (see README, "
        f"'How to read the numbers'). Known case: light-chain CDR2 (L2)."
    )


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

    # Fraction of eval rows that contribute ANY core token. A loop of <= 2*ANCHOR_TRIM residues has
    # no core at all, so core_accuracy can be computed over a tiny unrepresentative slice while
    # looking like an ordinary score — light-chain CDR2 is 3aa in 98.8% of OAS, leaving ~30 of 2111
    # rows, which is why L2 core reads 1.0000 for scratch and 0.5957 for scratch+HC (both noise;
    # whole-span says 0.9921 vs 0.9590). And when NO row has a core, the ratio below falls back to
    # 0.0 — indistinguishable from "got everything wrong". Recording coverage makes both visible.
    rows_with_core = int((core_mask.sum(axis=-1) > 0).sum())
    n_rows = int(masked_positions.shape[0])
    core_coverage = (rows_with_core / n_rows) if n_rows else 0.0
    _warn_if_low_core_coverage(core_coverage, rows_with_core, n_rows)

    return {
        "accuracy": float(correct.sum() / total_masked),
        "core_accuracy": float((correct & core_mask).sum() / core_total) if core_total else 0.0,
        "anchor_accuracy": float((correct & anchor_mask).sum() / anchor_total) if anchor_total else 0.0,
        "core_coverage": float(core_coverage),
    }


def post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args):
    """Sample designs from the finished model and write generation_eval.json (opt-in).

    Uses trainer.model directly: with "load_best_model_at_end" that IS the peak checkpoint and it is
    already on the GPU, so nothing is reloaded from disk. The whole thing is guarded — a generation
    failure (e.g. an OOM from a judge) must never destroy a training run that already succeeded and
    saved, so it degrades to a warning.
    """
    from cdrseqgen.eval.generation import run_generative_eval, build_judge

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
            output_path=os.path.join(training_args.output_dir, "generation_eval.json"),
            title=f"{os.path.basename(training_args.output_dir)}   target={data_args.target_region}",
            meta={
                "model_dir": training_args.output_dir,
                "target_region": data_args.target_region,
                "include_partner_chain": bool(data_args.include_partner_chain),
                "include_antigen": bool(data_args.include_antigen),
            },
        )
    except Exception as e:
        logger.warning(f"⚠️ Generative eval failed: {e}")


def main():
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

    if gen_args.run_generative_eval:
        post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args)


if __name__ == "__main__":
    main()