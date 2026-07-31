"""Parses and processes CLI arguments. Consists of several dataclasses.
  
  * setup_args          -- parse config/CLI, name the run, create its directory, start its log

  * ModelArguments      -- backbone choice + scratch architecture knobs + early stopping
  * DataArguments       -- which JSON, which loop, which context, length cap
  * GenerationArguments -- the post-training design sweep (on by default)

  * TrainingArguments   -- predefined by HF, including fields (learning_rate, num_train_epochs,
                            metric_for_best_model, ...)  that can be set in a config or overridden 
                            on the CLI

Notes:
  * A config path is positional: `python train.py configs/h3_scratch.json`. Trailing flags OVERRIDE
    the JSON rather than replacing it, so one config plus a flag covers most one-off experiments.
  * The run folder is named from the RESOLVED arguments, not the config filename — `h3_scratch.json
    --target_region l3` produces an `l3_OAS_HC_scratch` folder, so an override can't be misfiled.
  * setup_args is where a run becomes a directory on disk: it names the run, creates it, attaches
    the file log, updates results/latest, and writes run_config.json. Everything after it can assume
    training_args.output_dir exists.
"""
import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Tuple

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.train.logging import (
    RUN_SUFFIX_ENV,
    attach_file_logging,
    dataset_tag,
    results_root,
    unique_run_dir,
    update_latest_link,
    write_run_config,
)

logger = logging.getLogger(__name__)


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
    num_attention_heads: int = field(
        default=8,
        metadata={"help": "Attention heads for the scratch model. Must divide hidden_size."}
    )
    intermediate_size: int = field(
        default=None,
        metadata={"help": "FFN width for the scratch model. Default None = 4 x hidden_size, so the "
                          "ratio tracks width automatically during an architecture sweep; set an "
                          "explicit value only to break that ratio deliberately."}
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "Stop after this many evals with no improvement in metric_for_best_model. "
                          "0 (default) disables early stopping and trains the full num_train_epochs."}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum change that counts as an improvement for early stopping."}
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
    context_dropout: float = field(
        default=0.0,
        metadata={"help": "Fraction of examples trained WITHOUT their optional context, so one "
                          "checkpoint survives both regimes. 0.0 (default) = current behaviour. "
                          "Without it a model is locked to its training regime in both directions "
                          "(H1/scratch: 0.984->0.261 when context is withheld, 0.997->0.159 when it "
                          "is supplied to a model trained without). Deterministic per example."}
    )


@dataclass
class GenerationArguments:
    """Post-training generative evaluation (see cdrseqgen/eval/generation.py). ON by default.

    On gen_judge: pseudo-perplexity under a model's OWN distribution measures self-consistency, not
    quality — an overconfident model rates its own samples highly regardless. Comparing self-ppl
    across models compares two different measuring sticks (empirically it put the scratch-vs-ESM-2
    gap at 2.65x where a judge put it at 1.52x). Scoring every candidate under one INDEPENDENT
    reference LM fixes this. Comparability requires the same judge across runs — NOT the same
    process — so per-run judging is valid. IgBert is a good default: antibody-specific, and
    independent of both the scratch model and ESM-2.
    """
    run_generative_eval: bool = field(
        default=True,
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
        default="Exscientia/IgBert",
        metadata={"help": "Independent LM scoring every candidate, making perplexity comparable ACROSS "
                          "runs (self-ppl is not). Empty string disables judging."}
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

    # Assemble batch dir name
    # Combine: [LOOP][DS_TAG][PARTNER_TAG][AG_TAG]_[MODEL][_SUFFIX]
    # Example: h3_SD2_LC_AG_esm2_t12_35M_UR50D  (heavy target, light-chain + antigen context)
    #          l3_OAS_HC_scratch                (light target, heavy-chain context)
    target_loop = data_args.target_region
    ds_tag = f"_{dataset_tag(data_args.data_path)}"
    if getattr(data_args, 'include_partner_chain', False):
        partner_tag = "_HC" if target_loop.lower().startswith("l") else "_LC"
    else:
        partner_tag = ""
    ag_tag = "_AG" if getattr(data_args, 'include_antigen', False) else ""
    model_name = model_args.model_type.split('/')[-1]

    # Optional tag from a batch runner to distinguish runs of the SAME config
    suffix = os.environ.get(RUN_SUFFIX_ENV, "").strip()
    suffix = f"_{suffix}" if suffix else ""

    run_name = f"{target_loop}{ds_tag}{partner_tag}{ag_tag}_{model_name}{suffix}"

    # Update paths
    training_args.output_dir = unique_run_dir(results_root(), run_name)
    training_args.run_name = os.path.basename(training_args.output_dir)

    os.makedirs(training_args.output_dir, exist_ok=True)
    log_path = attach_file_logging(training_args.output_dir)
    update_latest_link(training_args.output_dir)
    write_run_config(training_args.output_dir, model_args, data_args)
    logger.info(f"📁 Workspace initialized at: {training_args.output_dir}")
    logger.info(f"📝 Logging this run to: {log_path}")

    return model_args, data_args, gen_args, training_args
