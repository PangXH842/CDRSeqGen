"""Base run configuration: the arguments and run setup every method shares.

  * BaseModelArguments / BaseDataArguments / BaseGenerationArguments -- the fields all methods
    declare identically. A method subclasses one and adds only what is its own.
  * parse_config_and_cli  -- config JSON, or config + trailing CLI overrides, or bare CLI
  * initialise_run         -- name the run from RESOLVED arguments, create it, start its log

Splitting these out is what lets a master entrypoint dispatch: each method keeps its OWN parser, so
no parser ever accepts a flag that is meaningless for it, while the shared 28 fields cannot drift.
They already had: the same concept was called ``gen_judge_path`` on one side and
``gen_judge_weights`` on the other before this consolidation.

``method`` lives on the base because the dispatcher reads it out of the config to decide which
package to hand off to — and because a config carrying an undeclared key is rejected outright by
``HfArgumentParser``, so it has to be a real field somewhere.
"""
import json
import logging
import os
import sys
from dataclasses import dataclass, field, fields as dataclass_fields

from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.methods.common.logging import attach_file_logging
from cdrseqgen.methods.common.run_setup import (
    RUN_SUFFIX_ENV, dataset_tag, results_root, unique_run_dir, write_run_config,
)

logger = logging.getLogger(__name__)


@dataclass
class BaseModelArguments:
    """Backbone choice, from-scratch architecture knobs, early stopping."""

    method: str = field(
        default="mlm",
        metadata={"help": "Which method package trains this config: 'mlm' or 'ar'. Read by the "
                          "train.py dispatcher; a method's own default is authoritative."})
    model_type: str = field(default="scratch", metadata={"help": "Backbone id, hub id, or path."})
    hidden_size: int = field(default=128, metadata={"help": "Scratch model width."})
    num_layers: int = field(default=4, metadata={"help": "Scratch model depth."})
    num_attention_heads: int = field(default=8, metadata={"help": "Must divide hidden_size."})
    intermediate_size: int = field(
        default=None, metadata={"help": "FFN width; None = 4 x hidden_size, so the ratio tracks "
                                        "width and an architecture sweep is not confounded."})
    anchor_trim: int = field(
        default=ANCHOR_TRIM,
        metadata={"help": "Residues at each loop end counted as V/J-determined anchors."})
    early_stopping_patience: int = field(default=0, metadata={"help": "0 disables early stopping."})
    early_stopping_threshold: float = field(default=0.0, metadata={"help": "Minimum improvement."})


@dataclass
class BaseDataArguments:
    """Which JSON, which loop, which context, length cap."""

    data_path: str = field(default="data/processed/processed_oas_multi.json")
    target_region: str = field(default="h3", metadata={"help": "h1/h2/h3 or l1/l2/l3."})
    eval_split: str = field(
        default="",
        metadata={"help": "Named split driving early stopping and best-checkpoint selection. "
                          "REQUIRED when the dataset carries a 'test' partition."})
    test_size: float = field(default=0.1, metadata={"help": "Fallback split for unlabelled data."})
    max_len: int = field(default=256, metadata={"help": "Cap on the assembled sequence."})
    include_partner_chain: bool = field(default=False)
    include_antigen: bool = field(default=False)
    context_dropout: float = field(
        default=0.0,
        metadata={"help": "Fraction of examples dropping each context kind, drawn independently."})


@dataclass
class BaseGenerationArguments:
    """Post-training generative evaluation. ON by default."""

    run_generative_eval: bool = field(default=True)
    gen_eval_context_ablation: bool = field(default=False, metadata={
        "help": "Also evaluate without partner context, keeping judge context fixed; requires partner training."})
    gen_num_examples: int = field(default=150, metadata={"help": "Eval frameworks (-1 = all)."})
    gen_num_samples: int = field(default=20, metadata={"help": "Candidates per framework."})
    gen_temperature: float = field(default=1.0)
    gen_top_k: int = field(default=0)
    gen_top_p: float = field(default=1.0)
    gen_seed: int = field(default=42, metadata={"help": "Seed for reproducible sampling."})
    gen_stratify: bool = field(default=False, metadata={"help": "Balance across length buckets."})
    gen_skip_perplexity: bool = field(default=False)
    gen_judge: str = field(
        default="Exscientia/IgBert",
        metadata={"help": "Independent judge; '' disables. Self-perplexity is NOT comparable across "
                          "models, so any cross-run claim needs this."})
    gen_judge_partner_context: str = field(
        default="trained",
        metadata={"help": "Judge partner context: trained, with, or without; independent of generator context.",
                  "choices": ("trained", "with", "without")})
    gen_judge_path: str = field(
        default=None, metadata={"help": "Local weights for the judge, if not the hub id."})


def validate_common_args(model_args, data_args, gen_args, *, expected_method):
    """Validate the shared method contract before creating a run directory."""
    method = str(model_args.method).strip().lower()
    if method != expected_method:
        raise ValueError(
            f"{expected_method!r} entrypoint received method={model_args.method!r}; route through "
            "train.py or pass a matching --method override."
        )
    model_args.method = method

    if not 0.0 <= data_args.context_dropout <= 1.0:
        raise ValueError("context_dropout must be between 0 and 1")
    if gen_args.gen_judge_partner_context not in ("trained", "with", "without"):
        raise ValueError("gen_judge_partner_context must be trained, with, or without")
    if getattr(gen_args, "gen_eval_context_ablation", False):
        if not data_args.include_partner_chain or not gen_args.run_generative_eval or gen_args.gen_num_examples == 0:
            raise ValueError("gen_eval_context_ablation requires partner context and generative evaluation")
    if (isinstance(model_args.anchor_trim, bool)
            or not isinstance(model_args.anchor_trim, int)
            or model_args.anchor_trim < 0):
        raise ValueError("anchor_trim must be a non-negative integer")
    if gen_args.gen_num_examples < -1:
        raise ValueError("gen_num_examples must be -1 (all) or non-negative")
    if gen_args.gen_num_samples <= 0:
        raise ValueError("gen_num_samples must be positive")
    if gen_args.gen_temperature < 0:
        raise ValueError("gen_temperature must be non-negative")
    if gen_args.gen_top_k < 0:
        raise ValueError("gen_top_k must be non-negative")
    if not 0.0 < gen_args.gen_top_p <= 1.0:
        raise ValueError("gen_top_p must be in (0, 1]")


def _normalise_warmup_config(parser, config):
    """Translate legacy ratio configs only when the parser uses the new warmup_steps API."""
    config = dict(config)
    supported = {item.name for kind in parser.dataclass_types for item in dataclass_fields(kind)
                 if item.init}
    if "warmup_steps" in supported and "warmup_ratio" not in supported and "warmup_ratio" in config:
        ratio = config.pop("warmup_ratio")
        if ratio is not None:
            if isinstance(ratio, bool) or not 0 <= float(ratio) < 1:
                raise ValueError("Legacy warmup_ratio must be in [0, 1) for this Transformers version")
            if not config.get("warmup_steps"):
                config["warmup_steps"] = float(ratio)
    return config


def parse_config_dict(parser, config, *, allow_extra_keys=False):
    """Parse an execution config with the installed TrainingArguments warmup conventions."""
    return parser.parse_dict(_normalise_warmup_config(parser, config), allow_extra_keys=allow_extra_keys)


def parse_config_and_cli(parser):
    """Parse a config JSON, a config plus trailing CLI overrides, or bare CLI arguments.

    Trailing flags OVERRIDE the JSON rather than replacing it, so any field is tunable without
    writing a new config.
    """
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        config_path = os.path.abspath(sys.argv[1])
        with open(config_path) as handle:
            json_cfg = _normalise_warmup_config(parser, json.load(handle))
        if len(sys.argv) > 2:
            argv = []
            for key, value in json_cfg.items():
                # JSON null means "use the dataclass default". Serialising it as the literal
                # string "None" makes argparse reject optional numeric fields whenever a caller
                # adds an otherwise unrelated trailing override.
                if value is None:
                    continue
                argv += [f"--{key}", str(value)]
            argv += sys.argv[2:]
            return parser.parse_args_into_dataclasses(args=argv)
        return parser.parse_dict(json_cfg)
    return parser.parse_args_into_dataclasses()


def initialise_run(model_args, data_args, gen_args, training_args, *, marker=""):
    """Name the run from RESOLVED arguments, create its directory, attach its log, write provenance.

    Naming: ``[LOOP][_DS][_PARTNER][_AG][_MARKER]_[MODEL][_SUFFIX]`` — e.g. ``h3_OAS_LC_scratch``
    for the masked method and ``h3_OAS_ar_scratch`` for the causal one. Resolved, not the config
    filename, so ``--target_region l3`` cannot be misfiled under the config it came from.

    ``results/runs/latest`` is deliberately NOT updated here: everything above can succeed for a
    config that ``get_datasets`` then rejects, and repointing at a dead run would take latest away
    from the last usable batch. The entrypoint links it once the datasets have loaded.
    """
    target_loop = data_args.target_region
    ds_tag = f"_{dataset_tag(data_args.data_path)}"
    partner_tag = ("_HC" if target_loop.lower().startswith("l") else "_LC") \
        if getattr(data_args, "include_partner_chain", False) else ""
    ag_tag = "_AG" if getattr(data_args, "include_antigen", False) else ""
    marker_tag = f"_{marker}" if marker else ""
    model_name = model_args.model_type.split("/")[-1]
    suffix = os.environ.get(RUN_SUFFIX_ENV, "").strip()
    suffix = f"_{suffix}" if suffix else ""
    run_name = f"{target_loop}{ds_tag}{partner_tag}{ag_tag}{marker_tag}_{model_name}{suffix}"

    training_args.output_dir = unique_run_dir(results_root(), run_name)
    training_args.run_name = os.path.basename(training_args.output_dir)
    os.makedirs(training_args.output_dir, exist_ok=True)
    log_path = attach_file_logging(training_args.output_dir)
    write_run_config(training_args.output_dir, model_args, data_args, gen_args, training_args)
    logger.info(f"📁 Workspace initialized at: {training_args.output_dir}")
    logger.info(f"📝 Logging this run to: {log_path}")
    return training_args
