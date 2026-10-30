"""Define shared arguments, parse configuration and initialize training runs.

Arguments:
  * Argument groups:
    * model_args    -- backbone, scratch architecture, anchors, early stopping
    * data_args     -- corpus, target loop, split, context, input length cap
    * gen_args      -- post-training sampling and the independent judge
    * training_args -- Hugging Face TrainingArguments

  * A method extends the three Base*Arguments with its own fields and builds its own parser.
  * Every setup_args() follows this sequence; argument checks precede run-directory creation:
      parse_config_and_cli -> shared checks, then the method's own -> initialize_run -> the four groups
  * A more detailed list of arguments can be found in docs/ARGUMENTS.md

Functions:
  This file:
  * parse_config_dict    -- construct argument groups from a config dictionary
  * parse_config_and_cli -- read JSON + optional overrides, or bare CLI arguments
  * validate_common_args, validate_anchor_trim, validate_early_stopping, validate_attention_heads,
    validate_no_label_smoothing      -- reject a bad config before initialize_run
  * initialize_run                  -- name the run from RESOLVED arguments, create it, start its log
  * add_sampling_arguments          -- flags shared by evaluate.py and generate.py (gen_* twins)
  In each method:
  * setup_args()                    -- called by its train_cli
  * validate_inference_args(args)   -- check method-specific inference options;
                                      AR/S2S also resolve args.decoding
  * validate_args(model, data, gen) -- AR and S2S only; MLM checks inline in setup_args()

Notes:
  * Explicit CLI flags override JSON values. Omitted fields use dataclass defaults;
    JSON nulls, lists and dictionaries retain their values. metadata["help"] supplies CLI help.
  * Comparing judge scores requires the same judge and scoring context. Generator
    self-perplexities are not comparable across models; sequence metrics need no judge.
  * ``method`` lives on the base because train.py reads it from the config to choose the package,
    and HfArgumentParser rejects a config key that is not a declared field.
"""
from argparse import SUPPRESS
from copy import deepcopy
import json
import logging
import os
import sys
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Optional

from cdrseqgen.datasets.dataset_common import AntibodyFrameworkDataset
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.methods.common.logging import attach_file_logging
from cdrseqgen.methods.common.run_setup import (
    RUN_SUFFIX_ENV, dataset_tag, results_root, unique_run_dir, write_run_config,
)
from cdrseqgen.methods.registry import decoding_choices

logger = logging.getLogger(__name__)

# Context settings for the judge (and for standalone evaluate.py's generator): follow training, or force.
PARTNER_CONTEXTS = ("trained", "with", "without")


@dataclass
class BaseModelArguments:
    """Backbone choice, from-scratch architecture knobs, early stopping."""

    method: str = field(
        default="mlm",
        metadata={"help": "Which method package trains this config: 'mlm', 'ar' or 's2s'. Read by "
                          "the train.py dispatcher; a method's own default is authoritative."})
    model_type: str = field(default="scratch", metadata={"help": "Backbone id, hub id, or path."})
    hidden_size: int = field(default=128, metadata={"help": "Scratch model width."})
    num_layers: int = field(default=4, metadata={"help": "Scratch model depth."})
    num_attention_heads: int = field(
        default=8,
        metadata={"help": "Must divide hidden_size; AR and S2S (rotary positions) also need an even "
                          "head width."})
    intermediate_size: Optional[int] = field(
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

    data_path: str = field(default="data/processed/processed_oas_multi.json",
                           metadata={"help": "Processed corpus JSON to train and evaluate on."})
    target_region: str = field(default="h3", metadata={"help": "h1/h2/h3 or l1/l2/l3."})
    eval_split: str = field(
        default="",
        metadata={"help": "Named split driving early stopping and best-checkpoint selection. "
                          "REQUIRED when the dataset carries a 'test' partition."})
    test_size: float = field(default=0.1, metadata={"help": "Fallback split for unlabelled data."})
    max_len: int = field(default=256, metadata={"help": "Cap on the assembled sequence."})
    include_partner_chain: bool = field(
        default=False,
        metadata={"help": "Add the paired chain (light for an h* target, heavy for l*) to the context."})
    include_antigen: bool = field(
        default=False, metadata={"help": "Add the antigen sequence to the context."})
    context_dropout: float = field(
        default=0.0,
        metadata={"help": "Fraction of examples dropping each context kind, drawn independently."})


@dataclass
class BaseGenerationArguments:
    """Post-training generative evaluation. ON by default."""

    run_generative_eval: bool = field(
        default=True,
        metadata={"help": "After training, sample candidates for held-out frameworks and score them "
                          "(settings: the gen_* fields)."})
    gen_eval_context_ablation: bool = field(default=False, metadata={
        "help": "Also evaluate without partner context, keeping judge context fixed; requires partner training."})
    gen_num_examples: int = field(default=150, metadata={"help": "Eval frameworks (-1 = all)."})
    gen_num_samples: int = field(default=20, metadata={"help": "Candidates per framework."})
    gen_temperature: float = field(
        default=1.0, metadata={"help": "Softmax temperature (>1 flatter, <1 sharper); 0 is greedy."})
    gen_top_k: int = field(default=0, metadata={"help": "Keep only the top-k logits (0 = disabled)."})
    gen_top_p: float = field(
        default=1.0, metadata={"help": "Nucleus sampling cumulative-prob cutoff (1.0 = disabled)."})
    gen_seed: int = field(default=42, metadata={"help": "Seed for reproducible sampling."})
    gen_stratify: bool = field(default=False, metadata={"help": "Balance across length buckets."})
    gen_skip_perplexity: bool = field(
        default=False,
        metadata={"help": "Skip self-scoring (pseudo-perplexity for MLM, perplexity for AR/S2S)."})
    gen_judge: Optional[str] = field(
        default="Exscientia/IgBert",
        metadata={"help": "Independent judge; '' or JSON null disables. Compare judge scores using the same "
                          "judge and context; generator self-perplexities are not comparable "
                          "across models. Sequence metrics do not require a judge."})
    gen_judge_partner_context: str = field(
        default="trained",
        metadata={"help": "Judge partner context: trained, with, or without; independent of generator context.",
                  "choices": PARTNER_CONTEXTS})
    gen_judge_path: Optional[str] = field(
        default=None, metadata={"help": "Local weights for the judge, if not the hub id."})


def validate_anchor_trim(value):
    """Reject anything but a non-negative integer; a bool is not a count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("anchor_trim must be a non-negative integer")


def validate_common_args(model_args, data_args, gen_args, *, expected_method):
    """Validate the method, target loop, context and sampling settings every method shares.

    Run it before initialize_run() so a bad config leaves no run directory behind.
    """
    method = str(model_args.method).strip().lower()
    if method != expected_method:
        raise ValueError(
            f"{expected_method!r} entrypoint received method={model_args.method!r}; route through "
            "train.py or pass a matching --method override."
        )
    model_args.method = method

    # The dataset's own check (same table, same message), made before the run directory exists.
    AntibodyFrameworkDataset._resolve_target(data_args.target_region)
    if not 0.0 <= data_args.context_dropout <= 1.0:
        raise ValueError("context_dropout must be between 0 and 1")
    if gen_args.gen_judge_partner_context not in PARTNER_CONTEXTS:
        raise ValueError("gen_judge_partner_context must be trained, with, or without")
    if getattr(gen_args, "gen_eval_context_ablation", False):
        if not data_args.include_partner_chain or not gen_args.run_generative_eval or gen_args.gen_num_examples == 0:
            raise ValueError("gen_eval_context_ablation requires partner context and generative evaluation")
    validate_anchor_trim(model_args.anchor_trim)
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


def validate_early_stopping(model_args, training_args):
    """Reject a patience that build_callbacks() would silently ignore.

    Early stopping is only attached when load_best_model_at_end is on, so a config asking for
    patience without it would train every epoch and say nothing about it.
    """
    if model_args.early_stopping_patience and not training_args.load_best_model_at_end:
        raise ValueError("early_stopping_patience > 0 requires load_best_model_at_end=true; without "
                         "it early stopping is silently not enabled")


def validate_no_label_smoothing(training_args):
    """Reject label smoothing for a method whose model computes its own loss (AR, S2S).

    With label_smoothing_factor on, Hugging Face's Trainer replaces the model's loss with its own
    smoother and shifts the labels only for the models on its causal-LM list, which ours are not:
    on AR it would train against labels one position off (measured: the smoothed loss equals the
    UNshifted one). The masked method applies smoothing itself, in LossConfig. If AR/S2S gain a
    selectable loss (``build_trainer(compute_loss_func=...)``), smoothing belongs in that function
    and this is the check to relax.
    """
    if training_args.label_smoothing_factor:
        raise ValueError("label_smoothing_factor is only supported by the masked method; this "
                         "method's model computes its own loss, which the Trainer's smoother "
                         "would replace")


def validate_attention_heads(model_args, *, even_head_width=False):
    """Check that the heads can split the model width, before any dataset or model is built.

    Only a model built from these settings reads them (a scratch model; S2S also builds its own
    decoder around a pretrained encoder), so each method decides when to call this. Without it a
    bad pair fails after the run directory exists: at model construction, or for the causal
    model's rotary attention at the first forward pass, as a tensor-shape error.
    even_head_width is the extra rule of the rotary-position models (AR, S2S), which rotate pairs
    of channels; MLM's nn.MultiheadAttention has no such rule.
    """
    heads, width = model_args.num_attention_heads, model_args.hidden_size
    if width < 1:
        raise ValueError(f"hidden_size must be positive, got {width}")
    if heads < 1 or width % heads:
        raise ValueError(f"hidden_size={width} must be divisible by num_attention_heads={heads}")
    if even_head_width and (width // heads) % 2:
        raise ValueError(f"hidden_size={width} / num_attention_heads={heads} gives heads of width "
                         f"{width // heads}; rotary attention needs an even head width")


def _normalize_warmup_config(parser, config):
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
    return parser.parse_dict(_normalize_warmup_config(parser, config), allow_extra_keys=allow_extra_keys)


def _parse_cli_overrides(parser, argv):
    """Return only explicitly supplied CLI fields, using the parser's types and aliases.

    Example: --seed 7 -> {"seed": 7}, with no unrelated default values.
    Required fields may come from JSON. Suppress defaults on a copy so neither
    missing CLI fields nor boolean complement defaults overwrite the config.
    """
    # HF uses dict_keys for some choices; these read-only collections cannot be deep-copied.
    choices = {id(action.choices): action.choices for action in parser._actions}
    override_parser = deepcopy(parser, choices)
    override_parser._defaults.clear()
    for action in override_parser._actions:
        action.default = SUPPRESS
        action.required = False
    namespace, remaining = override_parser.parse_known_args(argv)
    if remaining:
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {remaining}")
    return vars(namespace)


def parse_config_and_cli(parser):
    """Parse a config JSON, a config plus trailing CLI overrides, or bare CLI arguments.

    Example: JSON {"label_names": [], "gen_judge": null} plus --seed 7
      -> preserve the empty list and disabled judge; set only seed to 7.

    Omitted keys use dataclass defaults; explicit JSON values are preserved.
    With trailing flags, precedence is dataclass defaults < script.args < JSON
    < CLI. JSON-only parsing ignores script.args, as before. The supplied parser
    is unchanged; validation and dataclass construction run on the merged values.
    """
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".json"):
        return parser.parse_args_into_dataclasses()

    with open(sys.argv[1]) as handle:
        config = _normalize_warmup_config(parser, json.load(handle))
    if len(sys.argv) > 2:
        args_file = Path(sys.argv[0]).with_suffix(".args")
        if args_file.is_file():
            config = {**_parse_cli_overrides(parser, args_file.read_text().split()), **config}
        config.update(_parse_cli_overrides(parser, sys.argv[2:]))
    return parse_config_dict(parser, config)


def initialize_run(model_args, data_args, gen_args, training_args, *, marker=""):
    """Name the run from RESOLVED arguments, create its directory, attach its log, write provenance.

    Naming: ``[LOOP][_DS][_PARTNER][_AG][_MARKER]_[MODEL][_SUFFIX]`` — e.g. ``h3_OAS_LC_scratch``
    for the masked method and ``h3_OAS_ar_scratch`` for the causal one. Resolved, not the config
    filename, so ``--target_region l3`` cannot be misfiled under the config it came from.

    ``results/runs/latest`` is deliberately NOT updated here: everything above can succeed for a
    config that ``get_datasets`` then rejects, and repointing at a dead run would take latest away
    from the last usable batch. Each method links it once its datasets have loaded.
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


def add_sampling_arguments(parser):
    """Add the decoder, sampling and length flags that evaluate.py and generate.py share.

    Every method accepts the same standalone flags, so they are defined once here instead of in each
    command; the method's prepare_evaluation / prepare_generation rejects what it cannot honour
    (MLM rejects --max_new_tokens, AR/S2S reject length bounds). Each flag mirrors a ``gen_*`` field
    of the training-time Generation arguments with the same default wherever the field has one
    (test_shared_contracts pins this). Where the field has a trained default but the flag is None
    (--decoding, --max_new_tokens), the command takes the value from the checkpoint's
    run_config.json instead.
    """
    parser.add_argument("--decoding", choices=decoding_choices(), default=None,
                        help="Decoder for the selected method. MLM: independent (one pass) or "
                             "iterative (one residue per pass); default is a compatible saved "
                             "choice, otherwise independent. AR and S2S have a single decoder.")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Candidate loops to sample per framework (collapses to 1 when "
                             "--temperature 0)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature (>1 flatter, <1 sharper)")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Keep only the top-k logits (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling cumulative-prob cutoff (1.0 = disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling")
    parser.add_argument("--skip_perplexity", action="store_true",
                        help="Skip self-scoring (pseudo-perplexity for MLM, perplexity for AR/S2S), "
                             "the compute-heavy per-candidate part")
    parser.add_argument("--min_length", type=int, default=None,
                        help="Inclusive minimum CDR length, alone or with --max_length "
                             "(MLM only; AR/S2S choose length via <eos>)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Inclusive maximum CDR length, alone or with --min_length (MLM only)")
    parser.add_argument("--length_temperature", type=float, default=1.0,
                        help="Temperature for sampled length classes; 0 selects the most likely "
                             "allowed length (MLM only)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="AR/S2S decode cap, including EOS (default: the run's recorded "
                             "gen_max_new_tokens)")
