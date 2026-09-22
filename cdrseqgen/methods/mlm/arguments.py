"""Parses and processes CLI arguments. Consists of several dataclasses.
  
  * setup_args          -- parse config/CLI, name the run, create its directory, start its log

  * ModelArguments      -- backbone, scratch architecture, loss, and early-stopping settings
  * DataArguments       -- which JSON, which loop, which context, length cap
  * GenerationArguments -- the post-training design sweep (on by default)

  * TrainingArguments   -- predefined by HF, including fields (learning_rate, num_train_epochs,
                            metric_for_best_model, ...)  that can be set in a config or overridden 
                            on the CLI

Notes:
  * A config path is positional: `python train.py configs/mlm/heavy/h3_scratch.json`. Trailing flags OVERRIDE
    the JSON rather than replacing it, so one config plus a flag covers most one-off experiments.
  * The run folder is named from the RESOLVED arguments, not the config filename — `h3_scratch.json
    --target_region l3` produces an `l3_OAS_HC_scratch` folder, so an override can't be misfiled.
  * setup_args is where a run becomes a directory on disk: it names the run, creates it, attaches
    the file log, and writes run_config.json. Everything after it can assume training_args.output_dir
    exists. It does NOT update results/runs/latest — train.py does that once the datasets have loaded, so
    a config rejected by get_datasets cannot take the pointer from the last usable batch.
"""
from dataclasses import dataclass, field
from typing import Tuple

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.datasets.dataset_mlm import normalize_masking_policy
from cdrseqgen.methods.mlm.length import resolve_generation_length_policy
from cdrseqgen.methods.common.arguments import (BaseDataArguments,
                                                BaseGenerationArguments,
                                                BaseModelArguments,
                                                initialise_run, parse_config_and_cli,
                                                validate_common_args)
from cdrseqgen.methods.mlm.losses import loss_config_from_args


@dataclass
class ModelArguments(BaseModelArguments):
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    loss_type: str = field(
        default="cross_entropy",
        metadata={"help": "MLM objective: cross_entropy or weighted_cross_entropy."}
    )
    core_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Core-residue weight under weighted_cross_entropy."}
    )
    anchor_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Anchor-residue weight under weighted_cross_entropy."}
    )


@dataclass
class DataArguments(BaseDataArguments):
    """Arguments pertaining to what data we are going to input."""
    masking_policy: str = field(
        default="full_span",
        metadata={"help": "Training corruption: 'full_span' preserves the original CDR-infilling "
                          "objective; 'mixed' combines fully masked loops with partially revealed "
                          "loops for iterative-decoder conditioning."}
    )
    mixed_full_span_probability: float = field(
        default=0.5,
        metadata={"help": "Under masking_policy=mixed, probability that a training example still "
                          "uses the full-span mask. Other examples mask a uniformly sampled proper "
                          "subset. Evaluation is always full-span."}
    )
    train_length_predictor: bool = field(
        default=True,
        metadata={"help": "Fit and save the small gap-sentinel length classifier after sequence "
                          "training. Fixed-length generation does not require this artifact."}
    )
    length_predictor_epochs: int = field(
        default=20, metadata={"help": "Epochs for the frozen-backbone length classifier."}
    )
    length_predictor_learning_rate: float = field(
        default=1e-2, metadata={"help": "Learning rate for the length classifier."}
    )
    length_predictor_batch_size: int = field(
        default=16, metadata={"help": "Feature-extraction/training batch size for length prediction."}
    )


@dataclass
class GenerationArguments(BaseGenerationArguments):
    """Post-training generative evaluation (see cdrseqgen/methods/mlm/evaluation.py). ON by default.

    On gen_judge: pseudo-perplexity under a model's OWN distribution measures self-consistency, not
    quality — an overconfident model rates its own samples highly regardless. Comparing self-ppl
    across models compares two different measuring sticks (empirically it put the scratch-vs-ESM-2
    gap at 2.65x where a judge put it at 1.52x). Scoring every candidate under one INDEPENDENT
    reference LM fixes this. Comparability requires the same judge across runs — NOT the same
    process — so per-run judging is valid. IgBert is a good default: antibody-specific, and
    independent of both the scratch model and ESM-2.
    """
    gen_decoding: str = field(
        default="independent",
        metadata={"help": "independent | iterative. Independent is the safe default for full_span "
                          "training; iterative requires masking_policy=mixed. Set gen_temperature "
                          "to 0 for greedy decoding."}
    )
    gen_length_mode: str = field(
        default="fixed_native",
        metadata={"help": "Length policy for post-training eval: fixed_native, bounded, or inferred."}
    )
    gen_min_length: int = field(
        default=None, metadata={"help": "Optional lower length bound for bounded generation."}
    )
    gen_max_length: int = field(
        default=None, metadata={"help": "Optional upper length bound for bounded generation."}
    )
    gen_length_temperature: float = field(
        default=1.0, metadata={"help": "Temperature for sampling the learned length distribution."}
    )


def setup_args() -> Tuple[ModelArguments, DataArguments, GenerationArguments, TrainingArguments]:
    """Parses CLI arguments and sets up the timestamped output directory."""
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, gen_args, training_args = parse_config_and_cli(parser)

    validate_common_args(model_args, data_args, gen_args, expected_method="mlm")
    data_args.masking_policy = normalize_masking_policy(data_args.masking_policy)
    loss_config = loss_config_from_args(model_args, training_args)
    model_args.loss_type = loss_config.loss_type
    # Loss metadata is consumed by MLMLossTrainer rather than the model's forward signature. Keep
    # it available to the data collator instead of letting Trainer remove it as an unused column.
    training_args.remove_unused_columns = False
    if not 0.0 <= data_args.mixed_full_span_probability <= 1.0:
        raise ValueError("mixed_full_span_probability must be between 0 and 1")
    if (
        data_args.length_predictor_epochs <= 0
        or data_args.length_predictor_learning_rate <= 0
        or data_args.length_predictor_batch_size <= 0
    ):
        raise ValueError("length predictor epochs, learning rate, and batch size must be positive")
    gen_args.gen_length_mode = gen_args.gen_length_mode.strip().lower()
    resolve_generation_length_policy(
        gen_args.gen_length_mode,
        min_length=gen_args.gen_min_length,
        max_length=gen_args.gen_max_length,
    )
    if gen_args.gen_length_temperature < 0:
        raise ValueError("gen_length_temperature must be non-negative")


    initialise_run(model_args, data_args, gen_args, training_args)
    return model_args, data_args, gen_args, training_args
