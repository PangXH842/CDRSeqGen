"""Arguments configuration and setup -- MLM method.

Caller:
  mlm.train_cli.main() -> setup_args()

Argument groups:
  model_args    -- backbone, architecture, loss, anchors, early stopping
  data_args     -- corpus, target loop, context, masking, length-head fitting
  gen_args      -- candidate sampling, generation length, independent judge
  training_args -- Hugging Face training, validation and checkpoint settings

Workflow:
  config JSON + optional CLI overrides, or CLI arguments alone
    -> parse -> validate/normalize -> initialise run -> return four groups

Notes:
  * Base classes in common.arguments supply shared fields; the classes below
    add MLM settings. TrainingArguments comes directly from transformers.
  * Trailing flags override JSON values; omitted fields use dataclass defaults.
    Field metadata["help"] supplies the corresponding CLI help text.
  * gen_judge scores samples with one independent reference LM (IgBert by
    default). Cross-run comparisons need the same judge and scoring context;
    self-perplexity uses a different measuring stick for each generator.
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
    """Combine shared model settings with the MLM residue-loss settings.

    Inherited: backbone, scratch architecture, anchors, early stopping.
    Added:     loss type and core/anchor weights.

    Example: loss_type="weighted_cross_entropy"
      core_loss_weight=2.0, anchor_loss_weight=1.0
      -> core residues receive twice the loss weight of anchor residues.

    Notes:
      * Plain cross_entropy ignores the core/anchor weights.
    """
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
    """Combine shared dataset settings with MLM masking and length-head fitting.

    Inherited: corpus, target loop, split, context, input length cap.
    Added:     masking policy and optional length-head training settings.

    Example: one training loop (each * represents one masked residue)
      native:    A C D E F
      full_span: * * * * *
      mixed:     * * * * *  or  A * * E *

    Notes:
      * mixed_full_span_probability selects the full-mask branch in mixed mode.
        Trainer validation always uses full_span masking.
      * Length-head fitting belongs to train_cli step 4A, after sequence
        training; it runs when do_train and train_length_predictor are both true.
    """
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
    """Combine shared generation settings with MLM decoding and length selection.

    Inherited: evaluation switch, sampling budget, residue sampling, judge.
    Added:     decoder, generation length mode/bounds, length temperature.

    Example: choosing a length before filling the residue masks
      fixed_native: use the native loop length (default)
      bounded:      sample a learned length within the supplied bounds
      inferred:     sample a learned length without supplying the native length

    Notes:
      * bounded and inferred use the trained length predictor.
      * These settings control train_cli step 5 (generate and score candidates).
        Trainer reconstruction validation is controlled by TrainingArguments.
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
    """Parse, validate and initialise an MLM run; return the four argument groups.

    Example: a JSON config plus --target_region l3
      JSON target_region: h3
      CLI override:       l3
      resolved target:    l3 -> used in both the dataset request and run name

    Returns:
      model_args, data_args, gen_args, training_args (in that order).

    Notes:
      * Creates output_dir, attaches train.log and writes run_config.json.
        output_dir and run_name are set from the resolved arguments.
      * mlm.train_cli updates latest after the datasets load successfully.
    """
    # 1.2.1 Parse the four groups: config + overrides, or CLI only.
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, gen_args, training_args = parse_config_and_cli(parser)

    # 1.2.2 Validate the shared method/context/sampling contract.
    validate_common_args(model_args, data_args, gen_args, expected_method="mlm")

    # 1.2.3 Normalize MLM masking and validate/normalize the loss settings.
    data_args.masking_policy = normalize_masking_policy(data_args.masking_policy)
    loss_config = loss_config_from_args(model_args, training_args)
    model_args.loss_type = loss_config.loss_type

    # 1.2.4 Preserve dataset metadata for the collator and MLMLossTrainer.
    # These columns need not appear in the model's forward signature.
    training_args.remove_unused_columns = False

    # 1.2.5 Check MLM masking, length-head fitting and generation-length settings.
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
    )  # Validate the request here; generation resolves it for each example later.
    if gen_args.gen_length_temperature < 0:
        raise ValueError("gen_length_temperature must be non-negative")

    # 1.2.6 Create the run directory, file log and resolved configuration.
    initialise_run(model_args, data_args, gen_args, training_args)

    # 1.2.7 Hand the configured groups back to train_cli.
    return model_args, data_args, gen_args, training_args
