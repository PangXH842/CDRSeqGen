"""Arguments configuration and setup -- MLM method.

Caller:
  mlm.train_cli.main() -> setup_args()

Arguments added to common.arguments:
  model_args
    * loss_type                   -- cross_entropy, or weighted_cross_entropy
    * [core/anchor]_loss_weight   -- loop-position weights of the weighted loss; only their ratio matters
  data_args
    * masking_policy              -- full_span, or mixed (partially revealed loops as well)
    * mixed_full_span_probability -- under mixed, the chance that an example uses the full-span mask
    * train_length_predictor      -- fit the length head after training
    * length_predictor_[epochs/learning_rate/batch_size] -- settings of that fit
  gen_args
    * gen_decoding                -- independent, or iterative, residue decoding
    * gen_length_mode             -- fixed_native, bounded or inferred generation length
    * gen_[min/max]_length        -- the bounds used by bounded
    * gen_length_temperature      -- temperature for sampling the learned length

Notes:
  * The length head is fitted after training, when do_train and train_length_predictor are both
    true; bounded and inferred generation use it, fixed-native generation does not.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.datasets.dataset_mlm import normalize_masking_policy
from cdrseqgen.methods.common.arguments import (
    BaseDataArguments,
    BaseGenerationArguments,
    BaseModelArguments,
    initialize_run,
    parse_config_and_cli,
    validate_common_args,
    validate_early_stopping,
)
from cdrseqgen.methods.mlm.backbones import resolve as resolve_backbone
from cdrseqgen.methods.mlm.length import resolve_generation_length_policy
from cdrseqgen.methods.mlm.losses import loss_config_from_args
from cdrseqgen.methods.registry import resolve as resolve_method


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
    method: str = field(default="mlm", metadata={"help": "Selects the masked-LM pipeline."})
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
      * Length-head fitting belongs to training.finish_training(), after sequence
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
      * These settings control training.finish_training() (generate and score candidates).
        Trainer reconstruction validation is controlled by TrainingArguments.
    """
    gen_decoding: str = field(
        default="independent",
        metadata={"help": "independent | iterative. Independent fills all masks in one pass; "
                          "iterative fills one per pass and can also be requested for full_span "
                          "checkpoints. Set gen_temperature "
                          "to 0 for greedy decoding."}
    )
    gen_length_mode: str = field(
        default="fixed_native",
        metadata={"help": "Length policy for post-training eval: fixed_native, bounded, or inferred."}
    )
    gen_min_length: Optional[int] = field(
        default=None, metadata={"help": "Optional lower length bound for bounded generation."}
    )
    gen_max_length: Optional[int] = field(
        default=None, metadata={"help": "Optional upper length bound for bounded generation."}
    )
    gen_length_temperature: float = field(
        default=1.0, metadata={"help": "Temperature for sampling the learned length distribution."}
    )


def validate_inference_args(args):
    """Validate the saved-checkpoint decoder options of standalone evaluate.py / generate.py.

    Called by evaluation.prepare_evaluation() and generation.prepare_generation()
    before loading weights. Reject AR-only options and decoder names outside the
    MLM registry. Leave args.decoding unchanged: evaluation.resolve_decoding()
    handles the explicit, saved or fallback choice.
    """
    if getattr(args, 'max_new_tokens', None) is not None:
        raise ValueError('--max_new_tokens applies to autoregressive decoding, not masked infilling')
    if args.decoding is not None and args.decoding not in resolve_method('mlm').decoders:
        raise ValueError('Masked infilling requires independent or iterative decoding')


def setup_args() -> Tuple[ModelArguments, DataArguments, GenerationArguments, TrainingArguments]:
    """Parse, validate and initialize an MLM run; return the four argument groups.

    Example: a JSON config plus --target_region l3
      JSON target_region: h3
      CLI override:       l3
      resolved target:    l3 -> used in both the dataset request and run name

    Returns:
      model_args, data_args, gen_args, training_args (in that order).

    Notes:
      * Creates output_dir, attaches train.log and writes run_config.json.
        output_dir and run_name are set from the resolved arguments.
      * mlm.training.prepare_training() updates latest after the datasets load successfully.
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, gen_args, training_args = parse_config_and_cli(parser)

    validate_common_args(model_args, data_args, gen_args, expected_method="mlm")
    validate_early_stopping(model_args, training_args)

    data_args.masking_policy = normalize_masking_policy(data_args.masking_policy)
    loss_config = loss_config_from_args(model_args, training_args)
    model_args.loss_type = loss_config.loss_type

    training_args.remove_unused_columns = False

    resolve_backbone(model_args.model_type).validate_training_args(model_args, data_args)
    if not 0.0 <= data_args.mixed_full_span_probability <= 1.0:
        raise ValueError("mixed_full_span_probability must be between 0 and 1")
    if (
        data_args.length_predictor_epochs <= 0
        or data_args.length_predictor_learning_rate <= 0
        or data_args.length_predictor_batch_size <= 0
    ):
        raise ValueError("length predictor epochs, learning rate, and batch size must be positive")
    gen_args.gen_decoding = gen_args.gen_decoding.strip().lower()
    decoders = resolve_method("mlm").decoders
    if gen_args.gen_decoding not in decoders:
        raise ValueError(f"gen_decoding must be one of {decoders}, got {gen_args.gen_decoding!r}")
    gen_args.gen_length_mode = gen_args.gen_length_mode.strip().lower()
    resolve_generation_length_policy(
        gen_args.gen_length_mode,
        min_length=gen_args.gen_min_length,
        max_length=gen_args.gen_max_length,
    )
    if gen_args.gen_length_temperature < 0:
        raise ValueError("gen_length_temperature must be non-negative")

    initialize_run(model_args, data_args, gen_args, training_args)
    return model_args, data_args, gen_args, training_args
