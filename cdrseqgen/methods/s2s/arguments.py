"""Arguments configuration and setup -- S2S method.

Caller:
  s2s.train_cli.main() -> setup_args()

Arguments added to common.arguments:
  model_args
    * num_[encoder/decoder]_layers -- separate stack depths (scratch encoder; decoder, unset = num_layers)
    * encoder_init                -- PLM encoder pretrained, or random (a same-architecture control)
    * freeze_encoder              -- keep the PLM encoder's weights and dropout fixed
    * conditioning_probe_steps    -- fixed validation conditioning probes; 0 disables
    * diagnostic_stop_epoch       -- stop at this epoch without shortening the LR schedule; 0 disables
  gen_args
    * gen_max_new_tokens          -- decode steps including <eos>; a draw that never emits it is dropped

Notes:
  * Separate encoder/decoder depths help control parameter count in method comparisons;
    the same num_layers value does not give AR and S2S the same number of parameters.
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.datasets.dataset_s2s import S2S_DECODE_STEPS, validate_s2s_decode_steps
from cdrseqgen.methods.common.arguments import (
    BaseDataArguments,
    BaseGenerationArguments,
    BaseModelArguments,
    initialize_run,
    parse_config_and_cli,
    validate_common_args,
    validate_early_stopping,
    validate_no_label_smoothing,
)
from cdrseqgen.methods.s2s.backbones import resolve as resolve_backbone


@dataclass
class ModelArguments(BaseModelArguments):
    method: str = field(default="s2s", metadata={"help": "Selects the encoder-decoder pipeline."})
    conditioning_probe_steps: int = field(
        default=0, metadata={"help": "0 disables fixed validation conditioning probes."})
    diagnostic_stop_epoch: float = field(
        default=0.0,
        metadata={"help": "Stop at this epoch without shortening the LR schedule; 0 disables."})
    encoder_init: str = field(
        default="pretrained", metadata={"help": "PLM encoder initialization: pretrained or random "
                                                "(same-architecture control)."})
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Keep PLM encoder weights and dropout fixed; train only "
                                        "the projection and residue decoder."})
    num_encoder_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Scratch encoder depth; leave unset for PLMs. Set both this and num_decoder_layers "
                          "to compare against an `ar` run at matched parameter count."})
    hidden_size: int = field(
        default=BaseModelArguments.hidden_size,
        metadata={"help": "Scratch width, or PLM residue decoder width."})
    num_layers: int = field(
        default=BaseModelArguments.num_layers,
        metadata={"help": "Default scratch stack depth or PLM decoder depth."})
    num_decoder_layers: Optional[int] = field(
        default=None, metadata={"help": "Decoder depth; None = num_layers."})


@dataclass
class DataArguments(BaseDataArguments):
    """Use shared data settings; max_len caps only the encoder context.

    AR also reserves room for its answer inside max_len. Equal caps therefore
    need not allow equal amounts of context. max_len is a ceiling, not a fixed
    padding length.
    """


@dataclass
class GenerationArguments(BaseGenerationArguments):
    gen_max_new_tokens: int = field(
        default=S2S_DECODE_STEPS,
        metadata={"help": "Decode steps including <eos>. A draw that never emits <eos> within "
                          "the cap is incomplete and is dropped rather than scored as a loop."})


def validate_inference_args(args):
    """Validate saved-checkpoint decoder/length options and set args.decoding.

    Called by eval_cli.prepare_evaluation and gen_cli.prepare_generation.
    """
    if args.decoding not in (None, 'seq2seq'):
        raise ValueError('S2S checkpoints require seq2seq decoding')
    if args.min_length is not None or args.max_length is not None:
        raise ValueError('S2S chooses length via EOS; bounded/fixed length is unsupported')
    if args.length_temperature != 1.0:
        raise ValueError('S2S has no separate length-temperature control')
    args.decoding = 'seq2seq'


def validate_args(model_args, data_args, gen_args):
    """Validate the seq2seq-only limits and backbone on top of the shared method contract."""
    validate_common_args(model_args, data_args, gen_args, expected_method="s2s")
    validate_s2s_decode_steps(gen_args.gen_max_new_tokens)
    resolve_backbone(model_args.model_type).validate_training_args(model_args, data_args)
    if (model_args.conditioning_probe_steps < 0 or not math.isfinite(model_args.diagnostic_stop_epoch)
            or model_args.diagnostic_stop_epoch < 0):
        raise ValueError("Diagnostic probe interval and stop epoch must be nonnegative and finite")
    if model_args.conditioning_probe_steps and data_args.eval_split != "validation":
        raise ValueError("Conditioning probes require eval_split=validation")


def setup_args() -> Tuple[ModelArguments, DataArguments, GenerationArguments, TrainingArguments]:
    """Parse config/CLI, validate, then create and log the run directory."""
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, gen_args, training_args = parse_config_and_cli(parser)

    validate_args(model_args, data_args, gen_args)
    validate_early_stopping(model_args, training_args)
    validate_no_label_smoothing(training_args)

    initialize_run(model_args, data_args, gen_args, training_args, marker="s2s")
    return model_args, data_args, gen_args, training_args
