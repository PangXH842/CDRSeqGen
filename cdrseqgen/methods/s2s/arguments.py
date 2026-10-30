"""Seq2seq run configuration: only what differs from the shared bases.

The fields every method declares identically live in
:mod:`cdrseqgen.methods.common.arguments`, along with config/CLI parsing and run-directory setup.

**What is deliberately absent**, because it has no encoder-decoder meaning: ``masking_policy`` /
``mixed_full_span_probability`` (nothing is masked out of a supervised stream), the loss registry
(§13 settled that re-weighting core against anchor moves nothing; ``label_smoothing_factor``,
which only that registry applies, is rejected), ``gen_decoding`` (one decoder,
not an independent/iterative choice), and the length policy plus ``train_length_predictor`` — the
decoder ends its own loop with ``<eos>``, so §10's gap-sentinel head has no counterpart, exactly as
on the causal side.

**What is added**: separate encoder and decoder depths. ``num_layers`` alone would make an `s2s`
run and an `ar` run with the same config look comparable while the former carries 2.3x the
parameters (800k vs 1.86M at the shipped defaults), and Findings §9 records what one un-audited
hyperparameter cost across four findings at once.
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.datasets.dataset_s2s import S2S_DECODE_STEPS, validate_s2s_decode_steps
from cdrseqgen.methods.common.arguments import (BaseDataArguments, BaseGenerationArguments,
                                                BaseModelArguments, initialize_run,
                                                parse_config_and_cli,
                                                validate_attention_heads,
                                                validate_common_args,
                                                validate_early_stopping,
                                                validate_no_label_smoothing)
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
    """Encoder-decoder training needs nothing the base does not already carry.

    Note ``max_len`` BUYS more here than on the causal side at the same value: it caps the encoder
    stream alone, which holds only framework and context, while the causal prompt must additionally
    reserve 96 positions for its answer. At max_len 128 that is 128 tokens of context here against
    31 there. The shipped configs still pin 384, matching the causal arm numerically — neither
    truncates a heavy chain plus partner at that cap, and an identical number removes the question
    from any comparison table. It is a ceiling, not a padding target, so the extra headroom is free.
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
    resolve_backbone(model_args.model_type)
    # The residue decoder is built from these settings for every backbone, around a PLM encoder too.
    validate_attention_heads(model_args, even_head_width=True)
    if (model_args.conditioning_probe_steps < 0 or not math.isfinite(model_args.diagnostic_stop_epoch)
            or model_args.diagnostic_stop_epoch < 0):
        raise ValueError("Diagnostic probe interval and stop epoch must be nonnegative and finite")
    if model_args.conditioning_probe_steps and data_args.eval_split != "validation":
        raise ValueError("Conditioning probes require eval_split=validation")
    if model_args.model_type == "scratch":
        if model_args.freeze_encoder or model_args.encoder_init != "pretrained":
            raise ValueError("freeze_encoder and encoder_init controls require a PLM backbone")
    else:
        from cdrseqgen.methods.s2s.models.plm import validate_encoder_options
        validate_encoder_options(model_args.encoder_init, model_args.freeze_encoder,
                                 model_args.num_encoder_layers)
    for name in ("num_encoder_layers", "num_decoder_layers"):
        value = getattr(model_args, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer or unset (got {value!r})")


def setup_args() -> Tuple[ModelArguments, DataArguments, GenerationArguments, TrainingArguments]:
    """Parse config/CLI, validate, then create and log the run directory."""
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, gen_args, training_args = parse_config_and_cli(parser)

    validate_args(model_args, data_args, gen_args)
    validate_early_stopping(model_args, training_args)
    validate_no_label_smoothing(training_args)

    initialize_run(model_args, data_args, gen_args, training_args, marker="s2s")
    return model_args, data_args, gen_args, training_args
