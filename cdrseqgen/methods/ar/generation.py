"""Generate AR CDR candidates and case records for supplied frameworks.

Entry path:
  generate.py -> gen_cli.main -> common.generation.run_generation_pipeline
              -> prepare_generation -> generation_state -> generate_case

Functions:
  * prepare_generation -- validate EOS options, resolve the decode cap and target region
  * generation_state   -- return None; no separate method state needs loading
  * generate_case      -- sample one case, calculate measurements and return its record

Notes:
  * gen_cli owns logging; common.generation loads cases/model/tokenizer, seeds
    sampling and writes the output. This module supplies the three method hooks.
  * eos_case_record shares case reporting with S2S. AR supplies its span-layout
    prompt, sampler and self-scorer; ar.sampling also supports native IgLM decoding.
  * EOS selects length. Native truth is optional and used only for measurements.
  * Dataset selection, judge scoring and aggregation across evaluation frameworks
    belong to the evaluation workflow.
"""
from cdrseqgen.datasets.dataset_ar import (
    CAUSAL_DECODE_STEPS, build_causal_prompt, validate_causal_decode_steps,
)
from cdrseqgen.methods.ar.arguments import validate_inference_args
from cdrseqgen.methods.ar.sampling import perplexities, sample_loops
from cdrseqgen.methods.common.eos_decoding import (
    eos_case_record, resolve_eos_decode_cap, validate_eos_case_lengths,
)


def prepare_generation(args, cases, config):
    """Validate AR generation, resolve settings in args and return None.

    Steps:
      validate_inference_args -> validate_eos_case_lengths -> resolve_eos_decode_cap
                       -> set the public target region for layouts that use it

    max_new_tokens comes from args, then saved gen_max_new_tokens, then the AR
    default. target_region comes from target_loop or the checkpoint; native IgLM
    may use it to select a chain token. No model loading or sampling happens here.
    """
    validate_inference_args(args)
    validate_eos_case_lengths(args, cases, method="AR")
    args.max_new_tokens = getattr(args, 'max_new_tokens', None)
    resolve_eos_decode_cap(args, config, default=CAUSAL_DECODE_STEPS,
                          validate=validate_causal_decode_steps)
    # Only a layout that declares target_region as a public field reads this (IgLM's chain token).
    args.target_region = getattr(args, 'target_loop', None) or config.get('target_region')
    return None


def generation_state(args, policies, device):
    """Return None; AR needs no separately loaded length head or method state.

    The shared pipeline calls this after loading the model and passes its return
    value to generate_case(). The arguments follow the same hook contract as MLM.
    """
    return None


def generate_case(model, tokenizer, device, args, generator, case, state, print_callback):
    """Return candidates, measurements and attempt accounting for one supplied case.

    eos_case_record() coordinates:
      build_causal_prompt -> sample_loops -> optional generator perplexities
                          -> sequence metrics -> print_callback -> case record

    state is unused. Native truth/length stays out of the generation framework;
    truth supports reference measurements after sampling. Failed attempts remain
    in accounting; completed nonempty candidates are retained with duplicates.
    """
    return eos_case_record(model, tokenizer, device, args, generator, case, print_callback,
                           mode='causal', build_context=build_causal_prompt,
                           sample_loops=sample_loops, perplexities=perplexities)
