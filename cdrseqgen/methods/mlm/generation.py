"""Configure MLM case generation and connect it to the case-record builder.

Entry path:
  generate.py -> gen_cli.main -> common.generation.run_generation_pipeline
              -> prepare_generation -> generation_state -> generate_case

Callbacks:
  * prepare_generation -- validate decoder options and resolve each case's policy.
  * generation_state   -- load one saved length head if any policy needs it.
  * generate_case      -- resolve native truth/policy and build one scored record.

Candidate flow:
  generate_case -> case_generation.generate_case_record
                -> length.sample_length_conditioned_spans -> sampling decoders
                -> metrics and optional self-scoring -> display callback -> record

length.py owns policies, prompt construction, length selection and span sampling.
common.generation loads cases/model/tokenizer, seeds sampling and writes the outer
payload. gen_cli.main configures logging; this module installs no handlers.
"""

from cdrseqgen.methods.mlm.arguments import validate_inference_args
from cdrseqgen.methods.mlm.case_generation import generate_case_record
from cdrseqgen.methods.mlm.evaluation import resolve_decoding
from cdrseqgen.methods.mlm.length import LengthPredictor, resolve_case_length


def prepare_generation(args, cases, config):
    """Resolve MLM command settings before the shared pipeline loads model weights.

    Steps:
      1. validate_inference_args(): reject max_new_tokens and decoder names MLM lacks.
      2. resolve_decoding(): set args.decoding from an explicit request or a
         compatible saved setting; otherwise independent.
      3. resolve_case_length() -> one LengthPolicy per case, in input order.

    Example: cases request exact length 8 and inferred length
      return: [fixed policy, inferred policy]
      later:  generation_state() sees that a length head is required

    Recorded iterative decoding is selected automatically only for mixed masking.
    This function updates args.decoding and checks length settings; it does not
    load a model/head or generate candidates.
    """
    validate_inference_args(args)
    args.decoding = resolve_decoding(args.decoding, run_config=config)
    return [resolve_case_length(case, args)[1] for case in cases]


def generation_state(args, policies, device):
    """Return one saved length head for all cases that need it, or None.

    Policies                    Returned state
      fixed only                None
      any bounded or inferred   LengthPredictor.load(args.model_dir, device=device)

    The shared pipeline calls this after loading the sequence model and seeding
    sampling. The resulting state is passed to every generate_case() call.
    """
    return (
        LengthPredictor.load(args.model_dir, device=device)
        if any(p.requires_predictor for p in policies) else None
    )


def generate_case(model, tokenizer, device, args, generator, case, state, print_callback):
    """Generate and score one supplied framework; return its JSON-ready case record.

    Flow:
      resolve_case_length -> native truth + length policy
      case_generation.generate_case_record
        -> sample_length_conditioned_spans -> sequences
        -> eval.metrics + optional self pseudo-perplexity -> samples and metrics
        -> print_callback -> case record

    model/tokenizer/generator are already loaded and seeded by the shared pipeline.
    state is the optional predictor returned by generation_state(). The record
    builder uses length.sample_length_conditioned_spans() for every length policy.
    The shared pipeline collects records and writes the outer model/cases payload.
    """
    native, length_policy = resolve_case_length(case, args)
    return generate_case_record(
        model,
        tokenizer,
        device,
        args,
        generator,
        case,
        native=native,
        length_policy=length_policy,
        length_predictor=state,
        print_callback=print_callback,
    )
