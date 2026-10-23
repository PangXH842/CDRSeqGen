"""Set up MLM evaluation, generate candidate loops and coordinate their scoring.

Entry paths:
  * Standalone: evaluate.py -> mlm.eval_cli.main
               -> common.evaluation.run_evaluation_pipeline -> run_generative_eval
  * After training: mlm.training -> common.evaluation.post_training_generation
                    -> mlm.eval_cli.evaluate (alias for run_generative_eval)
  Both paths supply a loaded model, tokenizer, dataset and optional judge.

CLI setup (defined here and re-exported by mlm.eval_cli):
  * prepare_evaluation(args, config)
      Set length/decoder defaults in args and check MLM-specific options.
      Calls resolve_generation_length_policy() and resolve_decoding().
  * resolve_decoding(requested, ...)
      Choose an explicit decoder, a compatible saved choice, or independent.
      Called by prepare_evaluation(); returns the decoder name.
  * evaluation_options(args, device)
      Load LengthPredictor for bounded/inferred modes and return evaluator kwargs.
      For fixed_native, the returned length_predictor is None.
  The shared standalone pipeline calls prepare_evaluation before loading resources,
  then evaluation_options after loading the model. Post-training callers supply
  their evaluator options directly.

run_generative_eval (main evaluation flow):
  1. Put the model in evaluation mode, validate anchor_trim and seed sampling.
  2. select_indices() -> framework indices, optionally balanced by native length.
  3. resolve_generation_length_policy() -> choose the generation strategy:
       fixed_native     -> FixedNativeStrategy (native mask count)
       bounded/inferred -> VariableLengthStrategy (fitted length head)
  4. run_eval_loop(strategy, selected) -> one scored record per returned Draw:
       strategy.draw(i)      -> candidate sequences and their prompts
       metrics.calculate()  -> sequence, length and diversity measurements
       strategy.self_scores -> optional generator pseudo-perplexities
       score_with_judge()   -> optional judge scores via strategy.judge_inputs()
       positional/confidence metrics -> candidate details + framework record
  5. log_report() -> overall and length-bucket report.
     aggregate() -> summary means across contributing framework records.
     write_payload() / write_confidence_figure() -> optional JSON/PNG output.
     Return the payload dict, including records, settings and supplied metadata.

Functions behind candidate generation:
  * FixedNativeStrategy.draw -> masked_view -> sampling.sample_masked_span
      masked_view extracts prompt IDs, supervised positions and native token IDs.
      Existing mask positions expose the native loop length to generation.
  * VariableLengthStrategy.draw -> datasets.dataset_mlm.framework_for_dataset
      -> generation.sample_length_conditioned_spans
      Framework context and the length head determine candidate lengths; native
      sequences/lengths remain available for selection and scoring.
  * Both strategies return Draw objects and implement self_scores(), judge_inputs()
    and payload_length_policy() for the shared scoring loop and output metadata.

Ownership:
  * common.evaluation handles checkpoint checks, resource loading and context views.
  * cdrseqgen.eval.metrics defines measurements; eval.generative owns the scoring
    loop, aggregation and output helpers. Imported helpers remain re-exported here.
  * mlm.sampling supplies residue decoders and pseudo_log_likelihood().
  * mlm.recovery calculates Trainer validation metrics during training; this module
    evaluates generated candidates in a separate stage.

Logging:
  * Root evaluate.py calls common.logging.configure_logging() for console output.
  * eval.generative logs reports, scoring warnings and output-write messages.
  * Training setup attaches a train.log handler, which also records post-training
    evaluation logs. Standalone evaluation does not attach a file handler.
  * resolve_decoding uses print() for console messages; those bypass log handlers.
"""

import random

import torch

from cdrseqgen.artifacts import read_run_config
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.eval.generative import (  # noqa: F401 (compatibility re-exports)
    SUMMARY_KEYS,
    Draw,
    aggregate,
    bucket_of,
    build_judge,
    format_report,
    log_report,
    run_eval_loop,
    score_with_judge,
    select_indices,
    write_confidence_figure,
    write_payload,
)
from cdrseqgen.eval.metrics import (
    LENGTH_BUCKETS,  # noqa: F401 (compatibility re-export)
    blosum_provenance,
)
from cdrseqgen.methods.mlm.length import (
    native_mask_length,
    resolve_generation_length_policy,
)
from cdrseqgen.methods.mlm.length_predictor import LengthPredictor
from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood, sample_masked_span


def resolve_decoding(requested, model_dir=None, run_config=None):
    """Resolve an MLM decoder, using independent decoding as the fallback.

    Generation decoders:
      * independent -- fill all masks from one forward pass over the same prompt. (default)
      * iterative   -- fill the most-confident remaining mask per pass; filled
                       residues become context for later predictions.

    Training masking policies:
      * full_span -- hide the entire target loop. (default)
      * mixed     -- mix fully masked loops with partially revealed native loops.

    Selection:
      1. Use an explicitly requested decoder.
      2. Otherwise use a compatible saved gen_decoding.
      3. Otherwise fall back to independent, including for mixed checkpoints.

    Notes:
      * run_config takes precedence over metadata read from model_dir.
      * Missing masking metadata is treated as full_span. A recorded iterative
        choice is used automatically only with mixed training; this preserves
        the safeguard against legacy full-span iterative defaults. An explicit
        iterative request still overrides it.
      * The caller validates decoder names; this function does not reject them.
      * Prints the selection and returns the decoder name without generating
        candidates.
    """
    meta = run_config if run_config is not None else (read_run_config(model_dir) or {})
    recorded = meta.get("gen_decoding")
    policy = (meta.get("masking_policy") or "full_span").strip().lower()
    if requested is None:
        chosen = "independent"
        if policy == "mixed" and recorded in {"independent", "iterative"}:
            chosen = recorded
        source = f"masking_policy={policy!r}"
        if recorded and recorded != chosen:
            source += f"; ignored incompatible recorded gen_decoding={recorded!r}"
        print(f"🎲 Decoding: {chosen} ({source})")
    else:
        chosen = requested
        if recorded and recorded != chosen:
            print(f"🎲 Decoding: {chosen} (CLI override; the run recorded '{recorded}')")
        else:
            print(f"🎲 Decoding: {chosen}")
    return chosen


def prepare_evaluation(args, config):
    """Resolve MLM command options in place before the shared pipeline loads resources.

    Example: args.length_mode=None, args.decoding=None, config={}
      length_mode -> fixed_native
      decoding    -> independent

    Calls length.resolve_generation_length_policy() to check the length request,
    then resolve_decoding() to apply the explicit/saved/fallback decoder choice.
    Rejects max_new_tokens and causal decoding, which do not apply to MLM.
    Returns None; subsequent stages read the updated args.
    """
    args.length_mode = args.length_mode or 'fixed_native'
    if args.max_new_tokens is not None:
        raise ValueError('--max_new_tokens applies to autoregressive decoding, not masked infilling')
    if args.decoding == 'causal':
        raise ValueError('Masked infilling requires independent or iterative decoding')
    resolve_generation_length_policy(args.length_mode, min_length=args.min_length, max_length=args.max_length)
    args.decoding = resolve_decoding(args.decoding, run_config=config)


def evaluation_options(args, device):
    """Build run_generative_eval kwargs from the resolved command options.

    Length mode       length_predictor in returned dict
      fixed_native    None
      bounded         LengthPredictor.load(args.model_dir, device=device)
      inferred        LengthPredictor.load(args.model_dir, device=device)

    Called by the shared standalone pipeline after prepare_evaluation() and model
    loading. Also forwards decoding, bounds and length_temperature. The saved
    length head is loaded here; framework selection and generation happen later.
    """
    predictor = None
    if args.length_mode != 'fixed_native':
        predictor = LengthPredictor.load(args.model_dir, device=device)
    return dict(decoding=args.decoding, length_mode=args.length_mode, length_predictor=predictor,
                min_length=args.min_length, max_length=args.max_length,
                length_temperature=args.length_temperature)


def masked_view(item):
    """Return prompt IDs, supervised positions and native IDs as three lists.

    Example: labels use -100 for positions excluded from the loss
      input_ids:      [10, 99, 99, 20]    (99 is a mask in this example)
      labels:         [-100, 3, 4, -100]
      mask_positions: [1, 2]
      native_ids:     [3, 4]

    Positions come from labels, without checking the input token at each position.
    Fixed-native generation expects the dataset's fully masked evaluation view.
    """
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    mask_positions = [p for p, l in enumerate(labels) if l != -100]
    native_ids = [labels[p] for p in mask_positions]
    return input_ids, mask_positions, native_ids


class FixedNativeStrategy:
    """Adapt the dataset's native-length masked prompt to the shared scoring loop.

    Example: three supervised loop positions -> three masks -> length-3 candidates.

    The supplied dataset determines partner/antigen context, including any ablation.
    Candidate generation receives the native mask count. Each candidate retains its
    prompt for self-scoring; the judge uses a separate prompt from judge_dataset.
    """

    supplies_native_length = True

    def __init__(self, *, model, tokenizer, device, generator, eval_dataset, judge_dataset,
                 judge_tokenizer, decoding, num_samples, temperature, top_k, top_p):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.generator = generator
        self.eval_dataset, self.judge_dataset = eval_dataset, judge_dataset
        self.judge_tokenizer = judge_tokenizer
        self.decoding, self.num_samples = decoding, num_samples
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self._judge_cache = (None, None)

    def payload_length_policy(self):
        """Describe the native-mask-count policy for the evaluation payload."""
        return {"mode": "fixed", "source": "native_mask_count"}

    def draw(self, i):
        """Return candidates and scoring prompts for index i, or None if no masks exist."""
        input_ids, mask_positions, native_ids = masked_view(self.eval_dataset[i])
        if not mask_positions:
            return None
        length_decision = native_mask_length(len(mask_positions))
        native = self.tokenizer.decode(native_ids).upper().strip()
        sampled_ids = sample_masked_span(
            self.model, self.tokenizer, self.device, self.generator,
            input_ids, mask_positions,
            decoding=self.decoding, num_samples=self.num_samples,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
        )
        spans = [(input_ids, mask_positions, ids) for ids in sampled_ids]
        seqs = [self.tokenizer.decode(ids).upper().strip() for ids in sampled_ids]
        return Draw(
            native=native,
            seqs=seqs,
            spans=spans,
            target_len=length_decision.length,
            length_policy=length_decision.as_dict(),
            record_extra={},
        )

    def self_scores(self, drawn):
        """Return generator pseudo-perplexities in candidate order.

        Each filled loop is re-masked one position at a time; the other candidate
        residues remain visible. sampling.pseudo_log_likelihood performs the scoring.
        """
        return [
            pseudo_log_likelihood(self.model, input_ids, mask_positions, token_ids,
                                  self.tokenizer.special_ids, self.tokenizer.mask_id, self.device)[1]
            for input_ids, mask_positions, token_ids in drawn.spans
        ]

    def judge_inputs(self, i, _seq):
        """Return and cache the judge's masked prompt for evaluation index i.

        All candidates share the native length, so _seq does not change this prompt.
        A Subset index is mapped back to the corresponding judge_dataset entry.
        """
        cached_index, cached_prompt = self._judge_cache
        if cached_index == i:
            return cached_prompt
        # Resolve one outer Subset layer; the judge dataset uses the full corpus order.
        full_idx = self.eval_dataset.indices[i] if hasattr(self.eval_dataset, "indices") else i
        judge_input, judge_mask_positions, _ = masked_view(self.judge_dataset[full_idx])
        self._judge_cache = (i, (judge_input, judge_mask_positions))
        return self._judge_cache[1]


def run_generative_eval(model, tokenizer, eval_dataset, *, device,
                        decoding="independent", num_samples=20,
                        temperature=1.0, top_k=0, top_p=1.0,
                        length_mode="fixed_native", length_predictor=None,
                        min_length=None, max_length=None,
                        length_temperature=1.0,
                        num_examples=200, stratify=False, seed=42,
                        skip_perplexity=False,
                        judge_model=None, judge_tokenizer=None,
                        judge_dataset=None, judge_name=None,
                        anchor_trim=ANCHOR_TRIM,
                        output_path=None, meta=None, title=""):
    """Generate candidate loops, score them and return an evaluation payload dict.

    Inputs:
      Loaded model/tokenizer/eval_dataset, sampling settings and device.
      Bounded/inferred modes also need a fitted length_predictor. Judge scoring
      needs judge_model, judge_tokenizer and the corresponding judge_dataset.
      eval_cli.evaluate is an alias for this function.

    Flow:
      1. model.eval(), validate anchor_trim and seed Torch/local sampling RNGs.
      2. Read target_len from the dataset and call select_indices() using
         num_examples, stratify and a seeded Python RNG.
      3. Resolve the length policy and construct one strategy for the run:
           fixed_native     -> FixedNativeStrategy
           bounded/inferred -> VariableLengthStrategy
      4. run_eval_loop() calls the strategy for each selected framework:
           draw -> sequence metrics -> optional self/judge scores
                -> positional/confidence metrics -> framework record
      5. Log the report; combine records, aggregate means, decoding/length settings,
         BLOSUM provenance and caller metadata; invoke the JSON/figure writers.

    Example: two selected frameworks, num_samples=20, both draws succeed
      records:     2 framework records, each with 20 candidate sequences
      n_generated: 2                 (framework records, not candidate count)
      summary:     metric means across contributing framework records

    Notes:
      * Native lengths support selection/reporting. Only fixed_native supplies
        the native mask count to generation; bounded/inferred use the length head.
      * skip_perplexity disables self-scoring; judge scoring is independent.
      * A strategy returning None contributes no record.
      * output_path=None skips file output; the payload is still returned.
      * Generation/scoring errors propagate. Output helpers handle some write/plot
        failures with warnings. Model evaluation mode and the Torch seed are not
        restored on return.
    """
    model.eval()
    if isinstance(anchor_trim, bool) or not isinstance(anchor_trim, int) or anchor_trim < 0:
        raise ValueError("anchor_trim must be a non-negative integer")

    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # This length list selects examples; each strategy controls its generation inputs.
    lengths = [eval_dataset[i]["target_len"] for i in range(len(eval_dataset))]
    rng = random.Random(seed)
    selected = select_indices(lengths, num_examples, stratify, rng)

    length_policy = resolve_generation_length_policy(
        length_mode, min_length=min_length, max_length=max_length,
    )
    if length_policy is not None:
        if length_predictor is None:
            raise FileNotFoundError("variable-length evaluation requires a length predictor")
        from cdrseqgen.methods.mlm.variable_length_evaluation import VariableLengthStrategy
        strategy = VariableLengthStrategy(
            model=model, tokenizer=tokenizer, device=device, generator=generator,
            eval_dataset=eval_dataset, judge_dataset=judge_dataset,
            judge_tokenizer=judge_tokenizer, length_predictor=length_predictor,
            length_policy=length_policy, length_temperature=length_temperature,
            decoding=decoding, num_samples=num_samples, temperature=temperature,
            top_k=top_k, top_p=top_p,
        )
    else:
        strategy = FixedNativeStrategy(
            model=model, tokenizer=tokenizer, device=device, generator=generator,
            eval_dataset=eval_dataset, judge_dataset=judge_dataset,
            judge_tokenizer=judge_tokenizer, decoding=decoding, num_samples=num_samples,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

    records = run_eval_loop(
        strategy, selected,
        model=model, tokenizer=tokenizer, device=device,
        skip_perplexity=skip_perplexity,
        judge_model=judge_model, judge_tokenizer=judge_tokenizer,
        anchor_trim=anchor_trim,
    )

    log_report(records, title=title, judge_name=judge_name)

    payload = dict(meta or {})
    payload.update({
        "schema_version": "cdrseqgen.generation_eval.v1",
        # A BLOSUM number without its matrix and gap penalties is not reproducible.
        "blosum": blosum_provenance(),
        "judge": judge_name,
        "length_policy": strategy.payload_length_policy(),
        "anchor_trim_each_end": anchor_trim,
        "decoding": {
            "mode": decoding, "num_samples": num_samples, "temperature": temperature,
            "top_k": top_k, "top_p": top_p, "seed": seed,
        },
        "n_eval_total": len(eval_dataset),
        "n_generated": len(records),
        "summary": {key: aggregate(records, key)[0] for key, _ in SUMMARY_KEYS},
        "records": records,
    })

    write_payload(payload, output_path, len(records))
    write_confidence_figure(payload, output_path, title)
    return payload


