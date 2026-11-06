"""Share EOS-based training, candidate evaluation and case reporting for AR/S2S.

Command setup:
  * resolve_eos_decode_cap    -- resolve/validate the generation or evaluation cap
  * validate_eos_case_lengths -- reject unsupported lengths in supplied cases
                               -> called by: ar/s2s.generation.prepare_generation()
Training:
  * get_infilling_datasets  -- tokenizer and train/eval subsets with separate context views
                             -> called by: ar/s2s.training.get_datasets()
  * eos_recovery_metrics    -- token-pooled recovery from already aligned predictions/labels
                             -> called by: ar/s2s.training.compute_metrics()
Evaluation:
  * EosDecodingStrategy     -- base for AR's CausalStrategy and S2S's Seq2SeqStrategy
  * run_eos_generative_eval -- select frameworks, run shared scoring and return a payload
                             -> called by: ar/s2s.evaluation.run_generative_eval() 
Generation:
  * eos_case_record         -- generate and score one supplied framework case
                             -> called by: ar/s2s.generation.generate_case() 

Notes:
  * Each method supplies its context builder, sampler and self-scorer. EOS determines
    generated length; native targets support selection/scoring. Failed draws stay in
    attempt accounting even when they produce no scored candidate record.
    EOS means the tokenizer's end marker, including native IgLM's [CLS] token.
"""
import logging
import random
from collections.abc import Mapping

import numpy as np
import torch

from cdrseqgen.artifacts import GENERATION_EVAL_SCHEMA
from cdrseqgen.datasets.dataset_common import (
    evaluation_context_view, framework_for_dataset, select_split_indices, unwrap_subsets,
)
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.eval.generative import (
    Draw, SUMMARY_KEYS, aggregate, build_masked_judge_prompt, log_report, normalize_scores, run_eval_loop,
    select_indices, write_confidence_figure, write_payload,
)
from cdrseqgen.eval.metrics import (
    amino_acid_recovery, blosum_provenance, calculate, core_anchor_indices,
)
from cdrseqgen.methods.common.arguments import validate_anchor_trim

logger = logging.getLogger(__name__)


def resolve_eos_decode_cap(args, config, *, default, validate):
    """Resolve args.max_new_tokens in place, validate it and return the cap.

    Priority: explicit args value -> saved gen_max_new_tokens -> method default.
    Called by ar/s2s prepare_generation() and prepare_evaluation(). Each method
    supplies its own default and validator. A saved None is retained and validated,
    rather than treated as an absent key. The cap is assigned before validation,
    preserving the resolved settings when a validator raises.
    """
    if args.max_new_tokens is None:
        args.max_new_tokens = config.get("gen_max_new_tokens", default)
    validate(args.max_new_tokens)
    return args.max_new_tokens


def validate_eos_case_lengths(args, cases, *, method):
    """Reject requested fixed/bounded/native lengths for EOS-generated cases.

    A present case value overrides its CLI counterpart, including None or False.
    Only those two values disable a control; numerical zero remains unsupported.
    method labels errors as AR or S2S. Checks run before checkpoint loading.
    """
    for case in cases:
        for key in ("min_length", "max_length", "target_length", "use_native_length"):
            value = case.get(key, getattr(args, key, None))
            if value is not None and value is not False:
                raise ValueError(f"{method} chooses length via EOS; {key} is unsupported")


def _invalid_draw(result):
    """Identify a terminated but invalid draw, separate from an unterminated one."""
    return bool(result.get("terminated", False)) and not result.get("valid", result["complete"])


def _partition_attempts(results):
    """Return retained draw dicts and counts shared by evaluation and case reporting.

    Retain completed nonempty loops in their original order, including duplicates.
    Count completed empty loops, terminated invalid draws and incomplete draws
    separately. Each result must supply a string loop and boolean complete.
    Optional valid/terminated flags must agree with completion; contradictory
    records raise ValueError rather than corrupting attempt counts.
    """
    kept = []
    empty = invalid = 0
    for result in results:
        if not isinstance(result, Mapping) or not isinstance(result.get("loop"), str):
            raise ValueError("sampler attempts must provide a string loop")
        if not isinstance(result.get("complete"), bool):
            raise ValueError("sampler attempts must provide a boolean complete flag")
        for flag in ("valid", "terminated"):
            if flag in result and not isinstance(result[flag], bool):
                raise ValueError(f"sampler attempt {flag} must be a boolean")
        if "valid" in result and result["valid"] != result["complete"]:
            raise ValueError("sampler attempt valid and complete flags disagree")
        if result["complete"] and result.get("terminated") is False:
            raise ValueError("a complete sampler attempt cannot be unterminated")
        if result["complete"]:
            if result["loop"]:
                kept.append(result)
            else:
                empty += 1
        invalid += _invalid_draw(result)
    return kept, {
        "attempts": len(results),
        "complete_nonempty": len(kept),
        "complete_empty": empty,
        "incomplete": len(results) - len(kept) - empty - invalid,
        "invalid": invalid,
    }


# --- training: metrics and datasets ---------------------------------------------------------------
def eos_recovery_metrics(preds, gold, *, trim):
    """Return token-pooled whole/core/anchor recovery and row-based core coverage.

    Called by ar.training.compute_metrics() after its next-token shift, and by
    s2s.training.compute_metrics() with predictions already aligned. Both inputs
    are token-ID arrays [rows, positions]; gold=-100 marks ignored positions.

    Example: trim=1, with EOS in the last supervised position
      gold:        -100 A C D EOS -100
      predictions:   ? A Q D  ?     ?
      regions:         A C A          (A=anchor, C=core)
      accuracy=2/3, core_accuracy=0/1, anchor_accuracy=2/2

    Drop each row's final supervised token (the dataset's terminator). Pool token
    hits/counts across rows; do not average per-row recovery. core_coverage is the
    fraction of supervised rows containing a core. EOS-only rows count in that
    denominator; entirely ignored rows do not. Callers validate trim beforehand.
    """
    whole_hits = whole_total = core_hits = core_total = anchor_hits = anchor_total = 0
    rows = rows_with_core = 0
    for row_preds, row_gold in zip(preds, gold):
        positions = np.flatnonzero(row_gold != -100)
        if positions.size == 0:
            continue
        rows += 1
        residues = positions[:-1]                      # drop the trailing <eos>
        if residues.size == 0:
            continue
        correct = row_preds[residues] == row_gold[residues]
        whole_hits += int(correct.sum())
        whole_total += residues.size

        core, anchor = core_anchor_indices(residues.size, trim)
        if core:
            rows_with_core += 1
            core_hits += int(correct[core].sum())
            core_total += len(core)
        anchor_hits += int(correct[anchor].sum())
        anchor_total += len(anchor)

    return {
        "accuracy": whole_hits / whole_total if whole_total else 0.0,
        "core_accuracy": core_hits / core_total if core_total else 0.0,
        "anchor_accuracy": anchor_hits / anchor_total if anchor_total else 0.0,
        "core_coverage": rows_with_core / rows if rows else 0.0,
    }


def get_infilling_datasets(dataset_class, kind, model_args, data_args, training_args, backbone, *,
                           checkpoint_path=None):
    """Return (train subset, eval subset, tokenizer) for AR/S2S training or evaluation.

    Called by each method's training.get_datasets(). dataset_class supplies its
    input/label format; kind only labels the log message. checkpoint_path selects
    a saved tokenizer when evaluating an existing checkpoint.

    Flow: tokenizer -> parse records once -> select split indices -> subset views.
    The evaluation view shares records but disables context dropout. The shared
    split guard rejects implicit test selection; explicitly requesting test is
    allowed. These are datasets; Trainer creates their DataLoaders later.
    """
    tokenizer = backbone.build_tokenizer(model_args.model_type, checkpoint_path=checkpoint_path)
    full = dataset_class(
        data_args.data_path,
        tokenizer,
        max_len=data_args.max_len,
        target_region=data_args.target_region,
        include_partner_chain=data_args.include_partner_chain,
        include_antigen=data_args.include_antigen,
        context_dropout=data_args.context_dropout,
        context_dropout_seed=training_args.seed,
    )
    train_idx, eval_idx, eval_label = select_split_indices(
        full.data, data_args.eval_split, data_path=data_args.data_path,
        seed=training_args.seed, test_size=data_args.test_size)

    # Share records while keeping training context dropout out of validation.
    train_ds = torch.utils.data.Subset(full, train_idx)
    eval_ds = torch.utils.data.Subset(evaluation_context_view(full), eval_idx)
    logger.info(f"📊 Data loaded: {len(train_ds)} train ({kind}), "
                f"{len(eval_ds)} eval (split: {eval_label}).")
    return train_ds, eval_ds, tokenizer


# --- evaluation: strategy for the shared loop in cdrseqgen.eval.generative ------------------------
class EosDecodingStrategy:
    """Adapt AR/S2S generation to eval.generative.run_eval_loop().

    Subclasses: ar.evaluation.CausalStrategy, s2s.evaluation.Seq2SeqStrategy.
    Each supplies three hooks:
      * context(framework) -- build the training-compatible prompt or encoder IDs
      * sample(context)   -- return raw draw dictionaries from the method sampler
      * score(spans)      -- self-perplexities for ordered (context, loop IDs) pairs

    Shared flow:
      draw(i) -> framework -> context -> sample -> account for every attempt
              -> Draw of retained candidates, or None when none remain
      self_scores(Draw) -> method self-scorer
      judge_inputs(i, seq) -> masked judge prompt at this candidate's length

    Counters include failed frameworks. Native-protocol tokenizers also retain
    raw draw details. The judge dataset is the full corpus or the eval dataset
    object itself; evaluation indices may pass through nested Subsets.
    """

    supplies_native_length = False

    def __init__(self, *, model, tokenizer, device, generator, eval_dataset, judge_dataset,
                 judge_tokenizer, num_samples, temperature, top_k, top_p, max_new_tokens):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.generator = generator
        self.eval_dataset, self.judge_dataset = eval_dataset, judge_dataset
        self.judge_tokenizer = judge_tokenizer
        self.num_samples, self.temperature = num_samples, temperature
        self.top_k, self.top_p = top_k, top_p
        self.max_new_tokens = max_new_tokens
        self._framework_cache = (None, None)
        self.attempts = 0
        self.complete_nonempty = 0
        self.complete_empty = 0
        self.incomplete = 0
        self.invalid = 0
        self.generation_attempts = []

        self.length_matches = 0
        self.length_absolute_error_capped = 0

    # --- the three hooks a method supplies --------------------------------------------------
    def context(self, framework):
        """Build the method's conditioning input from public framework context."""
        raise NotImplementedError("subclasses build the sampler's conditioning input")

    def sample(self, context):
        """Return raw sampler results using this strategy's sampling options."""
        raise NotImplementedError("subclasses call their sample_loops")

    def score(self, spans):
        """Return one self-perplexity per ordered (context, loop-token IDs) pair."""
        raise NotImplementedError("subclasses call their perplexities")

    def sampling_options(self):
        """Return shared sampler kwargs, including the seeded generator and decode cap."""
        return dict(device=self.device, num_samples=self.num_samples, temperature=self.temperature,
                    top_k=self.top_k, top_p=self.top_p, max_new_tokens=self.max_new_tokens,
                    generator=self.generator)

    # --- framework lookup -------------------------------------------------------------------
    def framework(self, i):
        """Resolve an eval-view index once; cache the most recently requested framework.

        framework_for_dataset() unwraps Subsets itself. Pass i unchanged here.
        """
        cached_index, cached = self._framework_cache
        if cached_index != i:
            cached = framework_for_dataset(self.eval_dataset, i)
            self._framework_cache = (i, cached)
        return cached

    def _judge_framework(self, i):
        """Map an eval-view index to the corresponding full-corpus judge framework.

        Example: eval=Subset(Subset(full, [4, 1, 5]), [2, 0])
          eval index 0 -> inner index 2 -> full judge index 5.
        An identical eval/judge dataset uses the eval lookup directly.
        """
        if self.judge_dataset is self.eval_dataset:
            return self.framework(i)
        _, full_index = unwrap_subsets(self.eval_dataset, i)
        return framework_for_dataset(self.judge_dataset, full_index)

    # --- the strategy protocol --------------------------------------------------------------
    def payload_length_policy(self):
        """Describe generated length as inferred by EOS termination."""
        return {"mode": "inferred", "source": "eos_token"}

    def draw(self, i):
        """Generate candidates for eval index i and update cumulative attempt counts.

        Only completed nonempty loops enter Draw.seqs; failed/empty draws still
        contribute to counters. Return None after accounting when none survive.
        Capped length error includes all attempts and penalizes each failed draw
        by at least one residue. Native length supports scoring only.
        """
        framework = self.framework(i)
        context = self.context(framework)
        native = framework["target"].upper().strip()
        results = self.sample(context)

        kept, counts = _partition_attempts(results)
        native_length = len(native)
        length_matches = sum(len(r["loop"]) == native_length for r in kept)
        capped_error_sum = sum(
            max(1, abs(len(r["loop"]) - native_length)) if not r["complete"]
            else abs(len(r["loop"]) - native_length) for r in results)
        self.attempts += counts["attempts"]
        self.complete_nonempty += counts["complete_nonempty"]
        self.complete_empty += counts["complete_empty"]
        self.incomplete += counts["incomplete"]
        self.invalid += counts["invalid"]
        self.length_matches += length_matches
        self.length_absolute_error_capped += capped_error_sum
        prompt_meta = getattr(self, "prompt_metadata", {})
        native_protocol = getattr(self.tokenizer, "causal_sampling_metadata", None)
        if native_protocol is not None:
            self.generation_attempts.append({
                "dataset_index": int(i), "conditioning": prompt_meta.get("conditioning"),
                "draws": results,
            })
        if not kept:
            return None
        seqs = [r["loop"].upper() for r in kept]
        record_extra = {
            "attempts": counts["attempts"], "incomplete": counts["incomplete"],
            "empty": counts["complete_empty"],
            "length_matches": length_matches,
            "length_absolute_error_sum_capped": capped_error_sum,
        }
        if native_protocol is not None:
            record_extra["invalid"] = counts["invalid"]
            record_extra["conditioning"] = prompt_meta.get("conditioning")
        return Draw(
            native=native,
            seqs=seqs,
            spans=[(context, self.tokenizer.encode(s)) for s in seqs],
            target_len=native_length,
            length_policy=self.payload_length_policy(),
            record_extra=record_extra,
        )

    @torch.no_grad()
    def self_scores(self, drawn):
        """Delegate candidate self-perplexity scoring to the method's score() hook.

        Each score is exp(mean answer-token NLL), including the terminator.
        AR/S2S currently use one forward pass per candidate; S2S reuses its encoder
        output across candidates from the same framework. No gradients are recorded.
        """
        return self.score(drawn.spans)

    def judge_inputs(self, i, seq):
        """Return a masked judge prompt at the candidate's length, or None for an empty loop."""
        framework = self._judge_framework(i)
        if not seq:
            return None
        return build_masked_judge_prompt(self.judge_tokenizer, framework, len(seq))


def run_eos_generative_eval(strategy_class, method, mode, model, tokenizer, eval_dataset, *, device,
                            judge_dataset=None, judge_model=None, judge_tokenizer=None,
                            num_examples=150, num_samples=20, temperature=1.0, top_k=0, top_p=1.0,
                            max_new_tokens, stratify=False, seed=42, skip_perplexity=False,
                            output_path=None, title="", judge_name=None, anchor_trim=ANCHOR_TRIM,
                            meta=None):
    """Run AR/S2S candidate evaluation and return its scored payload.

    Called by each method's evaluation.run_generative_eval(), including standalone
    and post-training evaluation. strategy_class supplies that method's three hooks;
    method/mode label the payload (ar/causal or s2s/seq2seq).

    Steps:
      1. Set eval mode, validate trim, seed sampling and select framework indices.
      2. Build the strategy -> eval.generative.run_eval_loop() -> scored records.
      3. Combine all-attempt validity with retained-candidate metric summaries.
      4. Log the report, optionally write JSON/figure outputs, return the payload.

    Frameworks with no retained candidates stay in validity counts, without a
    scored record. Native lengths may stratify selection and support scoring;
    the method samplers choose generated length through EOS.
    """
    # ------ 1. Prepare deterministic sampling and select evaluation frameworks. ------
    model.eval()
    validate_anchor_trim(anchor_trim)

    torch.manual_seed(seed)
    lengths = [framework_for_dataset(eval_dataset, i)["target_length"]
               for i in range(len(eval_dataset))]
    selected = select_indices(lengths, num_examples, stratify, random.Random(seed))
    generator = torch.Generator(device=device).manual_seed(seed)

    # ------ 2. Let the method strategy generate candidates for the shared scorer. ------
    strategy = strategy_class(
        model=model, tokenizer=tokenizer, device=device, generator=generator,
        eval_dataset=eval_dataset,
        judge_dataset=judge_dataset if judge_dataset is not None else eval_dataset,
        judge_tokenizer=judge_tokenizer, num_samples=num_samples, temperature=temperature,
        top_k=top_k, top_p=top_p, max_new_tokens=max_new_tokens)

    records = run_eval_loop(
        strategy, selected, model=model, tokenizer=tokenizer, device=device,
        skip_perplexity=skip_perplexity, judge_model=judge_model,
        judge_tokenizer=judge_tokenizer, anchor_trim=anchor_trim)

    # ------ 3. Keep every attempt in validity; summarize scored candidates separately. ------
    attempts = strategy.attempts
    validity = {
        "frameworks_selected": len(selected),
        "frameworks_with_valid_candidates": len(records),
        "attempts": attempts,
        "complete_nonempty": strategy.complete_nonempty,
        "complete_empty": strategy.complete_empty,
        "incomplete": strategy.incomplete,
        "valid_generation_rate": strategy.complete_nonempty / attempts if attempts else None,
        "completion_rate": ((strategy.complete_nonempty + strategy.complete_empty + strategy.invalid) / attempts
                            if attempts else None),
        "length_matches": strategy.length_matches,
        # Unlike summary.length_match_rate/length_absolute_error_mean, these denominators retain
        # incomplete and empty attempts, so a sweep cannot improve by dropping failed decodes.
        "fixed_length_acceptance_rate": strategy.length_matches / attempts if attempts else None,
        "length_absolute_error_mean_capped": (
            strategy.length_absolute_error_capped / attempts if attempts else None),
    }

    payload = dict(meta or {})
    payload.update({
        "schema_version": GENERATION_EVAL_SCHEMA,
        # A BLOSUM number without its matrix and gap penalties is not reproducible.
        "blosum": blosum_provenance(),
        "method": method,
        "judge": judge_name,
        "length_policy": strategy.payload_length_policy(),
        "anchor_trim_each_end": anchor_trim,
        "decoding": {
            "mode": mode, "num_samples": num_samples, "temperature": temperature,
            "top_k": top_k, "top_p": top_p, "max_new_tokens": max_new_tokens, "seed": seed,
        },
        "n_eval_total": len(eval_dataset),
        "n_selected": len(selected),
        "n_generated": len(records),
        "generation_validity": validity,
        "summary": {key: aggregate(records, key)[0] for key, _ in SUMMARY_KEYS},
        "records": records,
    })
    protocol = getattr(tokenizer, "causal_sampling_metadata", None)
    if protocol is not None:
        payload["decoding"]["protocol"] = dict(protocol)
        payload["generation_validity"]["invalid"] = strategy.invalid
        # Include failed frameworks too: otherwise an all-invalid framework
        # disappears from the scored-record population and loses its reasons.
        payload["generation_attempts"] = strategy.generation_attempts
    # ------ 4. Report, optionally persist outputs, and return the payload. ------
    log_report(records, title=title, judge_name=judge_name)
    write_payload(payload, output_path, len(records))
    write_confidence_figure(payload, output_path, title)
    return payload


# --- generate.py: one qualitative record ----------------------------------------------------------
def eos_case_record(model, tokenizer, device, args, generator, case, print_callback, *, mode,
                    build_context, sample_loops, perplexities):
    """Return one generated-case record for AR/S2S generate.py commands.

    Called by each method's generation.generate_case(), which supplies its context
    builder, sampler and self-scorer. mode labels the result as causal or seq2seq.

    Steps:
      1. Build the conditioning input from flanks and optional public context.
      2. Sample attempts; retain completed nonempty loops, preserving duplicates.
      3. Optionally self-score; use target_truth only for reference measurements.
      4. Call print_callback() and return samples, metrics and attempt accounting.

    The native sequence/length never enters the generation framework. Native-
    protocol records also include conditioning, raw draws and separate invalid
    counts. Legacy case records group all noncomplete draws under incomplete.

    The record matches MLM's: a candidate of another length than the native has
    recovery=None, decoding.num_samples counts retained candidates, and metrics gain
    perplexity_mean/min/max over the finite self-scores (MLM names them pseudo_*).
    """
    # ------ 1. Build the prompt/encoder input without native truth or length. ------
    pt1, pt2 = case['pt1'], case['pt2']
    # Native sequence/length never enters this generation framework.
    framework = dict(pt1=pt1, pt2=pt2, target_length=0, max_len=args.max_len,
                     partner_chain=case.get('partner_chain') or args.partner_chain,
                     antigen=case.get('antigen') or args.antigen)
    # Public chain/species conditioning, for a layout that declares it (see SpanLayout.public_fields);
    # every other layout ignores these keys. A case overrides the run-level value.
    for key in ('target_region', 'chain_token', 'species_token'):
        value = case.get(key, getattr(args, key, None))
        if value is not None:
            framework[key] = value
    if 'species_token' not in framework and case.get('species'):
        framework['species_token'] = case['species']
    context, context_meta = build_context(tokenizer, framework)

    # ------ 2. Generate and account for attempts before selecting scored candidates. ------
    attempts = sample_loops(model, tokenizer, context, device=device, generator=generator,
                            num_samples=args.num_samples, temperature=args.temperature,
                            top_k=args.top_k, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    kept, counts = _partition_attempts(attempts)
    seqs = [r['loop'].upper() for r in kept]

    # ------ 3. Score retained loops; reference truth is used only after generation. ------
    native = (case.get('target_truth') or '').strip().upper() or None
    scores = [None] * len(seqs) if args.skip_perplexity else perplexities(
        model, tokenizer, device, [(context, tokenizer.encode(seq)) for seq in seqs])
    scores = normalize_scores(scores, len(seqs), label="case self scorer")
    samples = []
    for seq, score in zip(seqs, scores):
        sample = {'seq': seq, 'length': len(seq)}
        if native:
            sample['length_match'] = len(seq) == len(native)
            sample['recovery'] = amino_acid_recovery(seq, native) if sample['length_match'] else None
        if score is not None:
            sample['perplexity'] = score
        samples.append(sample)
    metrics = calculate(seqs, native)
    scored = [sample['perplexity'] for sample in samples if 'perplexity' in sample]
    if scored:
        metrics['perplexity_mean'] = sum(scored) / len(scored)
        metrics['perplexity_min'] = min(scored)
        metrics['perplexity_max'] = max(scored)
    generated_lengths = [len(seq) for seq in seqs]
    distinct_lengths = sorted(set(generated_lengths))
    gap_display = '/'.join(map(str, distinct_lengths)) or '—'

    # ------ 4. Print the case and return its existing output schema. ------
    print_callback(case.get('name', 'Unnamed Test'), pt1, pt2, gap_display, native, args, samples, metrics)
    record = {'name': case.get('name', 'Unnamed Test'), 'pt1': pt1, 'pt2': pt2,
              'native': native, 'target_length': distinct_lengths[0] if len(distinct_lengths) == 1 else None,
              'generated_lengths': generated_lengths, 'length_policy': {'mode': 'inferred', 'source': 'eos_token'},
              'samples': samples, 'metrics': metrics,
              'decoding': {'mode': mode, 'temperature': args.temperature, 'top_k': args.top_k,
                           'top_p': args.top_p, 'seed': args.seed, 'max_new_tokens': args.max_new_tokens,
                           'num_samples': len(samples)},
              'generation_validity': {'attempts': counts['attempts'],
                                      'complete_nonempty': counts['complete_nonempty'],
                                      'incomplete': counts['incomplete'] + counts['invalid'],
                                      'complete_empty': counts['complete_empty']}}
    if 'conditioning' in context_meta:      # which chain/species tokens a native layout actually used
        record['conditioning'] = context_meta['conditioning']
    protocol = getattr(tokenizer, 'causal_sampling_metadata', None)
    if protocol is not None:
        record['decoding']['protocol'] = dict(protocol)
        record['generation_attempts'] = attempts
        record['generation_validity']['invalid'] = counts['invalid']
        record['generation_validity']['incomplete'] = counts['incomplete']
    return record
