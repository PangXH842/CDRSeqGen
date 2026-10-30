"""What the two decode-to-<eos> methods, AR and S2S, share.

Both train a plain Trainer on a model that returns its own loss, end each loop with ``<eos>`` instead
of choosing a length, and are evaluated by the same free-running sampler. They differ only in the
conditioning input (a causal prompt against encoder ids), the sampler, the scorer and how predictions
line up with labels, so each method supplies those and nothing else.

  * eos_recovery_metrics    -- pooled whole/core/anchor recovery from ALIGNED predictions and labels
  * get_infilling_datasets  -- tokenizer and train/eval subsets of a method's infilling dataset
  * EosDecodingStrategy     -- candidates, validity accounting and judge prompts for the shared loop
  * run_eos_generative_eval -- select examples, run the loop, write generation_eval.json
  * eos_case_record         -- one scored record for generate.py

The masked method shares none of it: it hides positions and chooses a length (see ``mlm/``).
"""
import logging
import random

import numpy as np
import torch

from cdrseqgen.datasets.dataset_common import (
    evaluation_context_view, framework_for_dataset, select_split_indices,
)
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.eval.generative import (
    Draw, SUMMARY_KEYS, aggregate, build_masked_judge_prompt, log_report, run_eval_loop,
    select_indices, write_confidence_figure, write_payload,
)
from cdrseqgen.eval.metrics import (
    amino_acid_recovery, blosum_provenance, calculate, core_anchor_indices,
)
from cdrseqgen.methods.common.arguments import validate_anchor_trim

logger = logging.getLogger(__name__)


# --- training: metrics and datasets ---------------------------------------------------------------
def eos_recovery_metrics(preds, gold, *, trim):
    """Teacher-forced recovery over each loop: whole-loop, plus anchor-excluded core and anchor.

    ``preds`` and ``gold`` are [N, S] arrays ALIGNED position by position: ``preds[r, t]`` is the
    model's guess for ``gold[r, t]``. Each method does that alignment itself, because it is the one
    thing they do differently (the causal logits arrive one position early; the encoder-decoder pair
    is already offset). Getting it wrong offsets every position by one and quietly reports a
    different quantity. ``gold`` is -100 outside the supervised stream.

    The last supervised position of each row is its <eos>; it is excluded so the number measures
    residue prediction, comparable to the masked side, rather than being diluted by a token the
    masked method has no equivalent of.
    """
    whole_hits = whole_total = core_hits = core_total = anchor_hits = anchor_total = 0
    rows_with_core = 0
    for row_preds, row_gold in zip(preds, gold):
        positions = np.flatnonzero(row_gold != -100)
        if positions.size == 0:
            continue
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

    rows = int(sum(1 for row in gold if (row != -100).any()))
    return {
        "accuracy": whole_hits / whole_total if whole_total else 0.0,
        "core_accuracy": core_hits / core_total if core_total else 0.0,
        "anchor_accuracy": anchor_hits / anchor_total if anchor_total else 0.0,
        "core_coverage": rows_with_core / rows if rows else 0.0,
    }


def get_infilling_datasets(dataset_class, kind, model_args, data_args, training_args, backbone, *,
                           checkpoint_path=None):
    """Tokenizer and train/eval subsets of ``dataset_class``; ``kind`` only labels the log line.

    Training and evaluation share the parsed records, and the evaluation view disables training
    context dropout. The split guard (``select_split_indices``) is the one every method uses, so
    checkpoints cannot be selected on the test partition by accident.
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
    """Decode-to-<eos> candidates for the shared evaluation loop.

    A method subclasses this and defines three hooks:

      * ``context(framework)``  -- the conditioning input its sampler takes (prompt or encoder ids);
                                   it must match the one training built for this framework
      * ``sample(context)``     -- call the method's ``sample_loops``; ``sampling_options()`` has
                                   the arguments every method passes
      * ``score(spans)``        -- the method's ``perplexities`` over (context, loop ids) pairs
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

        self.length_matches = 0
        self.length_absolute_error_capped = 0

    # --- the three hooks a method supplies --------------------------------------------------
    def context(self, framework):
        raise NotImplementedError("subclasses build the sampler's conditioning input")

    def sample(self, context):
        raise NotImplementedError("subclasses call their sample_loops")

    def score(self, spans):
        raise NotImplementedError("subclasses call their perplexities")

    def sampling_options(self):
        return dict(device=self.device, num_samples=self.num_samples, temperature=self.temperature,
                    top_k=self.top_k, top_p=self.top_p, max_new_tokens=self.max_new_tokens,
                    generator=self.generator)

    # --- framework lookup -------------------------------------------------------------------
    # `framework_for_dataset` already unwraps a Subset, so the eval side passes the SUBSET index
    # and lets it resolve. The judge dataset is the full one, so that side resolves the index first.
    # Doing both is the bug this comment exists to prevent: it indexes the mapping twice.
    def framework(self, i):
        cached_index, cached = self._framework_cache
        if cached_index != i:
            cached = framework_for_dataset(self.eval_dataset, i)
            self._framework_cache = (i, cached)
        return cached

    def _judge_framework(self, i):
        if self.judge_dataset is self.eval_dataset:
            return self.framework(i)
        full_index = self.eval_dataset.indices[i] if hasattr(self.eval_dataset, "indices") else i
        return framework_for_dataset(self.judge_dataset, full_index)

    # --- the strategy protocol --------------------------------------------------------------
    def payload_length_policy(self):
        return {"mode": "inferred", "source": "eos_token"}

    def draw(self, i):
        framework = self.framework(i)
        context = self.context(framework)
        native = framework["target"].upper().strip()
        results = self.sample(context)

        # Keep quality metrics conditional on a biologically non-empty completed loop, but retain
        # every failed attempt in the run-level validity accounting. Otherwise frameworks where the
        # model never emits EOS disappear and make the remaining sample look artificially healthy.
        complete = [r for r in results if r["complete"]]
        kept = [r for r in complete if r["loop"]]
        incomplete = len(results) - len(complete)
        empty = len(complete) - len(kept)
        self.attempts += len(results)
        length_matches = sum(
            r["complete"] and bool(r["loop"]) and len(r["loop"]) == len(native)
            for r in results)
        # An unterminated draw has no chosen length. Its cutoff length is still useful for sweep
        # ranking as long as it is penalized by at least one residue rather than looking exact when
        # the cutoff happens to equal the native length.
        capped_errors = [max(1, abs(len(r["loop"]) - len(native))) if not r["complete"]
                         else abs(len(r["loop"]) - len(native)) for r in results]
        self.complete_nonempty += len(kept)
        self.complete_empty += empty
        self.incomplete += incomplete
        self.length_matches += length_matches
        self.length_absolute_error_capped += sum(capped_errors)
        if not kept:
            return None
        seqs = [r["loop"].upper() for r in kept]
        return Draw(
            native=native,
            seqs=seqs,
            spans=[(context, self.tokenizer.encode(s)) for s in seqs],
            target_len=len(native),
            length_policy=self.payload_length_policy(),
            record_extra={
                "attempts": len(results), "incomplete": incomplete, "empty": empty,
                "length_matches": length_matches,
                "length_absolute_error_sum_capped": sum(capped_errors),
            },
        )

    @torch.no_grad()
    def self_scores(self, drawn):
        """True perplexity of each candidate under the model that produced it.

        exp of the mean negative log-likelihood over the answer tokens (loop + <eos>), from a single
        forward pass. The masked side needs one pass per position for its pseudo-perplexity; a
        properly factorised decoder does not.
        """
        return self.score(drawn.spans)

    def judge_inputs(self, i, seq):
        """Rebuild the judge's masked prompt at THIS candidate's length; generated lengths vary."""
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
    """Sample from held-out frameworks and write generation_eval.json for an <eos> method.

    ``method`` and ``mode`` are the labels the payload records ("ar"/"causal", "s2s"/"seq2seq").
    """
    model.eval()
    validate_anchor_trim(anchor_trim)

    torch.manual_seed(seed)
    lengths = [framework_for_dataset(eval_dataset, i)["target_length"]
               for i in range(len(eval_dataset))]
    selected = select_indices(lengths, num_examples, stratify, random.Random(seed))
    generator = torch.Generator(device=device).manual_seed(seed)

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

    attempts = strategy.attempts
    validity = {
        "frameworks_selected": len(selected),
        "frameworks_with_valid_candidates": len(records),
        "attempts": attempts,
        "complete_nonempty": strategy.complete_nonempty,
        "complete_empty": strategy.complete_empty,
        "incomplete": strategy.incomplete,
        "valid_generation_rate": strategy.complete_nonempty / attempts if attempts else None,
        "completion_rate": ((strategy.complete_nonempty + strategy.complete_empty) / attempts
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
        "schema_version": "cdrseqgen.generation_eval.v1",
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
    log_report(records, title=title, judge_name=judge_name)
    write_payload(payload, output_path, len(records))
    write_confidence_figure(payload, output_path, title)
    return payload


# --- generate.py: one qualitative record ----------------------------------------------------------
def eos_case_record(model, tokenizer, device, args, generator, case, print_callback, *, mode,
                    build_context, sample_loops, perplexities):
    """Sample, score and record one supplied framework, using the same context builder, sampler and
    self-scorer as evaluation. ``mode`` labels the decoding in the record ("causal", "seq2seq")."""
    pt1, pt2 = case['pt1'], case['pt2']
    # Native sequence/length never enters this generation framework.
    framework = dict(pt1=pt1, pt2=pt2, target_length=0, max_len=args.max_len,
                     partner_chain=case.get('partner_chain') or args.partner_chain,
                     antigen=case.get('antigen') or args.antigen)
    context, _ = build_context(tokenizer, framework)
    attempts = sample_loops(model, tokenizer, context, device=device, generator=generator,
                            num_samples=args.num_samples, temperature=args.temperature,
                            top_k=args.top_k, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    valid = [r['loop'] for r in attempts if r['complete'] and r['loop']]
    native = (case.get('target_truth') or '').strip().upper() or None
    scores = [None] * len(valid) if args.skip_perplexity else perplexities(
        model, tokenizer, device, [(context, tokenizer.encode(seq)) for seq in valid])
    samples = []
    for seq, score in zip(valid, scores):
        record = {'seq': seq, 'length': len(seq)}
        if native:
            record['length_match'] = len(seq) == len(native)
            record['recovery'] = amino_acid_recovery(seq, native) if record['length_match'] else None
        if score is not None:
            record['perplexity'] = score
        samples.append(record)
    metrics = calculate(valid, native)
    lengths = [len(s) for s in valid]
    print_callback(case.get('name', 'Unnamed Test'), pt1, pt2,
                   '/'.join(map(str, sorted(set(lengths)))) or '—', native, args, samples, metrics)
    return {'name': case.get('name', 'Unnamed Test'), 'pt1': pt1, 'pt2': pt2,
            'native': native, 'target_length': lengths[0] if len(set(lengths)) == 1 else None,
            'generated_lengths': lengths, 'length_policy': {'mode': 'inferred', 'source': 'eos_token'},
            'samples': samples, 'metrics': metrics,
            'decoding': {'mode': mode, 'temperature': args.temperature, 'top_k': args.top_k,
                         'top_p': args.top_p, 'seed': args.seed, 'max_new_tokens': args.max_new_tokens,
                         'num_samples': len(samples)},
            'generation_validity': {'attempts': len(attempts), 'complete_nonempty': len(valid),
                                    'incomplete': sum(not r['complete'] for r in attempts),
                                    'complete_empty': sum(r['complete'] and not r['loop'] for r in attempts)}}
