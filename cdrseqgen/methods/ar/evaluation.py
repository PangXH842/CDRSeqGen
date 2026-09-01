"""Causal generative evaluation: sample loops from held-out frameworks and score them.

  * CausalStrategy            -- the four things the shared loop cannot do for a causal model
  * run_causal_generative_eval -- select examples, run the loop, write generation_eval.json

The loop itself lives in :mod:`cdrseqgen.eval.generative` and is shared with the masked method, so
an AR-versus-MLM table is produced by one implementation. This module supplies only the strategy:

  * ``draw(i)``            -- decode to <eos> instead of expanding a mask span
  * ``self_scores(drawn)`` -- a TRUE perplexity, not a pseudo one. A masked LM must re-score each
    position with the others visible; a causal model factorises properly, so this is just the NLL of
    the answer under one forward pass.
  * ``judge_inputs(i, seq)`` -- rebuilt per candidate, because causal lengths vary by construction.
    The judge stays a masked LM, so the prompt is masked — that is the judge's format, not ours.
  * ``payload_length_policy()`` -- reports that length came from <eos>, which is the whole point:
    Findings §10's gap-sentinel head has no counterpart here.

**These numbers are free-running, not teacher-forced.** ``training.causal_compute_metrics`` reports
argmax over the true prefix and is for checkpoint selection; this samples. The README's "Teacher-
forced accuracy is not generation accuracy" gap applies to the causal side too.
"""
import logging
import math

import torch

from cdrseqgen.data.dataset_ar import build_causal_prompt
from cdrseqgen.data.dataset_common import framework_for_dataset
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.eval.generative import (Draw, build_masked_judge_prompt, log_report, run_eval_loop,
                                       select_indices, write_payload)
from cdrseqgen.methods.ar.sampling import sample_loops

logger = logging.getLogger(__name__)


class CausalStrategy:
    """Decode-to-<eos> candidates for the shared evaluation loop."""

    supplies_native_length = False

    def __init__(self, *, model, tokenizer, device, generator, eval_dataset, judge_dataset,
                 judge_tokenizer, num_samples, temperature, top_k, top_p, max_new_tokens=40):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.generator = generator
        self.eval_dataset, self.judge_dataset = eval_dataset, judge_dataset
        self.judge_tokenizer = judge_tokenizer
        self.num_samples, self.temperature = num_samples, temperature
        self.top_k, self.top_p = top_k, top_p
        self.max_new_tokens = max_new_tokens
        self._framework_cache = (None, None)

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
        prompt, _ = build_causal_prompt(self.tokenizer, framework, reserve=self.max_new_tokens)
        results = sample_loops(
            self.model, self.tokenizer, prompt, device=self.device,
            num_samples=self.num_samples, temperature=self.temperature,
            top_k=self.top_k, top_p=self.top_p, max_new_tokens=self.max_new_tokens,
            generator=self.generator)

        # An incomplete draw has no length the model chose, so it is dropped rather than scored as
        # a short loop — the same rule the IgLM adapter applies to its own attempt budget.
        kept = [r for r in results if r["complete"] and r["loop"]]
        if not kept:
            return None
        seqs = [r["loop"].upper() for r in kept]
        native = framework["target"].upper().strip()
        return Draw(
            native=native,
            seqs=seqs,
            spans=[(prompt, self.tokenizer.encode(s)) for s in seqs],
            target_len=len(native),
            length_policy=self.payload_length_policy(),
            record_extra={"attempts": len(results), "incomplete": len(results) - len(kept)},
        )

    @torch.no_grad()
    def self_scores(self, drawn):
        """True perplexity of each candidate under the model that produced it.

        exp of the mean negative log-likelihood over the answer tokens (loop + <eos>), from a single
        forward pass. The masked side needs one pass per position for its pseudo-perplexity; a
        causal factorisation does not.
        """
        scores = []
        for prompt, answer_ids in drawn.spans:
            ids = list(prompt) + list(answer_ids) + [self.tokenizer.eos_id]
            labels = [-100] * len(prompt) + list(answer_ids) + [self.tokenizer.eos_id]
            tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(input_ids=tensor)["logits"].float()
            gold = torch.tensor([labels], dtype=torch.long, device=self.device)
            nll = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), gold[:, 1:].reshape(-1),
                ignore_index=-100, reduction="mean")
            scores.append(float(math.exp(nll.item())) if torch.isfinite(nll) else None)
        return scores

    def judge_inputs(self, i, seq):
        """Rebuild the judge's masked prompt at THIS candidate's length; causal lengths vary."""
        framework = self._judge_framework(i)
        if not seq:
            return None
        return build_masked_judge_prompt(self.judge_tokenizer, framework, len(seq))


def run_causal_generative_eval(model, tokenizer, eval_dataset, *, device, judge_dataset=None,
                               judge_model=None, judge_tokenizer=None, num_examples=150,
                               num_samples=20, temperature=1.0, top_k=0, top_p=1.0,
                               max_new_tokens=40, stratify=False, seed=42,
                               skip_perplexity=False, output_path=None, title="",
                               anchor_trim=ANCHOR_TRIM):
    """Sample from held-out frameworks and write generation_eval.json, causal side."""
    import random

    lengths = [framework_for_dataset(eval_dataset, i)["target_length"]
               for i in range(len(eval_dataset))]
    selected = select_indices(lengths, num_examples, stratify, random.Random(seed))
    generator = torch.Generator(device="cpu").manual_seed(seed)

    strategy = CausalStrategy(
        model=model, tokenizer=tokenizer, device=device, generator=generator,
        eval_dataset=eval_dataset, judge_dataset=judge_dataset or eval_dataset,
        judge_tokenizer=judge_tokenizer, num_samples=num_samples, temperature=temperature,
        top_k=top_k, top_p=top_p, max_new_tokens=max_new_tokens)

    records = run_eval_loop(
        strategy, selected, model=model, tokenizer=tokenizer, device=device,
        skip_perplexity=skip_perplexity, judge_model=judge_model,
        judge_tokenizer=judge_tokenizer, anchor_trim=anchor_trim)

    payload = {"method": "causal", "length_policy": strategy.payload_length_policy(),
               "records": records}
    log_report(records, title=title)
    if output_path:
        write_payload(payload, output_path, len(records))
    return payload
