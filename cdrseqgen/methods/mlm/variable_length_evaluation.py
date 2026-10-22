"""Adapt bounded/inferred-length MLM generation to the shared evaluation loop.

Caller:
  mlm.evaluation.run_generative_eval creates VariableLengthStrategy when the
  requested length policy uses the checkpoint's fitted length predictor.

Strategy methods:
  * draw                  -- generate loops from framework context and the length policy
  * self_scores           -- score each loop in its own generator prompt
  * judge_inputs          -- build a judge prompt with the candidate's mask count
  * payload_length_policy -- describe the policy in the output payload

Data flow:
  framework + length head -> selected lengths -> masked prompts -> candidates
  candidates + native    -> eval.generative scoring/reporting helpers

The native loop and its length remain available for scoring and record metadata.
Neither is passed to candidate generation. mlm.evaluation selects the examples
and writes the payload; cdrseqgen.eval owns the shared measurements and scoring loop.
"""

import logging

from cdrseqgen.datasets.dataset_mlm import framework_for_dataset
from cdrseqgen.methods.mlm.evaluation import Draw
from cdrseqgen.methods.mlm.generation import (
    prepare_fixed_length_prompt,
    sample_length_conditioned_spans,
)

logger = logging.getLogger(__name__)


class VariableLengthStrategy:
    """Adapt length-head-based generation to the shared scoring loop.

    The fitted length head selects lengths within the requested policy's support.
    Each candidate retains its own prompt, which may differ in mask count from
    both the native loop and the other candidates for the same framework.
    """

    supplies_native_length = False

    def __init__(self, *, model, tokenizer, device, generator, eval_dataset, judge_dataset,
                 judge_tokenizer, length_predictor, length_policy, length_temperature,
                 decoding, num_samples, temperature, top_k, top_p):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.generator = generator
        self.eval_dataset, self.judge_dataset = eval_dataset, judge_dataset
        self.judge_tokenizer = judge_tokenizer
        self.length_predictor = length_predictor
        self.length_policy = length_policy
        self.length_temperature = length_temperature
        self.decoding, self.num_samples = decoding, num_samples
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self._judge_framework_cache = (None, None)

    def payload_length_policy(self):
        """Return the requested policy and length-sampling temperature for the payload."""
        return {**self.length_policy.as_dict(), "length_temperature": self.length_temperature}

    def draw(self, i):
        """Return generated loops and their prompts, with the native saved for scoring.

        Only framework flanks, optional context and length-policy settings enter
        sample_length_conditioned_spans; the native target stays in the Draw record.
        """
        framework = framework_for_dataset(self.eval_dataset, i)
        native = framework["target"].upper().strip()
        generated, length_metadata = sample_length_conditioned_spans(
            self.model,
            self.tokenizer,
            self.device,
            self.generator,
            pt1=framework["pt1"],
            pt2=framework["pt2"],
            policy=self.length_policy,
            length_predictor=self.length_predictor,
            partner_chain=framework.get("partner_chain"),
            antigen=framework.get("antigen"),
            max_len=framework["max_len"],
            decoding=self.decoding,
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            length_temperature=self.length_temperature,
        )
        seqs = [self.tokenizer.decode(span.token_ids).upper().strip() for span in generated]
        return Draw(
            native=native,
            seqs=seqs,
            spans=[(span.input_ids, span.mask_positions, span.token_ids) for span in generated],
            # The native length labels and buckets the example; generation never received it.
            target_len=len(native),
            length_policy=length_metadata,
            record_extra={"generated_lengths": [len(sequence) for sequence in seqs]},
        )

    def self_scores(self, drawn):
        """Return generator pseudo-perplexities using each candidate's own prompt.

        The scoring calculation is the same as in FixedNativeStrategy; here the
        number of scored positions can differ between candidates.
        """
        from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood
        return [
            pseudo_log_likelihood(self.model, input_ids, mask_positions, token_ids,
                                  self.tokenizer.special_ids, self.tokenizer.mask_id, self.device)[1]
            for input_ids, mask_positions, token_ids in drawn.spans
        ]

    def judge_inputs(self, i, seq):
        """Return a judge prompt containing len(seq) masks for evaluation index i.

        Example: native length=8, candidate length=6 -> six judge masks.

        The framework is cached per index; a prompt is assembled for each call.
        A Subset index is mapped back to the corresponding judge_dataset entry.
        """
        cached_index, cached_framework = self._judge_framework_cache
        if cached_index == i:
            framework = cached_framework
        else:
            full_index = (
                self.eval_dataset.indices[i] if hasattr(self.eval_dataset, "indices") else i
            )
            framework = framework_for_dataset(self.judge_dataset, full_index)
            self._judge_framework_cache = (i, framework)
        prompt = prepare_fixed_length_prompt(
            self.judge_tokenizer,
            framework["pt1"],
            framework["pt2"],
            len(seq),
            partner_chain=framework.get("partner_chain"),
            antigen=framework.get("antigen"),
            max_len=framework["max_len"],
        )
        return prompt.input_ids, prompt.mask_positions
