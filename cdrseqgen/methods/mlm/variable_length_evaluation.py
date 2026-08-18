"""Bounded / inferred-length candidate production for held-out evaluation.

Only the generation strategy lives here. Example selection, scoring, judging, reporting, and
writing are shared with the fixed-native track in ``evaluation.py`` — they were duplicated once and
had already drifted apart in four places (progress bar, write guard, summary statistic, empty-mask
guard), which is exactly the kind of difference that makes two tracks quietly incomparable.

The invariant this module exists to protect: the native loop reaches ``calculate`` for scoring and
nothing else. No native length is passed to generation, so the checkpoint's own length head decides
how many residues to fill.
"""

import logging

from cdrseqgen.data.dataset import framework_for_dataset
from cdrseqgen.methods.mlm.evaluation import Draw
from cdrseqgen.methods.mlm.generation import (
    prepare_fixed_length_prompt,
    sample_length_conditioned_spans,
)

logger = logging.getLogger(__name__)


class VariableLengthStrategy:
    """Generate without the native mask count, keeping the native for scoring only."""

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
        return {**self.length_policy.as_dict(), "length_temperature": self.length_temperature}

    def draw(self, i):
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

    def judge_inputs(self, i, seq):
        """Rebuild the judge's prompt at this candidate's own length, not the native one.

        The framework is fixed per example and only the mask count varies, so it is cached while
        the prompt itself is rebuilt for every candidate length.
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
