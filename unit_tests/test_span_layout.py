"""Span layouts: a second arrangement must actually change the example, and nothing else.

The abstraction exists so a pretrained infiller can keep the format it was pretrained on. It is
only worth having if a different layout genuinely produces a different example while the mechanics
around it -- framework split, context injection, capping, answer-only loss -- stay identical. One
layout cannot demonstrate that, so these tests supply a second.
"""
import json
import os
import tempfile
import unittest

from cdrseqgen.data.dataset_ar import (
    CDRSEQGEN_CAUSAL,
    LOOP_LENGTH_RESERVE,
    CausalInfillingDataset,
    SpanLayout,
    build_causal_prompt,
    causal_adapter,
    layout_for,
    validate_layout,
)

# A deliberately different arrangement: the gap marker moves to the FRONT of the prompt instead of
# sitting between the flanks. The answer shape is unchanged, because `validate_layout` requires it
# -- see WRAPPED_ANSWER below for the shape that is refused. Nothing about this is recommended; it
# exists to prove the layout is really in charge of the prompt.
FRONT_MARKED = SpanLayout(
    id="test_front_marked",
    n_layout_tokens=3,
    build_prompt=lambda t, pt1, pt2, fw: [t.mask_id] + pt1 + pt2 + [t.chain_sep_id],
    build_answer=lambda t, target: list(target) + [t.eos_id],
    terminator_id=lambda t: t.eos_id,
)

# Deliberately incompatible: wraps the answer, so the sampler would stop at the leading terminator
# and return an empty loop counted as a completed draw. `validate_layout` must refuse it.
WRAPPED_ANSWER = SpanLayout(
    id="test_wrapped_answer",
    n_layout_tokens=3,
    build_prompt=CDRSEQGEN_CAUSAL.build_prompt,
    build_answer=lambda t, target: [t.eos_id] + list(target) + [t.eos_id],
    terminator_id=lambda t: t.eos_id,
)


def _corpus(path):
    entry = {
        "sequence_id": "x", "species": "human", "isotype": "IGHM",
        "chain_sequence_ids": {"heavy": "h", "light": "l"},
        "source": {"dataset": "OAS", "run": 1, "subject": "D"},
        "split": "train",
    }
    for chain in ("heavy", "light"):
        entry[chain] = {"full_aa": "ACDEFGHIKLMNPQ", "imgt_numbering": {},
                        "fwr1": "AC", "cdr1": "DE", "fwr2": "FG", "cdr2": "HI",
                        "fwr3": "KL", "cdr3": "MNPQ", "fwr4": "RS"}
    with open(path, "w") as handle:
        json.dump([entry], handle)


class LayoutSelectionTests(unittest.TestCase):
    def test_the_default_is_cdrseqgens_own(self):
        self.assertIs(layout_for(causal_adapter()), CDRSEQGEN_CAUSAL)

    def test_an_adapter_without_a_layout_still_works(self):
        # An adapter built before layouts existed must behave exactly as it did.
        class Bare:
            pass
        self.assertIs(layout_for(Bare()), CDRSEQGEN_CAUSAL)

    def test_the_layout_lives_on_the_adapter_not_the_call(self):
        # build_causal_prompt takes no layout argument: training and generation resolve it from the
        # same adapter, which is what stops them diverging (2026-09-02).
        import inspect
        params = set(inspect.signature(build_causal_prompt).parameters)
        self.assertEqual(params, {"tokenizer", "framework"})


class LayoutValidationTests(unittest.TestCase):
    """The sampler reads generated tokens as residues-until-terminator. A layout that builds any
    other answer shape trains the model to emit it, and generation then returns an EMPTY loop that
    counts as a completed draw -- a silent zero, not a crash."""

    def test_the_shipped_layout_is_accepted(self):
        self.assertIs(validate_layout(CDRSEQGEN_CAUSAL, causal_adapter()), CDRSEQGEN_CAUSAL)

    def test_a_wrapped_answer_is_refused_with_the_reason(self):
        with self.assertRaises(ValueError) as ctx:
            validate_layout(WRAPPED_ANSWER, causal_adapter())
        self.assertIn("cannot read back", str(ctx.exception))
        self.assertIn("extraction step", str(ctx.exception))

    def test_the_sampler_refuses_it_too_not_just_the_adapter(self):
        # The guard has to sit where generation happens, not only where the adapter is built --
        # a layout can be attached to an adapter any caller constructs.
        from cdrseqgen.methods.ar.sampling import sample_loops
        adapter = causal_adapter()
        adapter.span_layout = WRAPPED_ANSWER
        with self.assertRaises(ValueError):
            sample_loops(None, adapter, [1, 2, 3], device="cpu", num_samples=1)


class SecondLayoutTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "c.json")
        _corpus(self.path)
        self.addCleanup(self.dir.cleanup)

    def _dataset(self, layout):
        adapter = causal_adapter()
        adapter.span_layout = layout
        return CausalInfillingDataset(self.path, adapter, target_region="h3", max_len=256), adapter

    def test_a_second_layout_produces_a_different_prompt(self):
        default, _ = self._dataset(CDRSEQGEN_CAUSAL)
        other, _ = self._dataset(FRONT_MARKED)
        self.assertNotEqual(default[0]["input_ids"].tolist(), other[0]["input_ids"].tolist())

    def test_the_mechanics_around_it_are_unchanged(self):
        # Same target, same prompt/answer split discipline -- only the arrangement moved.
        default, _ = self._dataset(CDRSEQGEN_CAUSAL)
        other, _ = self._dataset(FRONT_MARKED)
        self.assertEqual(default[0]["target_len"], other[0]["target_len"])
        for item in (default[0], other[0]):
            labels = item["labels"].tolist()
            self.assertEqual(labels[:item["prompt_len"]], [-100] * item["prompt_len"])
            self.assertTrue(all(x != -100 for x in labels[item["prompt_len"]:]))

    def test_generation_and_training_agree_under_a_second_layout_too(self):
        # The invariant the whole design exists to protect, checked on the layout that is NOT the
        # default -- a bug that only holds for CDRSEQGEN_CAUSAL would be invisible otherwise.
        for layout in (CDRSEQGEN_CAUSAL, FRONT_MARKED):
            with self.subTest(layout=layout.id):
                ds, adapter = self._dataset(layout)
                item = ds[0]
                trained = item["input_ids"].tolist()[:item["prompt_len"]]
                generated, _ = build_causal_prompt(adapter, ds.framework_parts(0))
                self.assertEqual(trained, generated)

    def test_the_layout_owns_the_length_budget(self):
        wide = SpanLayout(id="wide", n_layout_tokens=9,
                          build_prompt=CDRSEQGEN_CAUSAL.build_prompt,
                          build_answer=CDRSEQGEN_CAUSAL.build_answer,
                          terminator_id=CDRSEQGEN_CAUSAL.terminator_id)
        adapter = causal_adapter()
        adapter.span_layout = wide
        framework = dict(pt1="ACDEFGHIK", target="MNPQ", pt2="LMNPQRS", partner_chain=None,
                         antigen=None, max_len=LOOP_LENGTH_RESERVE + 9, target_length=4)
        # At exactly reserve + n_layout_tokens it fits; one less and it must refuse.
        build_causal_prompt(adapter, framework)
        framework["max_len"] = LOOP_LENGTH_RESERVE + 8
        with self.assertRaises(ValueError):
            build_causal_prompt(adapter, framework)


if __name__ == "__main__":
    unittest.main()
