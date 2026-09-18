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
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from cdrseqgen.datasets.dataset_ar import (
    CDRSEQGEN_CAUSAL,
    LOOP_LENGTH_RESERVE,
    CausalInfillingDataset,
    SpanLayout,
    build_causal_prompt,
    causal_adapter,
    layout_for,
    validate_layout,
    span_layout_metadata,
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

    def test_only_missing_or_none_layouts_select_the_default(self):
        adapter = causal_adapter()
        adapter.span_layout = None
        self.assertIs(layout_for(adapter), CDRSEQGEN_CAUSAL)
        for invalid in (False, 0, "", {}):
            with self.subTest(layout=invalid):
                adapter.span_layout = invalid
                with self.assertRaisesRegex(ValueError, "must be a SpanLayout"):
                    span_layout_metadata(adapter)

    def test_an_adapter_without_a_layout_still_works(self):
        # An adapter built before layouts existed must behave exactly as it did.
        class Bare:
            pass
        self.assertIs(layout_for(Bare()), CDRSEQGEN_CAUSAL)



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
                          build_prompt=lambda t, left, right, options: [t.mask_id] * 7 + left + right + [t.chain_sep_id],
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


class ContractBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.adapter = causal_adapter()
        self.framework = dict(pt1="ACDEFGHIK" * 12, pt2="LMNPQRS" * 12,
                              partner_chain="W" * 120, antigen="Y" * 40,
                              target="MNPQ", target_length=4, max_len=128)

    def test_builder_gets_only_capped_flanks_and_immutable_public_options(self):
        seen = []

        def builder(t, left, right, options):
            self.assertEqual(dict(options), {"max_len": 128})
            with self.assertRaises(TypeError):
                options["target"] = "MNPQ"
            self.assertLessEqual(len(left) + len(right), 128 - LOOP_LENGTH_RESERVE - 3)
            seen.append((left[:], right[:]))
            return CDRSEQGEN_CAUSAL.build_prompt(t, left, right, options)

        self.adapter.span_layout = replace(CDRSEQGEN_CAUSAL, build_prompt=builder)
        expected, _ = build_causal_prompt(self.adapter, self.framework)
        for target in ("", "WW", "A" * 1000):
            actual, _ = build_causal_prompt(self.adapter, dict(self.framework,
                                           target=target, target_length=len(target)))
            self.assertEqual(actual, expected)
        public = {k: v for k, v in self.framework.items() if k not in {"target", "target_length"}}
        self.assertEqual(build_causal_prompt(self.adapter, public)[0], expected)
        self.assertEqual(len(seen), 5)

    def test_old_builder_cannot_read_the_target(self):
        layout = replace(CDRSEQGEN_CAUSAL,
                         build_prompt=lambda t, left, right, options: t.encode(options["target"]))
        with self.assertRaises(KeyError):
            validate_layout(layout, self.adapter)

    def test_layout_cannot_leave_an_empty_prompt_after_capping(self):
        # With only the answer terminator budgeted, capping can remove both flanks.
        # The sampler then has no final prompt position from which to predict a token.
        layout = replace(CDRSEQGEN_CAUSAL, n_layout_tokens=1,
                         build_prompt=lambda t, left, right, options: left + right)
        with self.assertRaisesRegex(ValueError, "at least one prompt token"):
            validate_layout(layout, self.adapter)
        self.adapter.span_layout = layout
        with self.assertRaisesRegex(ValueError, "at least one prompt token"):
            build_causal_prompt(self.adapter,
                                dict(self.framework, max_len=LOOP_LENGTH_RESERVE + 1))

    def test_declared_overhead_must_match_the_builder(self):
        with self.assertRaisesRegex(ValueError, "actual prompt overhead"):
            validate_layout(replace(CDRSEQGEN_CAUSAL, n_layout_tokens=9), self.adapter)

    def test_invalid_metadata_and_terminators_fail_at_setup(self):
        for changes in ({"id": ""}, {"version": 0}, {"version": True},
                        {"n_layout_tokens": -1}, {"n_layout_tokens": 2.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_layout(replace(CDRSEQGEN_CAUSAL, **changes), self.adapter)
        for token in (self.adapter.encode("A")[0], -1, self.adapter.vocab_size, True):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "terminator"):
                validate_layout(replace(CDRSEQGEN_CAUSAL, terminator_id=lambda t: token), self.adapter)

    def test_reordered_answer_is_rejected(self):
        # The old all-alanine probe could not detect reversal.
        layout = replace(CDRSEQGEN_CAUSAL,
                         build_answer=lambda t, target: list(reversed(target)) + [t.eos_id])
        with self.assertRaisesRegex(ValueError, "cannot read back"):
            validate_layout(layout, self.adapter)

    def test_real_prompt_is_checked_even_if_setup_probes_pass(self):
        def builder(t, left, right, options):
            prompt = CDRSEQGEN_CAUSAL.build_prompt(t, left, right, options)
            return prompt + ([t.mask_id] if len(left) + len(right) > 10 else [])
        self.adapter.span_layout = replace(CDRSEQGEN_CAUSAL, build_prompt=builder)
        validate_layout(self.adapter.span_layout, self.adapter)
        with self.assertRaisesRegex(ValueError, "actual prompt overhead"):
            build_causal_prompt(self.adapter, self.framework)

    def test_invalid_token_ids_are_rejected(self):
        def builder(t, left, right, options):
            return left + [t.vocab_size] + right + [t.chain_sep_id]
        with self.assertRaisesRegex(ValueError, "invalid token ids"):
            validate_layout(replace(CDRSEQGEN_CAUSAL, build_prompt=builder), self.adapter)

    def test_dataset_refuses_an_invalid_layout_at_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "corpus.json")
            _corpus(path)
            self.adapter.span_layout = WRAPPED_ANSWER
            with self.assertRaisesRegex(ValueError, "cannot read back"):
                CausalInfillingDataset(path, self.adapter, target_region="h3", max_len=256)

    def test_real_answer_is_checked_even_if_setup_probes_pass(self):
        def answer(t, target):
            if target == t.encode("MNPQ"):
                return list(reversed(target)) + [t.eos_id]
            return target + [t.eos_id]
        self.adapter.span_layout = replace(CDRSEQGEN_CAUSAL, build_answer=answer)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "corpus.json")
            _corpus(path)
            ds = CausalInfillingDataset(path, self.adapter, target_region="h3", max_len=256)
            with self.assertRaisesRegex(ValueError, "cannot read back"):
                ds[0]


class CheckpointLayoutTests(unittest.TestCase):
    def setUp(self):
        from cdrseqgen.methods.ar.backbones import resolve
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.backbone = resolve("scratch")
        self.adapter = self.backbone.build_tokenizer("scratch")

    def test_metadata_preserves_the_native_tokenizer_save(self):
        from cdrseqgen.methods.ar.backbones import persist_layout_with_tokenizer
        adapter = causal_adapter()
        native_save = Mock()
        adapter.save_pretrained = native_save
        persist_layout_with_tokenizer(adapter).save_pretrained(self.root)
        native_save.assert_called_once_with(self.root)
        self.assertEqual(json.loads((self.root / "span_layout.json").read_text()),
                         span_layout_metadata(adapter))

    def test_checkpoint_rejects_each_changed_contract_field(self):
        self.adapter.save_pretrained(self.root)
        metadata = span_layout_metadata(self.adapter)
        for field, value in (("id", "different"), ("version", 2), ("version", True),
                             ("n_layout_tokens", 4), ("terminator_id", self.adapter.bos_id)):
            with self.subTest(field=field):
                (self.root / "span_layout.json").write_text(json.dumps(dict(metadata, **{field: value})))
                with self.assertRaisesRegex(ValueError, "Span layout mismatch"):
                    self.backbone.build_tokenizer("scratch", checkpoint_path=str(self.root))

    def test_conflicting_model_or_run_config_is_not_hidden_by_valid_sidecar(self):
        self.adapter.save_pretrained(self.root)
        for filename in ("config.json", "run_config.json"):
            with self.subTest(filename=filename):
                path = self.root / filename
                path.write_text(json.dumps({"span_layout": dict(span_layout_metadata(self.adapter), version=2)}))
                with self.assertRaisesRegex(ValueError, "Span layout mismatch"):
                    self.backbone.build_tokenizer("scratch", checkpoint_path=str(self.root))
                path.unlink()

    def test_legacy_checkpoint_only_accepts_default_with_warning(self):
        from cdrseqgen.methods.ar.backbones import validate_checkpoint_layout
        with self.assertLogs("cdrseqgen.methods.ar.backbones", level="WARNING") as logs:
            self.backbone.build_tokenizer("scratch", checkpoint_path=str(self.root))
        self.assertIn("Legacy checkpoint", logs.output[0])
        for layout in (FRONT_MARKED, replace(CDRSEQGEN_CAUSAL, version=2)):
            self.adapter.span_layout = layout
            with self.assertRaisesRegex(ValueError, "no span layout metadata"):
                validate_checkpoint_layout(self.root, self.adapter)

    def test_real_cpu_training_saves_reloadable_root_and_intermediate_contracts(self):
        import torch
        from transformers import TrainingArguments
        from cdrseqgen.methods.ar.backbones import record_training_layout
        from cdrseqgen.methods.ar.training import build_causal_trainer
        from cdrseqgen.methods.common.training import run_training_pipeline

        path = self.root / "corpus.json"
        _corpus(path)
        ds = CausalInfillingDataset(str(path), self.adapter, target_region="h3", max_len=128)
        model = self.backbone.load_training_model("scratch", hidden_size=16, num_layers=1,
                                                  num_attention_heads=2, max_len=128)
        (self.root / "run_config.json").write_text(json.dumps({"method": "ar", "seed": 42}))
        record_training_layout(model, self.adapter, self.root)
        args = TrainingArguments(output_dir=str(self.root), use_cpu=True, do_train=True,
                                 max_steps=1, per_device_train_batch_size=1,
                                 save_strategy="steps", save_steps=1, logging_steps=1,
                                 report_to=[], disable_tqdm=True)
        trainer = build_causal_trainer(model, args, ds, ds, self.adapter, callbacks=[])
        run_training_pipeline(trainer, args, tokenizer=self.adapter)
        expected = span_layout_metadata(self.adapter)
        self.assertEqual(json.loads((self.root / "run_config.json").read_text())["span_layout"], expected)
        for directory in (self.root, self.root / "checkpoint-1"):
            with self.subTest(directory=directory):
                self.assertEqual(json.loads((directory / "span_layout.json").read_text()), expected)
                self.assertEqual(json.loads((directory / "config.json").read_text())["span_layout"], expected)
                restored_tok = self.backbone.build_tokenizer("scratch", checkpoint_path=str(directory))
                restored = self.backbone.load_pretrained_model(str(directory))
                self.assertEqual(span_layout_metadata(restored_tok), expected)
                for name, tensor in model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)
        # Loading the model directly must also enforce the contract.
        (self.root / "span_layout.json").write_text(json.dumps(dict(expected, version=2)))
        with self.assertRaisesRegex(ValueError, "Span layout mismatch"):
            self.backbone.load_pretrained_model(str(self.root))


if __name__ == "__main__":
    unittest.main()
