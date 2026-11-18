"""Tests merged from: test_dataset_ar.py, test_dataset_common.py, test_dataset_s2s.py, test_span_layout.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import json
import os
import tempfile
import unittest
import torch
from cdrseqgen.datasets.dataset_common import AntibodyFrameworkDataset, AntibodyTokenizer
from cdrseqgen.datasets.dataset_ar import (
    LAYOUT_SPECIAL_TOKENS,
    CAUSAL_DECODE_STEPS,
    LOOP_LENGTH_RESERVE,
    CausalInfillingDataset,
    CausalTokenizer,
    build_causal_prompt,
    causal_adapter,
    causal_collate_fn,
)
from cdrseqgen.datasets.dataset_mlm import MaskedInfillingDataset
from cdrseqgen.datasets.dataset_common import (
    AntibodyFrameworkDataset,
    AntibodyTokenizer,
    TokenizerAdapter,
    framework_for_dataset,
    inject_context,
    select_split_indices,
)
from cdrseqgen.datasets.dataset_common import AntibodyTokenizer
from cdrseqgen.datasets.dataset_ar import CausalTokenizer, build_causal_prompt, causal_adapter
from cdrseqgen.datasets.dataset_s2s import (
    ENCODER_LAYOUT_TOKENS,
    MAX_LOOP_RESIDUES,
    S2S_DECODE_STEPS,
    Seq2SeqInfillingDataset,
    Seq2SeqTokenizer,
    build_decoder_pair,
    build_encoder_input,
    s2s_adapter,
    s2s_collate_fn,
    validate_s2s_decode_steps,
)
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


# ====================================================================================================
# from test_dataset_ar.py
# Causal span-infilling data: the layout properties an AR method depends on.
#
# The science-critical one is the reordering. A causal model reads left to right, so writing the
# sequence naturally — pt1, loop, pt2 — leaves the right flank unread at the moment the loop is
# generated. FR4 carries the J-gene motif that terminates CDR-H3, and Findings §11 measured the cost
# of losing it: 2% of loops ended in that motif when IgLM was forced past its stopping point against
# 20% when it chose freely. So `pt2 must appear in the prompt` is not a style preference, and it gets
# a test.
# ====================================================================================================

CHAIN = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
         "fwr3": "FFFF", "cdr3": "GGGGG", "fwr4": "HHHH"}
RECORD = {"heavy": dict(CHAIN, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
          "light": dict(CHAIN, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
          "antigen": {"full_aa": "WWWW"}, "split": "train"}

def _data(n=4):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump([RECORD] * n, handle)
    handle.close()
    return handle.name


class VocabTests(unittest.TestCase):
    def test_the_shared_vocab_is_a_prefix_of_the_causal_one(self):
        # Renumbering would silently mismatch every existing 25-row MLM embedding table.
        shared, causal = AntibodyTokenizer(), CausalTokenizer()
        self.assertEqual(causal.vocab[:len(shared.vocab)], shared.vocab)
        self.assertEqual(len(shared.vocab), 25)
        self.assertEqual((causal.bos_id, causal.eos_id), (25, 26))

    def test_the_adapter_exposes_an_end_token(self):
        # A masked LM has no EOS, which is why §10 needed a length classifier at all.
        self.assertIsNotNone(causal_adapter().eos_id)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.path = _data()
        self.tok = CausalTokenizer()
        self.ds = CausalInfillingDataset(self.path, causal_adapter(self.tok), target_region="h3")

    def tearDown(self):
        os.unlink(self.path)

    def test_the_right_flank_is_inside_the_prompt(self):
        item = self.ds[0]
        ids, prompt_len = item["input_ids"].tolist(), item["prompt_len"]
        adapter = self.ds.tokenizer
        between = ids[ids.index(adapter.mask_id) + 1: prompt_len - 1]
        self.assertEqual(self.tok.decode(between), "HHHH")     # pt2, before generation starts

    def test_the_answer_is_the_native_loop_then_eos(self):
        item = self.ds[0]
        ids, prompt_len = item["input_ids"].tolist(), item["prompt_len"]
        self.assertEqual(self.tok.decode(ids[prompt_len:-1]), "GGGGG")
        self.assertEqual(ids[-1], self.tok.eos_id)

    def test_the_prompt_is_never_supervised(self):
        # Scoring given framework residues would inflate any token-level number.
        item = self.ds[0]
        labels = item["labels"].tolist()
        self.assertEqual(set(labels[:item["prompt_len"]]), {-100})

    def test_eos_is_supervised_so_length_is_learned(self):
        # This is what replaces the gap-sentinel length head of Findings §10.
        self.assertEqual(self.ds[0]["labels"].tolist()[-1], self.tok.eos_id)

    def test_supervised_positions_are_exactly_the_loop_plus_eos(self):
        item = self.ds[0]
        supervised = (item["labels"] != -100).sum().item()
        self.assertEqual(supervised, item["target_len"] + 1)


class CapTests(unittest.TestCase):
    def setUp(self):
        self.path = _data()
        self.adapter = causal_adapter()

    def tearDown(self):
        os.unlink(self.path)

    def test_capping_truncates_context_not_the_answer(self):
        # max_len must clear LOOP_LENGTH_RESERVE + the three layout tokens; below that the reserve
        # itself cannot fit and the example is refused rather than silently truncated.
        # Derived from the constant, not hardcoded: these used to be (128, 64, 44), sized against a
        # reserve of 40, and broke the moment the reserve grew to fit a 49-residue H3 in OAS4.
        minimum = LOOP_LENGTH_RESERVE + LAYOUT_SPECIAL_TOKENS
        for max_len in (minimum + 84, minimum + 20, minimum):
            item = CausalInfillingDataset(self.path, self.adapter, target_region="h3",
                                          max_len=max_len, include_partner_chain=True)[0]
            self.assertLessEqual(len(item["input_ids"]), max_len)
            self.assertEqual(item["target_len"], 5)            # loop survives intact

    def test_a_max_len_too_small_for_the_reserve_is_refused(self):
        # Honouring it would mean cutting the training target; fail where the cause is named.
        ds = CausalInfillingDataset(self.path, self.adapter, target_region="h3", max_len=7)
        with self.assertRaises(ValueError) as ctx:
            ds[0]
        self.assertIn("max_len", str(ctx.exception))

    def test_the_generation_prompt_is_byte_identical_to_the_training_one(self):
        """The bug this fix exists for.

        The reserve used to be the native loop length at training and `max_new_tokens` at
        generation, so `inject_context` trimmed the flanks differently in the two paths. With
        max_len=128 that produced an 87-token prompt at sampling against 110 at training: the model
        met a framework 23 tokens shorter than anything it had trained on, and stopped emitting
        <eos> entirely (P(eos) ~ 1e-4 free-running against 0.91 teacher-forced). The prompt must be
        a function of (framework, max_len) alone, because generation cannot know the length it is
        trying to predict.
        """
        minimum = LOOP_LENGTH_RESERVE + LAYOUT_SPECIAL_TOKENS
        for max_len in (minimum, minimum + 64, minimum + 192):
            ds = CausalInfillingDataset(self.path, self.adapter, target_region="h3",
                                        max_len=max_len, include_partner_chain=True)
            item = ds[0]
            trained_prompt = item["input_ids"].tolist()[:item["prompt_len"]]
            generated_prompt, _ = build_causal_prompt(self.adapter, ds.framework_parts(0))
            self.assertEqual(trained_prompt, generated_prompt, f"diverged at max_len={max_len}")

    def test_a_loop_longer_than_the_reserve_is_refused(self):
        # Otherwise the assembled example could overflow max_len without anyone noticing.
        record = dict(RECORD, heavy=dict(RECORD["heavy"], cdr3="G" * (LOOP_LENGTH_RESERVE + 1)))
        with open(self.path, "w") as handle:
            json.dump([record], handle)
        ds = CausalInfillingDataset(self.path, self.adapter, target_region="h3", max_len=256)
        with self.assertRaisesRegex(ValueError, "exceeds.*reserve"):
            ds[0]

    def test_the_fixed_reserve_cannot_be_overridden(self):
        # Native length or decode settings must never affect flank truncation again.
        ds = AntibodyFrameworkDataset(self.path, self.adapter, target_region="h3", max_len=64)
        with self.assertRaises(TypeError):
            build_causal_prompt(self.adapter, ds.framework_parts(0), reserve=5)

    def test_the_decode_budget_includes_an_eos_step(self):
        self.assertEqual(CAUSAL_DECODE_STEPS, LOOP_LENGTH_RESERVE + 1)

    def test_the_reserve_covers_every_corpus_that_records_its_longest_loop(self):
        """A longer loop raises at data-load time and fails the whole run.

        The reserve was 40, which was *exactly* the longest H3 in the single-donor corpus -- no
        headroom -- and the 4-donor OAS4 corpus then arrived with one 49-residue H3. Read the
        preprocessing sidecar rather than the corpus: the sidecar is a few hundred bytes, so this
        stays in the fast suite, which promises to need no processed dataset.
        """
        import glob
        sidecars = sorted(glob.glob("data/processed/*.metadata.json"))
        checked = 0
        for path in sidecars:
            with open(path) as handle:
                longest = json.load(handle).get("longest_loop")
            if not longest:
                continue        # written by an older preprocessing run
            checked += 1
            with self.subTest(corpus=os.path.basename(path)):
                worst = max(longest.values())
                self.assertLessEqual(
                    worst, LOOP_LENGTH_RESERVE,
                    f"{os.path.basename(path)} holds a {worst}-residue loop; raise "
                    f"LOOP_LENGTH_RESERVE (currently {LOOP_LENGTH_RESERVE}) before training on it")
        if not checked:
            self.skipTest("no corpus sidecar records longest_loop yet")


class SharedWithTheMaskedMethodTests(unittest.TestCase):
    def test_the_causal_dataset_reuses_the_neutral_base(self):
        self.assertTrue(issubclass(CausalInfillingDataset, AntibodyFrameworkDataset))

    def test_context_dropout_is_inherited_not_reimplemented(self):
        path = _data(20)
        kw = dict(target_region="h3", include_partner_chain=True, context_dropout=0.5,
                  context_dropout_seed=5)
        causal = CausalInfillingDataset(path, causal_adapter(), **kw)
        neutral = AntibodyFrameworkDataset(path, causal_adapter(), **kw)
        kept_c = [causal.framework_parts(i)["partner_chain"] is not None for i in range(20)]
        kept_n = [neutral.framework_parts(i)["partner_chain"] is not None for i in range(20)]
        self.assertEqual(kept_c, kept_n)
        self.assertIn(False, kept_c)
        os.unlink(path)


class CollateTests(unittest.TestCase):
    def test_padding_is_masked_and_labels_stay_ignored(self):
        path = _data()
        adapter = causal_adapter()
        ds = CausalInfillingDataset(path, adapter, target_region="h3")
        items = [ds[0], ds[1]]
        items[1] = dict(items[1])
        items[1]["input_ids"] = items[1]["input_ids"][:-3]      # force a ragged batch
        items[1]["labels"] = items[1]["labels"][:-3]
        batch = causal_collate_fn(items, pad_token_id=adapter.pad_id)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertEqual(batch["attention_mask"][1].sum().item(), len(items[1]["input_ids"]))
        self.assertTrue(torch.all(batch["labels"][1, len(items[1]["labels"]):] == -100))
        self.assertEqual(set(batch), {"input_ids", "labels", "attention_mask"})

        # Generic AutoModelForCausalLM implementations need not accept arbitrary metadata kwargs.
        class StrictModel:
            def __call__(self, input_ids, labels, attention_mask):
                return input_ids, labels, attention_mask

        StrictModel()(**batch)
        os.unlink(path)


# ====================================================================================================
# from test_dataset_common.py
# The method-neutral data seam: what a non-masked method can build on.
#
# `dataset_common.py` was split out of `dataset_mlm.py` so an autoregressive or encoder-decoder method can
# reuse framework/target splitting, context injection and length capping without inheriting masking.
# These tests pin the two properties that make that seam real:
#
#   * the neutral layer never requires a mask token — that requirement was the concrete coupling, and
#     a causal tokenizer has no mask token to give it;
#   * the masked-LM view is a *subclass*, so a fix to framework splitting or context dropout reaches
#     every method rather than only the one that happened to be edited.
# ====================================================================================================

DATASET_COMMON_CHAIN = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
         "fwr3": "FFFF", "cdr3": "GGG", "fwr4": "HHHH"}
DATASET_COMMON_RECORD = {"heavy": dict(DATASET_COMMON_CHAIN, full_aa="AAAACCDDDDEEFFFFGGGHHHH"),
          "light": dict(DATASET_COMMON_CHAIN, full_aa="AAAACCDDDDEEFFFFGGGHHHH"),
          "antigen": {"full_aa": "WWWW"}, "split": "train"}

def _dataset_file(n=4):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump([DATASET_COMMON_RECORD] * n, handle)
    handle.close()
    return handle.name


def _causal_adapter():
    """An adapter with NO mask token — what an autoregressive method would supply."""
    base = AntibodyTokenizer()
    return TokenizerAdapter(
        encode=base.encode, decode=base.decode, vocab_size=len(base.vocab), pad_id=base.pad_id,
        cls_id=None, end_id=None, chain_sep_id=base.tok_to_id["<sep>"],
        special_ids={base.tok_to_id[t] for t in ("<pad>", "<unk>", "<cls>", "<sep>")},
        uses_special_tokens=False, position_type="absolute")


class NeutralLayerTests(unittest.TestCase):
    def test_the_adapter_does_not_require_a_mask_token(self):
        self.assertIsNone(_causal_adapter().mask_id)

    def test_context_injection_works_without_a_mask_token(self):
        t = _causal_adapter()
        pt1, pt2 = inject_context(t, t.encode("AAAA"), t.encode("HHHH"), 3,
                                  partner_seq="KKKK", max_len=64)
        self.assertIn(t.chain_sep_id, pt1)       # partner chain prepended
        self.assertEqual(pt2, t.encode("HHHH"))

    def test_framework_parts_needs_no_masking_configuration(self):
        path = _dataset_file()
        ds = AntibodyFrameworkDataset(path, _causal_adapter(), target_region="h3")
        parts = ds.framework_parts(0)
        self.assertEqual(parts["target"], "GGG")
        self.assertEqual(parts["pt1"], "AAAACCDDDDEEFFFF")
        self.assertEqual(parts["pt2"], "HHHH")
        self.assertEqual(parts["target_length"], 3)
        os.unlink(path)

    def test_a_causal_subclass_can_lay_out_its_own_inputs(self):
        class Causal(AntibodyFrameworkDataset):
            def __getitem__(self, idx):
                f = self.framework_parts(idx)
                t = self.tokenizer
                pt1, pt2 = inject_context(t, t.encode(f["pt1"]), t.encode(f["pt2"]),
                                          f["target_length"], max_len=self.max_len)
                return torch.tensor(pt1 + t.encode(f["target"]) + pt2)
        path = _dataset_file()
        ds = Causal(path, _causal_adapter(), target_region="h3")
        self.assertEqual(len(ds), 4)
        self.assertEqual(len(ds[0]), 23)          # the whole chain, nothing masked out
        os.unlink(path)


class SharedWithTheMaskedViewTests(unittest.TestCase):
    def test_the_masked_dataset_is_a_subclass_of_the_neutral_one(self):
        # Not a copy: a fix to splitting or context dropout must reach every method.
        self.assertTrue(issubclass(MaskedInfillingDataset, AntibodyFrameworkDataset))

    def test_both_views_resolve_identical_framework_parts(self):
        path = _dataset_file()
        kw = dict(target_region="h3", include_partner_chain=True, context_dropout=0.5,
                  context_dropout_seed=3)
        neutral = AntibodyFrameworkDataset(path, _causal_adapter(), **kw)
        masked = MaskedInfillingDataset(path, TokenizerAdapter.from_custom(AntibodyTokenizer()), **kw)
        for i in range(4):
            self.assertEqual(neutral.framework_parts(i), masked.framework_parts(i))
        os.unlink(path)

    def test_framework_for_dataset_unwraps_a_subset(self):
        path = _dataset_file()
        ds = AntibodyFrameworkDataset(path, _causal_adapter(), target_region="h3")
        subset = torch.utils.data.Subset(ds, [2, 3])
        self.assertEqual(framework_for_dataset(subset, 0), ds.framework_parts(2))
        os.unlink(path)

    def test_context_dropout_is_deterministic_across_methods(self):
        path = _dataset_file(20)
        kw = dict(target_region="h3", include_partner_chain=True, context_dropout=0.5,
                  context_dropout_seed=11)
        a = AntibodyFrameworkDataset(path, _causal_adapter(), **kw)
        b = MaskedInfillingDataset(path, TokenizerAdapter.from_custom(AntibodyTokenizer()), **kw)
        kept_a = [a.framework_parts(i)["partner_chain"] is not None for i in range(20)]
        kept_b = [b.framework_parts(i)["partner_chain"] is not None for i in range(20)]
        self.assertEqual(kept_a, kept_b)
        self.assertIn(False, kept_a)              # dropout actually dropped something
        os.unlink(path)

class HFAdapterTests(unittest.TestCase):
    def test_runtime_added_tokens_count_toward_vocab_and_tokenizer_is_saveable(self):
        class FakeTokenizer:
            vocab_size = 2
            pad_token_id = 0
            mask_token_id = 3
            cls_token_id = None
            bos_token_id = 1
            eos_token_id = 2
            sep_token_id = None
            all_special_ids = [0, 1, 2, 3]

            def __len__(self):
                return 4

            def encode(self, seq, add_special_tokens=False):
                return [1] * len(seq)

            def decode(self, ids):
                return "A" * len(ids)

            def save_pretrained(self, path):
                return path

        tokenizer = FakeTokenizer()
        adapter = TokenizerAdapter.from_hf(
            tokenizer, position_type="absolute", uses_special_tokens=True)

        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.vocab_size, 4)
        self.assertIs(adapter.backend_tokenizer, tokenizer)
        self.assertTrue(callable(adapter.save_pretrained))


class DatasetContractTests(unittest.TestCase):
    def select(self, records, eval_split=""):
        return select_split_indices(records, eval_split, data_path="probe.json", seed=42)

    def test_missing_and_null_splits_use_the_same_seeded_random_partition(self):
        missing = [{} for _ in range(20)]
        expected = self.select(missing)
        for records in ([{"split": None} for _ in missing],
                        [{"split": None} if i % 2 else {} for i in range(20)]):
            self.assertEqual(self.select(records), expected)
            with self.assertRaisesRegex(ValueError, "named split"):
                self.select(records, "validation")
        train, evaluation, label = expected
        self.assertEqual((len(train), len(evaluation)), (18, 2))
        self.assertFalse(set(train) & set(evaluation))
        self.assertEqual(label, "random split")

    def test_partially_labelled_corpora_never_repartition_the_holdout(self):
        for unlabelled in ({}, {"split": None}):
            for selected in ("", "test"):
                with self.subTest(unlabelled=unlabelled, selected=selected):
                    with self.assertRaisesRegex(ValueError, "partially labelled"):
                        self.select([{"split": "test"}] + [unlabelled] * 9, selected)

    def test_named_splits_keep_membership_and_explicit_test_selection(self):
        records = [{"split": name} for name in ("train", "test", "validation", "train")]
        self.assertEqual(self.select(records, "validation"), ([0, 3], [2], "validation"))
        self.assertEqual(self.select(records, "test"), ([0, 3], [1], "test"))
        with self.assertRaisesRegex(ValueError, "eval_split is unset"):
            self.select(records)

    def test_invalid_split_types_are_rejected_before_sorting(self):
        for label in ("", " ", 0, False, [], {}):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "Invalid split label"):
                self.select([{"split": "train"}, {"split": label}])

    def test_all_supported_targets_and_legacy_aliases_select_the_right_chain_and_loop(self):
        path = _dataset_file()
        self.addCleanup(os.unlink, path)
        for chain, prefixes in (("heavy", ("h", "cdr")), ("light", ("l", "lcdr"))):
            for number in (1, 2, 3):
                for prefix in prefixes:
                    name = f"{prefix}{number}"
                    with self.subTest(target=name):
                        self.assertEqual(AntibodyFrameworkDataset._resolve_target(name.upper()),
                                         (chain, f"cdr{number}"))
                        ds = AntibodyFrameworkDataset(path, _causal_adapter(), target_region=name)
                        self.assertEqual(ds.framework_parts(0)["target"], DATASET_COMMON_CHAIN[f"cdr{number}"])

    def test_invalid_targets_fail_at_construction_in_all_methods(self):
        from cdrseqgen.datasets.dataset_ar import CausalInfillingDataset
        from cdrseqgen.datasets.dataset_s2s import Seq2SeqInfillingDataset
        for dataset_class in (AntibodyFrameworkDataset, MaskedInfillingDataset,
                              CausalInfillingDataset, Seq2SeqInfillingDataset):
            for target in ("h4", "typo3", "", None, 3):
                with self.subTest(method=dataset_class.__name__, target=target):
                    with self.assertRaisesRegex(ValueError, "Unsupported target_region"):
                        dataset_class("does-not-exist.json", _causal_adapter(), target_region=target)
        with self.assertRaisesRegex(ValueError, "Unsupported CDR region"):
            AntibodyFrameworkDataset._split_regions(DATASET_COMMON_CHAIN, "cdr4")

    def test_mlm_length_budget_includes_wrappers_and_preserves_the_full_target(self):
        path = _dataset_file()
        self.addCleanup(os.unlink, path)
        for wrapped in (False, True):
            tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
            tokenizer.uses_special_tokens = wrapped
            tokenizer.cls_id, tokenizer.end_id = 2, 3
            minimum = len(DATASET_COMMON_CHAIN["cdr3"]) + (2 if wrapped else 0)
            for cap in (minimum - 1, minimum, minimum + 2):
                with self.subTest(wrapped=wrapped, max_len=cap):
                    ds = MaskedInfillingDataset(path, tokenizer, max_len=cap,
                                                include_partner_chain=True, include_antigen=True)
                    if cap < minimum:
                        with self.assertRaisesRegex(ValueError, "cannot hold.*target"):
                            ds[0]
                    else:
                        item = ds[0]
                        self.assertLessEqual(len(item["input_ids"]), cap)
                        labelled = item["labels"][item["labels"] != -100].tolist()
                        self.assertEqual(labelled, tokenizer.encode(DATASET_COMMON_CHAIN["cdr3"]))
                        if wrapped:
                            self.assertEqual(item["input_ids"][[0, -1]].tolist(), [2, 3])


# ====================================================================================================
# from test_dataset_s2s.py
# Encoder-decoder span-infilling data: the two-stream properties an s2s method depends on.
#
# The science-critical ones are leakage and the shift. The encoder must carry no residue and no
# length of the loop it is asking for — otherwise "length-free" generation is reading the answer —
# and the decoder pair must be offset by exactly one, since a double shift still trains and produces
# a checkpoint that looks merely bad rather than broken.
# ====================================================================================================

DATASET_S2S_CHAIN = {"fwr1": "AAAA", "cdr1": "CC", "fwr2": "DDDD", "cdr2": "EE",
         "fwr3": "FFFF", "cdr3": "GGGGG", "fwr4": "HHHH"}
DATASET_S2S_RECORD = {"heavy": dict(DATASET_S2S_CHAIN, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
          "light": dict(DATASET_S2S_CHAIN, full_aa="AAAACCDDDDEEFFFFGGGGGHHHH"),
          "antigen": {"full_aa": "WWWW"}, "split": "train"}

def _dataset_s2s_data(n=4):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump([DATASET_S2S_RECORD] * n, handle)
    handle.close()
    return handle.name


class DatasetS2sVocabTests(unittest.TestCase):
    def test_the_shared_vocab_is_a_prefix_of_the_seq2seq_one(self):
        shared, s2s = AntibodyTokenizer(), Seq2SeqTokenizer()
        self.assertEqual(s2s.vocab[:len(shared.vocab)], shared.vocab)
        self.assertEqual(len(shared.vocab), 25)
        self.assertEqual((s2s.bos_id, s2s.eos_id), (25, 26))

    def test_the_seq2seq_and_causal_vocabularies_agree_token_for_token(self):
        # dataset_s2s duplicates the <bos>/<eos> extension rather than importing it from the causal
        # module, so the two methods stay independently testable. Drift is the failure that buys,
        # and it would surface as a checkpoint decoding to the wrong residues, so it is asserted.
        self.assertEqual(Seq2SeqTokenizer().vocab, CausalTokenizer().vocab)


class EncoderStreamTests(unittest.TestCase):
    def setUp(self):
        self.path = _dataset_s2s_data()
        self.tok = Seq2SeqTokenizer()
        self.adapter = s2s_adapter(self.tok)
        self.ds = Seq2SeqInfillingDataset(self.path, self.adapter, target_region="h3")

    def tearDown(self):
        os.unlink(self.path)

    def test_both_flanks_are_present_and_separated_by_one_gap_marker(self):
        encoder_ids = self.ds[0]["input_ids"].tolist()
        self.assertEqual(encoder_ids.count(self.adapter.mask_id), ENCODER_LAYOUT_TOKENS)
        gap = encoder_ids.index(self.adapter.mask_id)
        self.assertEqual(self.tok.decode(encoder_ids[:gap]), "AAAACCDDDDEEFFFF")
        self.assertEqual(self.tok.decode(encoder_ids[gap + 1:]), "HHHH")

    def test_the_encoder_never_carries_the_loop(self):
        # "GGGGG" is the native H3. If it reaches the encoder the task is copying, not design.
        encoder_ids = self.ds[0]["input_ids"].tolist()
        self.assertNotIn("G", self.tok.decode(encoder_ids))

    def test_the_encoder_input_does_not_depend_on_the_loop_length(self):
        # The gap marker is ONE token whether the loop is 2 residues or 30. A marker that scaled
        # with the answer would hand the model the length it is supposed to predict — the §10
        # length-head problem, reintroduced through the back door.
        base = dict(pt1="AAAACCDDDDEEFFFF", pt2="HHHH", partner_chain=None, antigen=None,
                    max_len=128, target="GG", target_length=2)
        short, _ = build_encoder_input(self.adapter, base)
        long, _ = build_encoder_input(self.adapter, dict(base, target="G" * 30, target_length=30))
        self.assertEqual(short, long)

    def test_context_is_capped_against_the_whole_budget(self):
        # The reserve fix, asserted rather than described: at one max_len the causal prompt must
        # hold back LOOP_LENGTH_RESERVE positions for an answer in the same stream, while the
        # encoder spends one token on the gap. So s2s retains strictly more context here.
        framework = dict(pt1="A" * 200, pt2="H" * 40, partner_chain="P" * 80, antigen=None,
                         max_len=128, target="GGGGG", target_length=5)
        encoder_ids, _ = build_encoder_input(self.adapter, framework)
        causal_prompt, _ = build_causal_prompt(causal_adapter(CausalTokenizer()), dict(framework))
        self.assertGreater(len(encoder_ids), len(causal_prompt))
        self.assertLessEqual(len(encoder_ids), framework["max_len"])


class DecoderStreamTests(unittest.TestCase):
    def setUp(self):
        self.path = _dataset_s2s_data()
        self.tok = Seq2SeqTokenizer()
        self.adapter = s2s_adapter(self.tok)
        self.ds = Seq2SeqInfillingDataset(self.path, self.adapter, target_region="h3")

    def tearDown(self):
        os.unlink(self.path)

    def test_the_pair_is_offset_by_exactly_one(self):
        target = self.adapter.encode("GGGGG")
        decoder_input, labels = build_decoder_pair(self.adapter, target)
        self.assertEqual(len(decoder_input), len(labels))
        self.assertEqual(decoder_input[0], self.adapter.bos_id)
        self.assertEqual(labels[-1], self.adapter.eos_id)
        # position i of the input predicts position i of the labels, which is input position i+1
        self.assertEqual(decoder_input[1:], labels[:-1])

    def test_the_loop_is_supervised_in_full_with_no_ignore_index(self):
        # Unlike the causal layout there is no prompt in this stream, so answer-only supervision is
        # structural. A -100 appearing here would mean framework tokens leaked into the decoder.
        item = self.ds[0]
        self.assertNotIn(-100, item["labels"].tolist())
        self.assertEqual(self.tok.decode(item["labels"].tolist()[:-1]), "GGGGG")
        self.assertEqual(item["target_len"], 5)

    def test_a_loop_beyond_the_decoder_budget_is_rejected_at_construction(self):
        oversized = dict(DATASET_S2S_RECORD)
        oversized["heavy"] = dict(DATASET_S2S_CHAIN, cdr3="G" * (MAX_LOOP_RESIDUES + 1))
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump([oversized], handle)
        handle.close()
        try:
            ds = Seq2SeqInfillingDataset(handle.name, self.adapter, target_region="h3")
            with self.assertRaisesRegex(ValueError, "decoder budget"):
                ds[0]
        finally:
            os.unlink(handle.name)

    def test_the_decode_cap_leaves_a_step_for_the_terminator(self):
        self.assertEqual(S2S_DECODE_STEPS, MAX_LOOP_RESIDUES + 1)
        self.assertEqual(validate_s2s_decode_steps(S2S_DECODE_STEPS), S2S_DECODE_STEPS)
        for bad in (0, -1, S2S_DECODE_STEPS + 1, True, 3.0):
            with self.assertRaises(ValueError):
                validate_s2s_decode_steps(bad)


class CollatorTests(unittest.TestCase):
    def setUp(self):
        self.adapter = s2s_adapter()

    def _item(self, encoder_len, loop):
        target = self.adapter.encode(loop)
        decoder_input, labels = build_decoder_pair(self.adapter, target)
        return {
            "input_ids": torch.arange(1, encoder_len + 1, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_len": len(target),
            "pt1_len": 0,
        }

    def test_the_streams_are_padded_independently(self):
        # A long framework with a short loop must not stretch the decoder to the encoder's length.
        batch = s2s_collate_fn([self._item(30, "GG"), self._item(8, "GGGGGGGG")])
        self.assertEqual(batch["input_ids"].shape[1], 30)
        self.assertEqual(batch["decoder_input_ids"].shape[1], 9)     # 8 residues + <bos>
        self.assertEqual(batch["labels"].shape[1], 9)

    def test_padding_is_masked_on_both_sides_and_ignored_by_the_loss(self):
        batch = s2s_collate_fn([self._item(30, "GG"), self._item(8, "GGGGGGGG")])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1] * 8 + [0] * 22)
        self.assertEqual(batch["decoder_attention_mask"][0].tolist(), [1] * 3 + [0] * 6)
        self.assertEqual(batch["labels"][0].tolist()[3:], [-100] * 6)


# ====================================================================================================
# from test_span_layout.py
# Span layouts: a second arrangement must actually change the example, and nothing else.
#
# The abstraction exists so a pretrained infiller can keep the format it was pretrained on. It is
# only worth having if a different layout genuinely produces a different example while the mechanics
# around it -- framework split, context injection, capping, answer-only loss -- stay identical. One
# layout cannot demonstrate that, so these tests supply a second.
# ====================================================================================================

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
        from cdrseqgen.methods.ar.training import build_trainer
        from cdrseqgen.methods.common.training import train_and_save

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
        trainer = build_trainer(model, args, ds, ds, self.adapter, callbacks=[])
        train_and_save(trainer, args, tokenizer=self.adapter)
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
