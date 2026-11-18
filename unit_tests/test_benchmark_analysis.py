"""Tests merged from: test_benchmark_scorer.py, test_scoring_contracts.py, test_bootstrap.py, test_variability.py, test_benchmark_plotter.py, test_benchmark_track_contract.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

import unittest
from benchmark.scorer import (
    aggregate_records,
    assert_comparable,
    hamming_fraction,
    score_record,
    validate_adapter_record,
)
from unittest.mock import patch
import torch
from cdrseqgen.eval.generative import Draw, normalize_scores, run_eval_loop
import json
import os
import sys
import tempfile
import contextlib
import io
from pathlib import Path
import random
import statistics
import zlib
import numpy as np
from safetensors.numpy import save_file
from benchmark import variability as V
from benchmark._common import sha256_file
from benchmark.analysis import crossed_interval
from benchmark.extend import EXTENSION
from benchmark.scorer import score_payload
from benchmark.plot_comparison import (
    SCORE_SCHEMA_VERSION,
    benchmark_track,
    build_figure,
    is_development_artifact,
    normalized_position_curve,
    save_comparison_figure,
    validate_scores,
)
from benchmark.scorer import ADAPTER_SCHEMA_VERSION, validate_payload


# ====================================================================================================
# from test_benchmark_scorer.py
# ====================================================================================================

def adapter_record(native="AAAAAA", samples=None):
    samples = samples if samples is not None else ["AAAAAA", "AACAAT"]
    return {
        "example_id": "example-1",
        "native": native,
        "native_length": len(native),
        "samples": samples,
        "attempts": 3,
        "valid_attempts": 2,
        "invalid_attempts": 1,
        "length_matches": 2,
        "length_mismatches": 0,
        "complete": True,
        "runtime_seconds": 0.6,
    }


class BenchmarkScorerTests(unittest.TestCase):
    def test_scores_whole_core_anchor_and_duplicate_preserving_design_metrics(self):
        record = adapter_record(samples=["AAAAAA", "AACAAT"])
        metrics = score_record(record, anchor_trim=2)

        self.assertAlmostEqual(metrics["whole_span_aar_mean"], 5 / 6)
        self.assertAlmostEqual(metrics["core_aar_mean"], 0.75)
        self.assertAlmostEqual(metrics["anchor_aar_mean"], 0.875)
        self.assertAlmostEqual(metrics["diversity_hamming"], 1 / 3)
        self.assertEqual(metrics["unique_fraction"], 1.0)
        self.assertAlmostEqual(metrics["valid_generation_rate"], 2 / 3)
        self.assertEqual(metrics["length_match_rate"], 1.0)
        self.assertAlmostEqual(metrics["runtime_seconds_per_retained_sample"], 0.3)

    def test_short_loop_has_no_core_and_anchors_cover_each_position_once(self):
        record = adapter_record(native="AAAA", samples=["AAAT", "AAAA"])
        record.update(attempts=2, valid_attempts=2, invalid_attempts=0)
        metrics = score_record(record, anchor_trim=2)

        self.assertIsNone(metrics["core_aar_mean"])
        self.assertFalse(metrics["core_available"])
        self.assertEqual(metrics["anchor_length"], 4)
        self.assertAlmostEqual(metrics["anchor_aar_mean"], 0.875)

    def test_duplicate_draws_reduce_unique_fraction_and_pairwise_diversity(self):
        record = adapter_record(samples=["AAAAAA", "AAAAAA", "TTTTTT"])
        record.update(attempts=3, valid_attempts=3, invalid_attempts=0, length_matches=3)
        metrics = score_record(record, anchor_trim=2)

        self.assertAlmostEqual(metrics["unique_fraction"], 2 / 3)
        self.assertAlmostEqual(metrics["diversity_hamming"], 2 / 3)
        self.assertEqual(hamming_fraction("AAA", "AAT"), 1 / 3)

    def test_rejects_inconsistent_adapter_accounting_or_bad_samples(self):
        record = adapter_record()
        manifest = {"native": "AAAAAA", "native_length": 6}
        validate_adapter_record(record, manifest)

        with self.assertRaisesRegex(ValueError, "attempt accounting"):
            validate_adapter_record({**record, "attempts": 4}, manifest)
        with self.assertRaisesRegex(ValueError, "wrong length"):
            validate_adapter_record({**record, "samples": ["AAA"]}, manifest)

    def test_length_free_scoring_keeps_unequal_lengths_but_conditions_aar(self):
        record = adapter_record(native="AAAA", samples=["AAAA", "AAA", "AAAAA"])
        record.update(
            attempts=3, valid_attempts=3, invalid_attempts=0,
            length_matches=1, length_mismatches=2,
        )
        manifest = {"native": "AAAA", "native_length": 4}
        validate_adapter_record(record, manifest, track="length_free")
        metrics = score_record(record, anchor_trim=1)
        self.assertEqual(metrics["whole_span_aar_mean"], 1.0)
        self.assertAlmostEqual(metrics["length_match_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["length_absolute_error_mean"], 2 / 3)
        self.assertGreater(metrics["edit_similarity_mean"], 0.8)
        with self.assertRaisesRegex(ValueError, "retained|wrong length"):
            validate_adapter_record(record, manifest, track="fixed_native_length")

    def test_comparison_requires_same_examples_and_sampling_controls(self):
        base = {
            "sampling": {
                "track": "fixed_native_length", "num_samples": 5,
                "temperature": 1.0, "top_p": 0.9, "base_seed": 42,
            }
        }
        assert_comparable([("a", base, ["x"]), ("b", base, ["x"])])
        with self.assertRaisesRegex(ValueError, "framework IDs"):
            assert_comparable([("a", base, ["x"]), ("b", base, ["y"])])
        changed = {"sampling": {**base["sampling"], "top_p": 1.0}}
        with self.assertRaisesRegex(ValueError, "top_p"):
            assert_comparable([("a", base, ["x"]), ("b", changed, ["x"])])

    def test_aggregate_is_macro_average_over_examples(self):
        first = adapter_record(samples=["AAAAAA"])
        first.update(attempts=1, valid_attempts=1, invalid_attempts=0, length_matches=1)
        second = adapter_record(samples=["TTTTTT", "TTTTTT", "TTTTTT"])
        second.update(attempts=3, valid_attempts=3, invalid_attempts=0, length_matches=3)
        scored = []
        for record in (first, second):
            metrics = score_record(record, anchor_trim=2)
            scored.append({
                "metrics": metrics,
                "accounting": {
                    "attempts": record["attempts"],
                    "valid_attempts": record["valid_attempts"],
                    "length_matches": record["length_matches"],
                },
            })
        aggregate = aggregate_records(scored)
        self.assertEqual(aggregate["whole_span_aar_mean"]["macro_mean"], 0.5)
        self.assertEqual(aggregate["micro_accounting"]["retained_samples"], 4)


# ====================================================================================================
# from test_scoring_contracts.py
# Keep candidate populations intact when generator/judge scores are unavailable.
# ====================================================================================================

class Strategy:
    def __init__(self, scores):
        self.scores = scores
        self.self_calls = 0

    def draw(self, index):
        return Draw(native="ACD", seqs=["ACD", "AQD", "AC"], spans=[None] * 3,
                    target_len=3, length_policy={"mode": "inferred"})

    def self_scores(self, drawn):
        self.self_calls += 1
        return self.scores


def evaluate(strategy, judge_scores=None, *, skip_self=False):
    with patch("cdrseqgen.eval.generative.score_with_judge", return_value=judge_scores):
        return run_eval_loop(
            strategy, [7], model=None, tokenizer=None, device="cpu",
            skip_perplexity=skip_self, judge_model=object() if judge_scores is not None else None,
            judge_tokenizer=None,
        )


class ScoreContractTests(unittest.TestCase):
    def test_ordered_scalar_iterators_are_normalized_without_changing_finite_values(self):
        values = iter([2.5, None, torch.tensor(4.0), 8])
        self.assertEqual(normalize_scores(values, 4, label="custom scorer"),
                         [2.5, None, 4.0, 8.0])
        self.assertEqual(normalize_scores([], 0, label="empty scorer"), [])

    def test_noniterable_and_nonnumeric_scores_have_a_labelled_error(self):
        with self.assertRaisesRegex(ValueError, "custom scorer.*one score"):
            normalize_scores(None, 1, label="custom scorer")
        for invalid in (True, "2.0", object(), {}, [2.0], torch.tensor([2.0]),
                        torch.tensor(True), 2 + 3j):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "custom scorer score 0.*scalar"):
                    normalize_scores([invalid], 1, label="custom scorer")

    def test_short_and_long_self_score_lists_fail_before_returning_partial_records(self):
        for scores in ([2.0], [2.0, 3.0, 4.0, 5.0]):
            with self.subTest(count=len(scores)):
                with self.assertRaisesRegex(ValueError, r"self scorer at eval index 7.*scores for 3 candidates"):
                    evaluate(Strategy(scores))

    def test_judge_score_count_is_checked_even_when_generator_scoring_is_skipped(self):
        strategy = Strategy(None)
        with self.assertRaisesRegex(ValueError, r"judge scorer at eval index 7.*scores for 3 candidates"):
            evaluate(strategy, [2.0], skip_self=True)
        self.assertEqual(strategy.self_calls, 0)

    def test_nonfinite_scores_do_not_remove_candidates_or_inflate_judge_coverage(self):
        strategy = Strategy([2.0, float("nan"), float("inf")])
        with self.assertLogs("cdrseqgen.eval.generative", level="WARNING"):
            records = evaluate(strategy, [4.0, float("-inf"), 8.0])
        record = records[0]
        self.assertEqual(record["samples"], ["ACD", "AQD", "AC"])
        self.assertEqual([c["seq"] for c in record["candidates"]], record["samples"])
        self.assertEqual([c["self_perplexity"] for c in record["candidates"]], [2.0, None, None])
        self.assertEqual([c["judge_perplexity"] for c in record["candidates"]], [4.0, None, 8.0])
        self.assertEqual(record["metrics"]["self_perplexity_mean"], 2.0)
        self.assertEqual(record["metrics"]["judge_perplexity_mean"], 6.0)
        self.assertEqual(record["metrics"]["judge_candidates_scored"], 2)
        self.assertEqual(record["metrics"]["judge_coverage"], 2 / 3)

    def test_all_unavailable_scores_leave_no_misleading_mean(self):
        with self.assertLogs("cdrseqgen.eval.generative", level="WARNING"):
            record = evaluate(Strategy([float("nan"), None, float("inf")]),
                              [None, float("nan"), float("inf")])[0]
        self.assertEqual(len(record["candidates"]), 3)
        self.assertNotIn("self_perplexity_mean", record["metrics"])
        self.assertNotIn("judge_perplexity_mean", record["metrics"])
        self.assertEqual(record["metrics"]["judge_candidates_scored"], 0)
        self.assertEqual(record["metrics"]["judge_coverage"], 0.0)

    def test_finite_scores_keep_the_existing_mean_and_candidate_order(self):
        record = evaluate(Strategy([2.0, 4.0, 6.0]))[0]
        self.assertEqual(record["metrics"]["self_perplexity_mean"], 4.0)
        self.assertEqual([c["self_perplexity"] for c in record["candidates"]], [2.0, 4.0, 6.0])
        self.assertEqual([c["judge_perplexity"] for c in record["candidates"]], [None] * 3)


class CaseScoreContractTests(unittest.TestCase):
    def test_eos_case_rejects_short_scores_and_retains_candidates_with_nonfinite_scores(self):
        from types import SimpleNamespace
        from cdrseqgen.methods.common.eos_decoding import eos_case_record
        args = SimpleNamespace(max_len=64, partner_chain=None, antigen=None, num_samples=2,
            temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=5, skip_perplexity=False, seed=42)
        options = dict(mode="causal", build_context=lambda *_: ([1, 2], {}),
            sample_loops=lambda *a, **kw: [dict(loop="AC", complete=True), dict(loop="AD", complete=True)])
        tokenizer = SimpleNamespace(encode=lambda seq: list(range(len(seq))))
        case = dict(pt1="AC", pt2="DE", target_truth="AC")
        with self.assertRaisesRegex(ValueError, "case self scorer"):
            eos_case_record(None, tokenizer, "cpu", args, None, case, lambda *a: None,
                            perplexities=lambda *a: [2.0], **options)
        record = eos_case_record(None, tokenizer, "cpu", args, None, case, lambda *a: None,
                                perplexities=lambda *a: [float("inf"), 2.0], **options)
        self.assertEqual(len(record["samples"]), 2)
        self.assertNotIn("perplexity", record["samples"][0])
        self.assertEqual(record["metrics"]["perplexity_mean"], 2.0)

    def test_mlm_case_excludes_nonfinite_scores_from_samples_and_summary(self):
        from types import SimpleNamespace
        from cdrseqgen.methods.mlm import generation
        args = SimpleNamespace(max_len=64, partner_chain=None, antigen=None, num_samples=1,
            temperature=1.0, top_k=0, top_p=1.0, decoding="independent", skip_perplexity=False, seed=42)
        tokenizer = SimpleNamespace(decode=lambda ids: "AC", special_ids={0}, mask_id=4)
        span = SimpleNamespace(token_ids=[5, 6], input_ids=[4, 4], mask_positions=[0, 1])
        with patch.object(generation, "sample_length_conditioned_spans", return_value=([span], {})), \
                patch.object(generation, "pseudo_log_likelihood", return_value=(0.0, float("nan"))):
            record = generation.generate_case(None, tokenizer, "cpu", args, None,
                dict(pt1="AC", pt2="DE", target_length=2), None, lambda *a: None)
        self.assertEqual(len(record["samples"]), 1)
        self.assertNotIn("pseudo_perplexity", record["samples"][0])
        self.assertNotIn("pseudo_perplexity_mean", record["metrics"])


# ====================================================================================================
# from test_bootstrap.py
# Tests for the paired bootstrap in benchmark/bootstrap.py.
#
# The three properties worth pinning are the ones that decide whether an interval means anything:
#
#   * **pairing** — one index draw must be applied to both arms. An unpaired bootstrap of the same
#     data gives a much wider interval, so a bug here silently converts resolved findings into
#     unresolved ones (or, with the error in the other direction, manufactures significance).
#   * **the paired subset** — a metric may be missing on different examples for each model
#     (positional AAR is defined only where a candidate matched the native length), and the two arms
#     must be averaged over the *same* examples or they are being scored on different problems.
#   * **direction** — for a lower-is-better metric like perplexity, a positive difference means the
#     first model is *worse*. Getting this backwards would name the losing model as the winner while
#     every number on the row stays correct.
# ====================================================================================================

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmark"))
from bootstrap import compare, load_runs, paired_bootstrap, paired_values

def _scores(records_a, records_b):
    """A minimal two-run scores payload; each record list is (example_id, {metric: value})."""
    return {
        "manifest": {"id": "test_manifest"},
        "judge": {"model": "test-judge"},
        "runs": [
            {"label": "cdrseqgen",
             "records": [{"example_id": i, "metrics": m} for i, m in records_a]},
            {"label": "iglm",
             "records": [{"example_id": i, "metrics": m} for i, m in records_b]},
        ],
    }


def _write(payload):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(payload, handle)
    handle.close()
    return handle.name


class PairedSubsetTests(unittest.TestCase):
    def test_only_examples_both_runs_report_are_compared(self):
        payload = _scores(
            [("a", {"core_aar_mean": 0.2}), ("b", {"core_aar_mean": None}),
             ("c", {"core_aar_mean": 0.4})],
            [("a", {"core_aar_mean": 0.1}), ("b", {"core_aar_mean": 0.9}),
             ("c", {"core_aar_mean": None})],
        )
        _, runs = load_runs(_write(payload))
        ids, values_a, values_b = paired_values(runs, "core_aar_mean")
        # Only "a" has a finite value on both sides; b and c are each missing on one arm.
        self.assertEqual(ids, ["a"])
        self.assertEqual((values_a, values_b), ([0.2], [0.1]))

    def test_a_metric_missing_everywhere_is_dropped_not_zeroed(self):
        payload = _scores([("a", {"core_aar_mean": None})], [("a", {"core_aar_mean": None})])
        _, runs = load_runs(_write(payload))
        self.assertEqual(paired_values(runs, "core_aar_mean"), ([], [], []))
        self.assertIsNone(paired_bootstrap([], []))


class PairingTests(unittest.TestCase):
    def test_a_constant_offset_gives_a_zero_width_interval(self):
        # Every example differs by exactly +0.1, so the difference has no variance and pairing must
        # cancel the (large) between-example spread entirely. An unpaired bootstrap would not.
        values_a = [0.1, 0.5, 0.9, 0.3, 0.7]
        values_b = [v - 0.1 for v in values_a]
        stats = paired_bootstrap(values_a, values_b, iterations=2000, seed=1)
        self.assertAlmostEqual(stats["difference"], 0.1, places=9)
        self.assertAlmostEqual(stats["ci_low"], 0.1, places=9)
        self.assertAlmostEqual(stats["ci_high"], 0.1, places=9)
        self.assertTrue(stats["excludes_zero"])

    def test_the_interval_brackets_the_observed_difference(self):
        stats = paired_bootstrap([0.5, 0.2, 0.9, 0.1], [0.4, 0.4, 0.4, 0.4],
                                 iterations=2000, seed=7)
        self.assertLessEqual(stats["ci_low"], stats["difference"])
        self.assertGreaterEqual(stats["ci_high"], stats["difference"])

    def test_noise_around_zero_does_not_resolve(self):
        values_a = [0.30, 0.50, 0.20, 0.60, 0.40]
        values_b = [0.50, 0.30, 0.60, 0.20, 0.40]
        stats = paired_bootstrap(values_a, values_b, iterations=4000, seed=3)
        self.assertFalse(stats["excludes_zero"])

    def test_resampling_is_deterministic_for_a_seed(self):
        args = ([0.1, 0.4, 0.9], [0.2, 0.2, 0.2])
        first = paired_bootstrap(*args, iterations=500, seed=11)
        second = paired_bootstrap(*args, iterations=500, seed=11)
        self.assertEqual(first, second)


class DirectionTests(unittest.TestCase):
    def test_lower_is_better_names_the_smaller_value_as_the_winner(self):
        # cdrseqgen perplexity is higher, so iglm must be credited despite the positive difference.
        payload = _scores(
            [(str(i), {"judge_perplexity_mean": 9.0}) for i in range(8)],
            [(str(i), {"judge_perplexity_mean": 4.0}) for i in range(8)],
        )
        result = compare(_write(payload), iterations=500)
        row = next(r for r in result["results"] if r["metric"] == "judge_perplexity_mean")
        self.assertGreater(row["difference"], 0)
        self.assertEqual(row["better"], "iglm")

    def test_higher_is_better_names_the_larger_value_as_the_winner(self):
        payload = _scores(
            [(str(i), {"diversity_hamming": 0.8}) for i in range(8)],
            [(str(i), {"diversity_hamming": 0.6}) for i in range(8)],
        )
        result = compare(_write(payload), iterations=500)
        row = next(r for r in result["results"] if r["metric"] == "diversity_hamming")
        self.assertEqual(row["better"], "cdrseqgen")

    def test_an_unresolved_metric_names_no_winner(self):
        payload = _scores(
            [("a", {"core_aar_mean": 0.30}), ("b", {"core_aar_mean": 0.50}),
             ("c", {"core_aar_mean": 0.20})],
            [("a", {"core_aar_mean": 0.50}), ("b", {"core_aar_mean": 0.20}),
             ("c", {"core_aar_mean": 0.35})],
        )
        result = compare(_write(payload), iterations=2000)
        row = next(r for r in result["results"] if r["metric"] == "core_aar_mean")
        self.assertIsNone(row["better"])


class PayloadTests(unittest.TestCase):
    def test_a_payload_without_exactly_two_runs_is_refused(self):
        payload = _scores([("a", {"core_aar_mean": 0.2})], [("a", {"core_aar_mean": 0.1})])
        payload["runs"] = payload["runs"][:1]
        with self.assertRaises(SystemExit):
            load_runs(_write(payload))

    def test_result_declares_what_uncertainty_it_does_not_include(self):
        payload = _scores(
            [("a", {"core_aar_mean": 0.2}), ("b", {"core_aar_mean": 0.3})],
            [("a", {"core_aar_mean": 0.1}), ("b", {"core_aar_mean": 0.2})],
        )
        scope = compare(_write(payload), iterations=100)["uncertainty_scope"]
        self.assertEqual(scope["resampling_unit"], "example")
        self.assertEqual(scope["candidate_pools"], "fixed_observed_draws")
        self.assertEqual(scope["multiplicity_adjustment"], "none")


# ====================================================================================================
# from test_variability.py
# Seed variability, determinism, capacity and sensitivity analysis on a synthetic world with real scoring.
#
# Candidates are generated per arm, scored by the real scorer with a stand-in judge, sealed into batches shaped like the
# real ones, and given tiny checkpoints, so the module's verification, numbers and refusals are all exercised.
# ====================================================================================================

AA = 'ACDEFGHIKLMNPQRSTVWY'
SAMPLES = 6
FRAMEWORKS = [(f'f{i}', ''.join(random.Random(i).choice(AA) for _ in range(9 + i))) for i in range(6)]
METHODS = V.METHODS

class Judge:
    def score(self, record, samples):
        return [3.0] * len(samples)


def mutate(native, rng, rate):
    out = [c if rng.random() > rate else rng.choice(AA) for c in native]
    if rng.random() < .3:
        out = out[:-1]
    if rng.random() < .2:
        out.append(rng.choice(AA))
    return ''.join(out)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return path


class World:
    """Everything the analysis reads, built under one temporary directory."""

    def __init__(self, root, references=False):
        self.root = Path(root)
        records = [dict(example_id=i, source_index=k, native=n, native_length=len(n), pt1='QQQ', pt2='WWW', parent_sequence='QQQ' + n + 'WWW',
                        infill_range=[3, 3 + len(n)]) for k, (i, n) in enumerate(FRAMEWORKS)]
        self.manifest = write(self.root / 'manifest.json', dict(benchmark_id='b', schema_version='m', task=dict(track='length_free', anchor_trim_each_end=2),
                                                                source={}, records=records))
        self.clusters = write(self.root / 'clusters.json', dict(cluster_ids=[0, 0, 1, 1, 2, 2]))
        self.by_id = {r['example_id']: r for r in records}
        self.digest = sha256_file(self.manifest)
        self.runs, self.arms, self.samples = {}, {}, {}
        # Original six, IgLM and the empirical pool, IgCraft: one generation batch each, as in the real project.
        for method in METHODS:
            for seed in V.ORIGINAL_SEEDS:
                self.arm('original', f'{method}_s{seed}', .30 if method == 'mlm' else .35, checkpoint=(method, 'S', seed))
        self.arm('baselines', 'iglm', .33)
        self.arm('baselines', 'empirical', .7)
        self.arm('igcraft_batch', 'igcraft', .45)
        for method in METHODS:
            for seed in V.NEW_SEEDS:
                self.arm('seeds', f'{method}_s{seed}', .30 if method == 'mlm' else .35)
            for k in (1, 2):
                self.arm('repeats', f'{method}_s42_r{k}', .30 if method == 'mlm' else .35, checkpoint=(method, f'r{k}', 42),
                         copy_of=f'{method}_s42' if method == 'ar' else None)
            for size in V.SIZES:
                for seed in V.SCAN_SEEDS:
                    self.arm('capacity', f'{method}_{size}_s{seed}', .30 if method == 'mlm' else .35, checkpoint=(method, size, seed))
        self.arm('t1', 'igcraft_t1', .6, interior_gaps=True)
        if references:
            for label, rate in (('retrieval', .5), ('retrieval_j', .55), ('retrieval_fine', .45)):
                self.arm('references', label, rate)
        self.build_main()
        for batch in ('seeds', 'repeats', 'capacity', 't1', 'references'):
            if batch in {a for a, _ in self.arms.values()}:
                self.batch(batch)

    def arm(self, batch, label, rate, checkpoint=None, copy_of=None, interior_gaps=False):
        path = self.root / batch / 'generations' / f'{label}.json'
        rng = random.Random(label)
        records = []
        for example_id, native in FRAMEWORKS:
            samples = ([s for s in self.samples[copy_of][example_id]] if copy_of else [mutate(native, rng, rate) for _ in range(SAMPLES)])
            raw = [dict(attempt=k + 1, loop=s, length=len(s), valid_generation=True, interior_gap=interior_gaps and k == 0) for k, s in enumerate(samples)]
            matches = sum(len(s) == len(native) for s in samples)
            records.append(dict(example_id=example_id, native=native, native_length=len(native), samples=samples, attempts=SAMPLES,
                                valid_attempts=SAMPLES, invalid_attempts=0, length_matches=matches, length_mismatches=SAMPLES - matches,
                                runtime_seconds=0.1, complete=True, raw_generations=raw))
        write(path, dict(schema_version='cdrseqgen.adapter_output.v1', adapter='x', benchmark_id='b', manifest=dict(sha256=self.digest),
                         sampling=dict(track='length_free', num_samples=SAMPLES), model={}, finished_at='done', records=records))
        with contextlib.redirect_stdout(io.StringIO()):  # the scorer reports progress per framework
            run = score_payload(label, path, json.loads(path.read_text()), self.by_id, 2, Judge())
        self.runs[label] = run
        self.samples[label] = {r['example_id']: r['samples'] for r in records}
        self.arms[label] = (batch, checkpoint)

    def checkpoint(self, label):
        """A tiny checkpoint whose size encodes the model size and whose values encode the run."""
        batch, key = self.arms[label]
        method, tag, seed = key
        sized = dict(S=40, M=120, L=300, XL=700)
        n = sized.get(tag, 40)
        directory = self.root / 'runs' / label
        base = np.random.default_rng(zlib.crc32(f'{method}:{seed}'.encode())).standard_normal(n).astype(np.float32)
        identical = method == 'ar'
        shift = 0 if (identical or tag == 'S') else (.05 if method == 's2s' else .3) * (1 if tag == 'r1' else 2 if tag == 'r2' else 0)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({'weight': base + np.float32(shift), 'bias': np.ones(3, dtype=np.float32)}, str(directory / 'model.safetensors'))
        epochs = 20 if (method == 'mlm' and tag == 'r2') else 11
        for step in (100, 200):
            write(directory / f'checkpoint-{step}' / 'trainer_state.json', dict(global_step=step, epoch=epochs * step / 200,
                  best_model_checkpoint=f'{directory}/checkpoint-100', best_metric=.3 + shift / 100))
        return directory

    def batch(self, name):
        runs = [self.runs[l] for l, (b, _) in self.arms.items() if b == name]
        arms = [dict(label=r['label'], checkpoint=str(self.checkpoint(r['label']))) for r in runs if self.arms[r['label']][1]]
        write(self.root / name / 'plan.json', dict(arms=arms, stage='full', externals=[dict(label=r['label'], adapter='retrieval') for r in runs
                                                                                    if name in ('references', 't1')]))
        write(self.root / name / 'scores/comparison.json', dict(manifest=dict(sha256=self.digest), runs=runs, judge={}, scoring={}))

    def build_main(self):
        """The merged comparison and its sealed files, shaped like an extension of the original follow-up."""
        original = [self.runs[l] for l, (b, _) in self.arms.items() if b in ('original', 'baselines', 'igcraft_batch')]
        for name in ('original', 'baselines', 'igcraft_batch'):
            arms = [dict(label=l, checkpoint=str(self.checkpoint(l))) for l, (b, c) in self.arms.items() if b == name and c]
            write(self.root / name / 'plan.json', dict(arms=arms, stage='full'))
        main = self.root / 'main'
        write(main / 'scores/comparison.json', dict(manifest=dict(sha256=self.digest), runs=original, judge={}, scoring={}))
        write(main / 'capacity.json', dict(rows=[dict(arm='iglm', system_parameters=1000), dict(arm='igcraft', system_parameters=5000, parameters_by_module=dict(a=1))]))
        write(main / 'plan.json', dict(schema_version=EXTENSION, manifest=str(self.manifest),
                                      clusters=str(self.clusters), externals=['iglm', 'igcraft'], analysis=dict(iterations=1000, seed=42)))
        write(main / 'complete.json', dict(artifacts={n: sha256_file(main / n) for n in ('scores/comparison.json', 'capacity.json')}))
        recipe = dict(name='test_variability', main=str(main), analysis=dict(iterations=1000, seed=42, anchor_trim=2),
                      references=None, **{k: str(self.root / d) for k, d in (('seeds', 'seeds'), ('repeats', 'repeats'), ('capacity', 'capacity'), ('igcraft_t1', 't1'))})
        self.recipe_path = write(self.root / 'recipe.json', recipe)


class VariabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.world = World(Path(cls.temp.name) / 'plain')
        cls.reference_world = World(Path(cls.temp.name) / 'with_references', references=True)
        cls.plan = V.build_plan(cls.world.recipe_path)
        cls.result = V.compute(cls.plan)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def edit(self, label):
        return V.aggregate(self.world.runs[label], 'edit_similarity_mean')

    def test_core_definition(self):
        self.assertEqual((V.core('ABCDEFG', 2), V.core('ABCD', 2), V.core('', 2)), ('CDE', '', ''))
        self.assertEqual((V.core_similarity('', ''), V.core_similarity('', 'ABC'), V.core_similarity('ABC', '')), (1.0, 0.0, 0.0))
        self.assertEqual(V.core_similarity('ABC', 'ABC'), 1.0)
        self.assertAlmostEqual(V.core_similarity('ABC', 'ABD'), 1 - 1 / 3)

    def test_the_plan_freezes_inputs_and_covers_every_generation_file(self):
        frozen = set(self.plan['frozen_files'])
        for label, run in self.world.runs.items():
            self.assertIn(str(Path(run['input']['path']).resolve()), frozen, label)
        self.assertIsNone(self.plan['references'])

    def test_seed_spread_matches_the_runs(self):
        for method in METHODS:
            values = [self.edit(f'{method}_s{s}') for s in V.ALL_SEEDS]
            block = self.result['seeds']['methods'][method]['metrics']['edit']
            self.assertAlmostEqual(block['mean'], statistics.fmean(values))
            self.assertAlmostEqual(block['sd'], statistics.stdev(values))
            self.assertEqual((block['n'], block['min'], block['max']), (7, min(values), max(values)))
            new = self.result['seeds']['methods'][method]['new_seed_edit']
            self.assertEqual(new['n'], 5)
        self.assertEqual(set(self.result['seeds']['pools']), {'iglm', 'empirical', 'igcraft', 'igcraft_t1'})

    def test_contrasts_use_both_endpoints_and_the_planned_family(self):
        contrasts = self.result['seeds']['contrasts']
        self.assertEqual(len(contrasts), 20)
        self.assertEqual({c['endpoint'] for c in contrasts}, {'edit', 'core'})
        mlm_ar = next(c for c in contrasts if (c['a'], c['b'], c['endpoint']) == ('mlm', 'ar', 'edit'))
        seven = lambda m: statistics.fmean(self.edit(f'{m}_s{s}') for s in V.ALL_SEEDS)
        self.assertAlmostEqual(mlm_ar['difference'], seven('mlm') - seven('ar'))
        self.assertEqual((mlm_ar['n_seeds_a'], mlm_ar['n_seeds_b'], mlm_ar['n_groups']), (7, 7, 3))
        vs_iglm = next(c for c in contrasts if (c['a'], c['b'], c['endpoint']) == ('mlm', 'iglm', 'edit'))
        self.assertEqual(vs_iglm['n_seeds_b'], 1)
        self.assertIn('ci_low', mlm_ar['frameworks_only'])
        self.assertEqual(len(self.result['seeds']['new_seed_contrasts']), 3)

    def test_contrast_matches_the_statistic_it_wraps(self):
        study = V.Study(json.loads(self.world.manifest.read_text()), json.loads(self.world.clusters.read_text()), self.plan['analysis'])
        runs = {m: [self.world.runs[f'{m}_s{s}'] for s in V.ALL_SEEDS] for m in ('mlm', 'ar')}
        direct = crossed_interval(study.matrix(runs['mlm'], 'edit'), study.matrix(runs['ar'], 'edit'), study.groups, 1000,
                                  V.stable_example_seed(42, 'mlm:ar:edit'), 10)
        reported = next(c for c in self.result['seeds']['contrasts'] if (c['a'], c['b'], c['endpoint']) == ('mlm', 'ar', 'edit'))
        self.assertEqual((reported['ci_low'], reported['ci_high']), (direct['ci_low'], direct['ci_high']))

    def test_core_vectors_come_from_the_saved_candidates(self):
        study = V.Study(json.loads(self.world.manifest.read_text()), json.loads(self.world.clusters.read_text()), self.plan['analysis'])
        run = self.world.runs['ar_s42']
        expected = [statistics.fmean(V.core_similarity(V.core(s, 2), V.core(n, 2)) for s in self.world.samples['ar_s42'][i]) for i, n in FRAMEWORKS]
        self.assertTrue(np.allclose(study.vector(run, 'core'), expected))

    def test_determinism_separates_identical_candidates_from_drifting_ones(self):
        rows = self.result['determinism']['methods']
        self.assertEqual(rows['ar']['identical_frameworks'], {'s42_r1': 6, 'r1_r2': 6, 's42_r2': 6})
        self.assertEqual(rows['ar']['weights']['s42_r1']['tensors_differing'], 0)
        self.assertEqual(rows['mlm']['identical_frameworks']['r1_r2'], 0)
        self.assertGreater(rows['mlm']['weights']['r1_r2']['max_abs_difference'], 0.2)
        self.assertGreater(rows['s2s']['weights']['s42_r1']['max_abs_difference'], 0)
        self.assertEqual([rows['mlm']['runs'][k]['trajectory']['epochs'] for k in ('s42', 'r1', 'r2')], [11.0, 11.0, 20.0])
        self.assertEqual(rows['ar']['edit_range'], 0.0)

    def test_capacity_rows_report_parameters_and_the_change_in_seed_sd(self):
        rows = {(r['method'], r['size']): r for r in self.result['capacity']['rows']}
        self.assertEqual(len(rows), 12)
        self.assertLess(rows[('ar', 'S')]['parameters'], rows[('ar', 'M')]['parameters'])
        self.assertLess(rows[('ar', 'L')]['parameters'], rows[('ar', 'XL')]['parameters'])
        small = statistics.fmean(self.edit(f'mlm_s{s}') for s in V.SCAN_SEEDS)
        large = statistics.fmean(self.edit(f'mlm_L_s{s}') for s in V.SCAN_SEEDS)
        self.assertAlmostEqual(rows[('mlm', 'L')]['edit_minus_small'], large - small)
        sd = self.result['capacity']['small_model_seed_sd']['mlm']
        self.assertAlmostEqual(rows[('mlm', 'L')]['edit_minus_small_in_seed_sd'], (large - small) / sd)
        self.assertEqual(rows[('mlm', 'S')]['edit_minus_small'], 0.0)
        self.assertEqual(self.result['external_parameters']['igcraft']['by_module'], {'a': 1})

    def test_sensitivity_compares_lengths_gaps_and_scores(self):
        sensitivity = self.result['sensitivity']
        self.assertEqual(sensitivity['arms']['default']['label'], 'igcraft')
        self.assertEqual(sensitivity['arms']['sensitivity']['lengths']['interior_gap_slots'], 6)  # one per framework
        self.assertEqual(sensitivity['arms']['default']['lengths']['interior_gap_slots'], 0)
        self.assertAlmostEqual(sensitivity['edit_difference']['difference'], self.edit('igcraft_t1') - self.edit('igcraft'))
        self.assertEqual(sensitivity['arms']['default']['lengths']['attempts'], 36)

    def test_the_ladder_needs_references_and_reports_the_share_of_learned_gain(self):
        self.assertIsNone(self.result['ladder'])
        world = self.reference_world
        plan = V.build_plan(world.recipe_path, str(world.root / 'references'))
        result = V.compute(plan)
        ladder = result['ladder']['edit']
        floor = V.aggregate(world.runs['empirical'], 'edit_similarity_mean')
        learned = ladder['learned']['mlm']
        retrieval = V.aggregate(world.runs['retrieval'], 'edit_similarity_mean')
        self.assertAlmostEqual(ladder['captured']['retrieval']['mlm'], (retrieval - floor) / (learned - floor))
        self.assertEqual(set(ladder['references']), {'retrieval', 'retrieval_j', 'retrieval_fine'})
        self.assertEqual(len(result['seeds']['contrasts']), 2 * 16)  # six more reference contrasts per endpoint

    def test_report_and_figures_render(self):
        text = V.report(self.result)
        for heading in ('## Training seeds', 'Seed-aware contrasts', '## Same seed, identical config', '## Capacity', '## IgCraft temperature', '## Limitations'):
            self.assertIn(heading, text)
        with tempfile.TemporaryDirectory() as directory:
            V.capacity_figure(self.result, Path(directory) / 'c.png')
            V.seed_figure(self.result, Path(directory) / 's.png')
            for name in ('c.png', 'c.pdf', 's.png', 's.pdf'):
                self.assertGreater((Path(directory) / name).stat().st_size, 1000, name)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.world = World(Path(temp.name))

    def test_execute_seals_its_outputs_and_refuses_later_changes(self):
        plan = V.build_plan(self.world.recipe_path)
        output = self.world.root / 'out'
        V.execute(plan, output)
        complete = json.loads((output / 'complete.json').read_text())
        self.assertEqual(set(complete['artifacts']), set(V.ARTIFACTS))
        self.assertTrue(all(sha256_file(output / n) == d for n, d in complete['artifacts'].items()))
        generation = Path(self.world.runs['ar_s44']['input']['path'])
        generation.write_text(generation.read_text().replace('"complete": true', '"complete": true '))
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            V.validate_plan(plan)

    def test_generations_that_differ_from_what_was_scored_are_refused(self):
        path = Path(self.world.runs['mlm_s44']['input']['path'])
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'differ from what was scored'):
            V.build_plan(self.world.recipe_path)

    def test_unfinished_generations_are_refused(self):
        path = Path(self.world.runs['mlm_s44']['input']['path'])
        payload = json.loads(path.read_text())
        payload['finished_at'] = None
        path.write_text(json.dumps(payload))
        scores = self.world.root / 'seeds/scores/comparison.json'
        data = json.loads(scores.read_text())
        next(r for r in data['runs'] if r['label'] == 'mlm_s44')['input']['sha256'] = sha256_file(path)
        scores.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            V.build_plan(self.world.recipe_path)

    def test_missing_arms_and_mismatched_manifests_are_refused(self):
        scores = self.world.root / 'capacity/scores/comparison.json'
        data = json.loads(scores.read_text())
        kept = [r for r in data['runs'] if r['label'] != 'ar_XL_s43']
        scores.write_text(json.dumps(dict(data, runs=kept)))
        with self.assertRaisesRegex(ValueError, "capacity batch lacks \\['ar_XL_s43'\\]"):
            V.build_plan(self.world.recipe_path)
        scores.write_text(json.dumps(dict(data, manifest=dict(sha256='other'))))
        with self.assertRaisesRegex(ValueError, 'different manifests'):
            V.build_plan(self.world.recipe_path)

    def test_a_batch_that_was_never_run_is_explained(self):
        with self.assertRaisesRegex(ValueError, 'No scored comparison in .*never_run.*must run to completion first'):
            V.build_plan(self.world.recipe_path, str(self.world.root / 'never_run'))
        shutil_free = self.world.root / 'seeds/scores/comparison.json'
        shutil_free.rename(shutil_free.with_suffix('.moved'))  # a batch whose scoring stage did not finish
        with self.assertRaisesRegex(ValueError, 'No scored comparison'):
            V.build_plan(self.world.recipe_path)

    def test_a_changed_main_comparison_fails_its_seal(self):
        main = self.world.root / 'main/capacity.json'
        main.write_text(main.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Base artifact changed'):
            V.build_plan(self.world.recipe_path)


# ====================================================================================================
# from test_benchmark_plotter.py
# ====================================================================================================

def metric(value, examples=2):
    return {"macro_mean": value, "between_example_sd": 0.0, "examples": examples}


def scored_run(label, recovery, diversity):
    aggregate = {
        "whole_span_aar_mean": metric(recovery + 0.1),
        "core_aar_mean": metric(recovery),
        "anchor_aar_mean": metric(recovery + 0.3),
        "diversity_hamming": metric(diversity),
    }
    by_length = {
        name: {"core_aar_mean": metric(recovery, examples=count)}
        for name, count in (("short", 1), ("medium", 1), ("long", 0))
    }
    records = [
        {
            "metrics": {
                "position_recovery": [
                    {"relative_position": 0.0, "recovery": 0.0},
                    {"relative_position": 0.04, "recovery": 1.0},
                    {"relative_position": 1.0, "recovery": recovery},
                ]
            }
        },
        {
            "metrics": {
                "position_recovery": [
                    {"relative_position": 0.0, "recovery": 1.0},
                    {"relative_position": 1.0, "recovery": recovery},
                ]
            }
        },
    ]
    return {
        "label": label,
        "adapter": label,
        "sampling": {"track": "fixed_native_length", "num_samples": 5},
        "aggregate": aggregate,
        "by_length_bucket": by_length,
        "records": records,
    }


def benchmark_plotter_score_payload():
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "manifest": {"benchmark_id": "validation_smoke"},
        "runs": [
            scored_run("cdrseqgen", recovery=0.4, diversity=0.5),
            scored_run("iglm", recovery=0.2, diversity=0.8),
        ],
    }


def length_free_payload():
    payload = benchmark_plotter_score_payload()
    payload["manifest"]["benchmark_id"] = "validation_length_free"
    for index, run in enumerate(payload["runs"]):
        run["sampling"]["track"] = "length_free"
        run["aggregate"].update({
            "length_match_rate": metric(0.6 - index * 0.1),
            "length_absolute_error_mean": metric(0.8 + index * 0.2),
            "edit_similarity_mean": metric(0.9 - index * 0.1),
            "valid_generation_rate": metric(1.0),
            "micro_accounting": {"examples": 2},
        })
        for summary in run["by_length_bucket"].values():
            summary["length_absolute_error_mean"] = metric(1.0)
    # The length-free dashboard must remain valid even if one model has no exact-length draws.
    payload["runs"][1]["aggregate"].update({
        "whole_span_aar_mean": metric(None, 0),
        "core_aar_mean": metric(None, 0),
        "anchor_aar_mean": metric(None, 0),
    })
    return payload


class BenchmarkPlotterTests(unittest.TestCase):
    def test_position_curve_macro_averages_within_each_framework_first(self):
        _, recovery, counts = normalized_position_curve(benchmark_plotter_score_payload()["runs"][0], bins=11)

        self.assertAlmostEqual(recovery[0], 0.75)
        self.assertEqual(counts[0], 2)
        self.assertAlmostEqual(recovery[-1], 0.4)

    def test_rejects_wrong_schema(self):
        payload = benchmark_plotter_score_payload()
        payload["schema_version"] = "wrong"

        with self.assertRaisesRegex(ValueError, "expected score schema"):
            validate_scores(payload)

    def test_writes_a_nonempty_development_figure(self):
        payload = benchmark_plotter_score_payload()
        self.assertTrue(is_development_artifact(payload))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"
            written = save_comparison_figure(payload, output)

            self.assertEqual(written, output)
            self.assertGreater(output.stat().st_size, 1000)

    def test_length_free_track_uses_length_dashboard_without_requiring_aar(self):
        payload = length_free_payload()

        self.assertEqual(benchmark_track(payload), "length_free")
        validate_scores(payload)
        figure = build_figure(payload)
        try:
            self.assertEqual(
                figure._suptitle.get_text(), "CDR-H3 length-free benchmark comparison"
            )
            self.assertIn("Length-free", figure.axes[0].get_title())
            self.assertIn("Absolute length error", figure.axes[1].get_title())
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_rejects_mixed_tracks(self):
        payload = benchmark_plotter_score_payload()
        payload["runs"][1]["sampling"]["track"] = "length_free"

        with self.assertRaisesRegex(ValueError, "do not share one benchmark track"):
            validate_scores(payload)


# ====================================================================================================
# from test_benchmark_track_contract.py
# ====================================================================================================

class BenchmarkTrackContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "benchmark_id": "b",
            "task": {"track": "length_free"},
            "records": [{"example_id": "x"}],
        }
        self.payload = {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "benchmark_id": "b",
            "manifest": {"sha256": "hash"},
            "sampling": {"track": "length_free"},
            "records": [{"example_id": "x"}],
        }

    def test_matching_track_passes(self):
        ids = validate_payload(
            self.payload,
            label="model",
            manifest=self.manifest,
            manifest_hash="hash",
            allow_partial=False,
        )
        self.assertEqual(ids, ["x"])

    def test_examples_that_generated_nothing_are_refused(self):
        """attempts == 0 is self-consistent, so only this check separates a half-dead run from
        a complete one — every downstream metric silently skips those examples."""
        manifest = {
            "benchmark_id": "b",
            "task": {"track": "length_free"},
            "records": [{"example_id": name} for name in ("x", "y", "z", "w")],
        }
        records = [
            {"example_id": "x", "attempts": 0},
            {"example_id": "y", "attempts": 0},
            {"example_id": "z", "attempts": 20},
            {"example_id": "w", "attempts": 20},
        ]
        payload = {**self.payload, "records": records}
        with self.assertRaisesRegex(ValueError, "zero generation attempts"):
            validate_payload(payload, label="model", manifest=manifest,
                             manifest_hash="hash", allow_partial=False)

        ids = validate_payload(payload, label="model", manifest=manifest,
                               manifest_hash="hash", allow_partial=False,
                               allow_dead_records=True)
        self.assertEqual(ids, ["x", "y", "z", "w"])

    def test_mismatched_track_is_rejected(self):
        payload = {**self.payload, "sampling": {"track": "fixed_native_length"}}
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            validate_payload(
                payload,
                label="model",
                manifest=self.manifest,
                manifest_hash="hash",
                allow_partial=False,
            )


if __name__ == "__main__":
    unittest.main()
