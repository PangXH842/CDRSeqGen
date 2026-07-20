"""Pure sequence-level generative metrics for sampled CDR loops.

These take the *strings* a sampler produced (see sampling.py) and answer what aar cannot:
are the samples diverse, and how far do they stray from the native loop (novelty) rather than
just copying it. Model-facing scoring (pseudo-perplexity) lives in sampling.py; everything here
is a pure function of sequences, so it needs no model, no torch, and is trivially testable.

Within one test case every sample has the same length L (the gap length is fixed), so diversity
and per-position entropy assume equal length; get_aar still tolerates a length mismatch
against the native so it mirrors predict.py's original alignment semantics exactly.
"""
import math
from collections import Counter
from itertools import combinations


def get_aar(pred, native):
    """Per-residue identity of `pred` to `native` in [0, 1] (denominator = len(native)).

    Mirrors predict.py's original alignment: compare over the overlapping length and divide by the
    native length, so producing the wrong length is (correctly) penalised. Returns None if there
    is no native sequence to compare against.
    """
    if not native:
        return None
    pred = pred.upper().strip()
    native = native.upper().strip()
    if not native:
        return None
    overlap = min(len(pred), len(native))
    correct = sum(1 for i in range(overlap) if pred[i] == native[i])
    return correct / len(native)


def _hamming_fraction(a, b):
    """Normalised Hamming distance; a length mismatch counts the extra residues as differences."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    overlap = min(len(a), len(b))
    diff = sum(1 for i in range(overlap) if a[i] != b[i]) + (longest - overlap)
    return diff / longest


def mean_pairwise_diversity(samples):
    """Mean normalised Hamming distance over all unordered pairs (0 = identical, 1 = all differ)."""
    if len(samples) < 2:
        return 0.0
    dists = [_hamming_fraction(a, b) for a, b in combinations(samples, 2)]
    return sum(dists) / len(dists)


def unique_fraction(samples):
    """Fraction of the draws that are distinct sequences (1.0 = every sample unique)."""
    return len(set(samples)) / len(samples) if samples else 0.0


def per_position_entropy(samples):
    """Mean Shannon entropy (bits) across loop positions; None if samples aren't equal length."""
    if not samples:
        return None
    length = len(samples[0])
    if length == 0 or any(len(s) != length for s in samples):
        return None
    total = 0.0
    for i in range(length):
        counts = Counter(s[i] for s in samples)
        n = sum(counts.values())
        total += -sum((c / n) * math.log2(c / n) for c in counts.values())
    return total / length


def summarize(samples, native=None):
    """Aggregate the generative-metric block for one test case (native optional)."""
    out = {
        "n_samples": len(samples),
        "n_unique": len(set(samples)),
        "unique_fraction": unique_fraction(samples),
        "diversity_hamming": mean_pairwise_diversity(samples),
        "per_position_entropy_bits": per_position_entropy(samples),
    }

    if native:
        aars = [get_aar(s, native) for s in samples]
        aars = [r for r in aars if r is not None]
        if aars:
            mean_aar = sum(aars) / len(aars)
            out["aar_mean"] = mean_aar
            out["aar_max"] = max(aars)
            out["aar_min"] = min(aars)
            out["novelty_mean"] = 1.0 - mean_aar  # mean distance from the native loop

    return out
