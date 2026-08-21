"""Pure sequence-level generative metrics for sampled CDR loops.

  * amino_acid_recovery     -- calculate sequence recovery between `pred` and `native` seqs,
                                for each individual sample
  * mean_pairwise_diversity -- calculate mean Hamming distance between each pair of samples
  * unique_fraction         -- fraction of distinct sequences among all samples given
  * per_position_entropy    -- calculate mean Shannon entropy (bits) across loop positions
  * edit_similarity         -- 1 - normalised Levenshtein, for candidates of any length
  * spearman                -- rank correlation, ties averaged; None when undefined
  * confidence_metrics      -- does a per-candidate score rank candidates by recovery?
  * calculate               -- length error/edit similarity plus length-matched AAR when native exists
                                and aggregate all sequence metrics into one dict
"""
import math
import statistics
from collections import Counter
from itertools import combinations


# Reporting buckets for loop length. Shared by native evaluation and the benchmark scorer so a
# "long" loop means the same thing on both sides of a matched comparison.
LENGTH_BUCKETS = (("short", 0, 11), ("medium", 12, 17), ("long", 18, 10**6))


def amino_acid_recovery(pred, native):
    """Per-residue identity of `pred` to `native` in [0, 1]"""
    if not native:
        return None
    pred = pred.upper().strip()
    native = native.upper().strip()
    if not native:
        return None

    overlap = min(len(pred), len(native))
    correct = sum(1 for i in range(overlap) if pred[i] == native[i])
    return correct / len(native)


def hamming_fraction(a, b):
    """Normalised Hamming distance; a length mismatch counts the extra residues as differences"""
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0

    overlap = min(len(a), len(b))
    diff = sum(1 for i in range(overlap) if a[i] != b[i]) + (longest - overlap)
    return diff / longest


def mean_pairwise_diversity(samples):
    """Mean normalised Hamming distance over all unordered pairs (0 = identical, 1 = all differ)"""
    if len(samples) < 2:
        return 0.0

    dists = [hamming_fraction(a, b) for a, b in combinations(samples, 2)]
    return sum(dists) / len(dists)


def unique_fraction(samples):
    """Fraction of the draws that are distinct sequences (1.0 = every sample unique)"""
    return len(set(samples)) / len(samples) if samples else 0.0


def per_position_entropy(samples):
    """Mean Shannon entropy (bits) across loop positions; None if samples aren't equal length"""
    if not samples:
        return None
    length = len(samples[0])
    if length == 0 or any(len(s) != length for s in samples):
        return None

    entropies = []
    for i in range(length):
        counts = Counter(s[i] for s in samples)
        n = sum(counts.values())
        entropies.append(-sum((c / n) * math.log2(c / n) for c in counts.values()))
    # statistics.mean sums exactly; a running total drifted from the benchmark scorer's copy of
    # this function in the last ULP, which is precisely the kind of "same metric, two answers"
    # this module exists to prevent.
    return statistics.mean(entropies)


def edit_similarity(sample, reference):
    """One minus normalised Levenshtein distance; defined for unequal lengths, so it is the
    length-free counterpart to positional recovery."""
    longest = max(len(sample), len(reference))
    if longest == 0:
        return 1.0
    previous = list(range(len(reference) + 1))
    for i, left in enumerate(sample, start=1):
        current = [i]
        for j, right in enumerate(reference, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return 1.0 - previous[-1] / longest


def _average_ranks(values):
    """Ranks with ties averaged, so tied scores cannot fabricate an ordering."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Rank correlation in [-1, 1]; None when undefined (n < 3, or either side constant)."""
    if len(xs) != len(ys):
        raise ValueError("spearman needs equal-length sequences")
    if len(xs) < 3:
        return None
    rx, ry = _average_ranks(list(xs)), _average_ranks(list(ys))
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:      # a constant score ranks nothing
        return None
    return num / (dx * dy)


def confidence_metrics(recoveries, scores, *, lower_is_better=True, prefix="conf"):
    """Does a per-candidate confidence score rank candidates by recovery?

    Answers the question a designer actually faces: given N candidates for one framework, does
    picking the most confident one beat picking at random? ``baseline`` is the mean over the same
    pairs, so the top-k lift is measured on an identical population rather than against aar_mean,
    whose population may differ.

    Pairs with a missing value are dropped — a judge that skipped a candidate reduces n rather
    than biasing the ranking.
    """
    pairs = [(r, s) for r, s in zip(recoveries, scores) if r is not None and s is not None]
    out = {f"{prefix}_n": len(pairs)}
    if len(pairs) < 3:
        return out
    rec = [p[0] for p in pairs]
    # Orient so that larger always means "more confident"; perplexity is lower-is-better.
    oriented = [-p[1] for p in pairs] if lower_is_better else [p[1] for p in pairs]
    out[f"{prefix}_spearman"] = spearman(oriented, rec)
    order = sorted(range(len(pairs)), key=lambda i: -oriented[i])
    out[f"{prefix}_baseline_recovery"] = sum(rec) / len(rec)
    out[f"{prefix}_top1_recovery"] = rec[order[0]]
    k = min(3, len(pairs))
    out[f"{prefix}_top3_recovery"] = sum(rec[i] for i in order[:k]) / k
    return out


def calculate(samples, native=None):
    """Aggregate the generative-metric block for one test case (native optional)"""

    out = {
        "n_samples": len(samples),
        "n_unique": len(set(samples)),
        "unique_fraction": unique_fraction(samples),
        "diversity_hamming": mean_pairwise_diversity(samples),
        "per_position_entropy_bits": per_position_entropy(samples),
    }

    lengths = [len(sample) for sample in samples]
    if lengths:
        mean_length = sum(lengths) / len(lengths)
        out["generated_length_mean"] = mean_length
        out["generated_length_min"] = min(lengths)
        out["generated_length_max"] = max(lengths)

    if native:
        matching = [sample for sample in samples if len(sample) == len(native)]
        out["length_match_rate"] = len(matching) / len(samples) if samples else None
        if samples:
            out["length_absolute_error_mean"] = sum(
                abs(len(sample) - len(native)) for sample in samples
            ) / len(samples)
            out["edit_similarity_mean"] = sum(
                edit_similarity(sample, native) for sample in samples
            ) / len(samples)
        if matching:
            aars = [amino_acid_recovery(sample, native) for sample in matching]
            mean_aar = sum(aars) / len(aars)
            out["aar_length_matched_mean"] = mean_aar
            out["aar_mean"] = mean_aar
            out["aar_max"] = max(aars)
            out["aar_min"] = min(aars)
            # Positional AAR is conditioned on correct length; unequal lengths use edit similarity.
            out["novelty_mean"] = 1.0 - mean_aar

    return out
