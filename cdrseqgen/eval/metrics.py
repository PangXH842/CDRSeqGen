"""Pure sequence-level generative metrics for sampled CDR loops.

  * amino_acid_recovery     -- calculate sequence recovery between `pred` and `native` seqs,
                                for each individual sample
  * mean_pairwise_diversity -- calculate mean Hamming distance between each pair of samples
  * unique_fraction         -- fraction of distinct sequences among all samples given
  * per_position_entropy    -- calculate mean Shannon entropy (bits) across loop positions
  * calculate               -- length error/edit similarity plus length-matched AAR when native exists
                                and aggregate all sequence metrics into one dict
"""
import math
from collections import Counter
from itertools import combinations


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


def _hamming_fraction(a, b):
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

    dists = [_hamming_fraction(a, b) for a, b in combinations(samples, 2)]
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

    total = 0.0
    for i in range(length):
        counts = Counter(s[i] for s in samples)
        n = sum(counts.values())
        total += -sum((c / n) * math.log2(c / n) for c in counts.values())
    return total / length


def calculate(samples, native=None):
    """Aggregate the generative-metric block for one test case (native optional)"""

    def edit_similarity(sample, reference):
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
