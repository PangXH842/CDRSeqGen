"""Pure sequence-level generative metrics for sampled CDR loops.

  * amino_acid_recovery     -- calculate sequence recovery between `pred` and `native` seqs,
                                for each individual sample
  * mean_pairwise_diversity -- calculate mean Hamming distance between each pair of samples
  * unique_fraction         -- fraction of distinct sequences among all samples given
  * per_position_entropy    -- calculate mean Shannon entropy (bits) across loop positions
  * edit_similarity         -- 1 - normalised Levenshtein, for candidates of any length
  * blosum_positional       -- mean BLOSUM62 per position, equal-length pairs only
  * blosum_aligned          -- BLOSUM62 global-alignment score, ANY pair of lengths
  * blosum_provenance       -- matrix/gap parameters a BLOSUM number is meaningless without
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


def core_anchor_indices(length, anchor_trim):
    """Positions of the hypervariable core and the V/J-determined anchors, for a loop of `length`.

    A loop of <= 2*anchor_trim residues has no core and is entirely anchor, matching
    ``recovery._core_anchor_masks``. Kept here rather than in either caller so the training
    metrics, the generative eval, and the benchmark scorer cannot drift on what "core" means.
    """
    core = list(range(anchor_trim, length - anchor_trim)) if length > 2 * anchor_trim else []
    core_set = set(core)
    return core, [i for i in range(length) if i not in core_set]


def recovery_at_indices(sample, native, indices):
    """Per-residue identity over `indices` only; None when the subset is empty."""
    if not indices:
        return None
    return sum(sample[i] == native[i] for i in indices) / len(indices)


def position_recovery(samples, native):
    """Per-position recovery across the length-matched samples for one framework.

    Returns one entry per native position: its index, its position normalised to [0, 1] so loops of
    different length can share an axis, and the fraction of length-matched samples that reproduce
    the native residue there (None when nothing matched the length).

    Positional recovery is only defined against a same-length sample, which is why wrong-length
    draws are dropped here rather than counted as misses -- the same rule `recovery_at_indices`
    follows. Shared with the benchmark scorer so a position curve drawn from a training run and one
    drawn from a benchmark score mean the same thing.
    """
    length = len(native)
    matched = [sample for sample in samples if len(sample) == length]
    return [
        {
            "position": position,
            "relative_position": (position / (length - 1)) if length > 1 else 0.5,
            "recovery": (
                sum(sample[position] == native[position] for sample in matched) / len(matched)
                if matched else None
            ),
        }
        for position in range(length)
    ]


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


# --- BLOSUM62 -----------------------------------------------------------------------------------
# Exact-match recovery scores I->V the same as I->P: zero. A substitution matrix separates a
# conservative swap from a chemically disruptive one, which is the distinction that matters when
# asking whether a designed loop is *close* to native rather than identical to it.
#
# The table is inlined rather than pulled from biopython: it is 20x20 of fixed integers, it never
# changes, and one matrix is not worth a dependency. Parsed from the canonical NCBI layout below
# and validated for symmetry at import, so a typo cannot survive silently.
#
# Caveat this metric cannot escape, and which every caller should carry: BLOSUM is a generic
# evolutionary substitution matrix. It says nothing about antigen binding or structure, and it
# rewards safe generic substitutions. It belongs BESIDE recovery, judge perplexity, and the length
# metrics -- never instead of them.
BLOSUM62_MATRIX_NAME = "BLOSUM62"
_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def _build_blosum62():
    rows = [line.split() for line in _BLOSUM62_ROWS.strip().splitlines()]
    if len(rows) != len(_BLOSUM62_ORDER):
        raise ValueError("BLOSUM62 table is not square with its residue order")
    table = {}
    for a, row in zip(_BLOSUM62_ORDER, rows):
        if len(row) != len(_BLOSUM62_ORDER):
            raise ValueError(f"BLOSUM62 row {a} has {len(row)} entries")
        for b, value in zip(_BLOSUM62_ORDER, row):
            table[(a, b)] = int(value)
    for (a, b), value in table.items():
        if table[(b, a)] != value:          # a transposition typo shows up here, not in results
            raise ValueError(f"BLOSUM62 is asymmetric at {a}/{b}")
    return table


BLOSUM62 = _build_blosum62()
# Standard NCBI BLASTP defaults for BLOSUM62. Declared, and recorded in artifacts, because an
# alignment score is meaningless without the penalties that produced it.
BLOSUM62_GAP_OPEN = -11
BLOSUM62_GAP_EXTEND = -1
# Unknown/non-standard residues score as a mismatch rather than crashing or being skipped: skipping
# would quietly shorten the denominator and flatter a sequence containing them.
BLOSUM62_UNKNOWN = -4


def blosum_pair(a, b):
    """BLOSUM62 score for one residue pair; BLOSUM62_UNKNOWN for anything off the 20-letter set."""
    return BLOSUM62.get((a, b), BLOSUM62_UNKNOWN)


def blosum_positional(sample, native, indices=None):
    """Mean BLOSUM62 score per position, for an equal-length pair.

    Returns None when the lengths differ (no positional correspondence exists) or the index subset
    is empty -- the same rule `recovery_at_indices` follows, so core/anchor BLOSUM is defined on
    exactly the candidates core/anchor recovery is.
    """
    if not native or len(sample) != len(native):
        return None
    positions = range(len(native)) if indices is None else indices
    positions = list(positions)
    if not positions:
        return None
    return sum(blosum_pair(sample[i], native[i]) for i in positions) / len(positions)


def blosum_self_score(native, indices=None):
    """Mean BLOSUM62 score of `native` against itself -- the maximum `blosum_positional` can reach."""
    return blosum_positional(native, native, indices)


def blosum_identity_ratio(sample, native, indices=None):
    """`blosum_positional` divided by the native's self-score.

    1.0 for an identical loop, lower for substitutions, negative when they are worse than the
    matrix's expectation. The raw mean is always reported alongside this: a normalised score
    without its formula and its numerator is not auditable.
    """
    score = blosum_positional(sample, native, indices)
    reference = blosum_self_score(native, indices)
    if score is None or not reference:
        return None
    return score / reference


def blosum_aligned(sample, native, gap_open=BLOSUM62_GAP_OPEN, gap_extend=BLOSUM62_GAP_EXTEND):
    """Needleman-Wunsch global alignment score under BLOSUM62 with affine gaps (Gotoh).

    Unlike `blosum_positional` this is defined for ANY pair of lengths, which is the point: a
    causal model matches the native length on ~7% of H3 draws, so a length-matched-only metric
    describes a small and self-selected subset. Reported over every attempt instead.
    """
    if not native or not sample:
        return None
    n, m = len(sample), len(native)
    neg = float("-inf")
    # M: ends aligned; X: gap in native; Y: gap in sample.
    prev_m = [neg] * (m + 1)
    prev_x = [neg] * (m + 1)
    prev_y = [neg] * (m + 1)
    prev_m[0] = 0.0
    for j in range(1, m + 1):
        prev_y[j] = gap_open + (j - 1) * gap_extend
    for i in range(1, n + 1):
        cur_m = [neg] * (m + 1)
        cur_x = [neg] * (m + 1)
        cur_y = [neg] * (m + 1)
        cur_x[0] = gap_open + (i - 1) * gap_extend
        for j in range(1, m + 1):
            score = blosum_pair(sample[i - 1], native[j - 1])
            cur_m[j] = score + max(prev_m[j - 1], prev_x[j - 1], prev_y[j - 1])
            cur_x[j] = max(prev_m[j] + gap_open, prev_x[j] + gap_extend)
            cur_y[j] = max(cur_m[j - 1] + gap_open, cur_y[j - 1] + gap_extend)
        prev_m, prev_x, prev_y = cur_m, cur_x, cur_y
    return float(max(prev_m[m], prev_x[m], prev_y[m]))


def blosum_aligned_ratio(sample, native, **kwargs):
    """`blosum_aligned` over the native's self-alignment score, so 1.0 is an identical loop."""
    score = blosum_aligned(sample, native, **kwargs)
    reference = blosum_aligned(native, native, **kwargs)
    if score is None or not reference:
        return None
    return score / reference


def blosum_provenance():
    """The parameters a BLOSUM number is meaningless without, for the result artifact."""
    return {
        "matrix": BLOSUM62_MATRIX_NAME,
        "gap_open": BLOSUM62_GAP_OPEN,
        "gap_extend": BLOSUM62_GAP_EXTEND,
        "unknown_residue_score": BLOSUM62_UNKNOWN,
        "alignment": "needleman-wunsch global, affine gaps (Gotoh)",
        "positional_normalisation": "mean_per_position_score / native_self_score",
        "aligned_normalisation": "alignment_score / native_self_alignment_score",
    }


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
        # Two BLOSUM views, deliberately separate. The aligned one covers EVERY attempt, so it
        # does not inherit the self-selection that makes length-matched recovery describe ~7% of
        # causal H3 draws; the positional one is the like-for-like partner to recovery@L.
        aligned = [blosum_aligned_ratio(sample, native) for sample in samples]
        aligned = [value for value in aligned if value is not None]
        if aligned:
            out["blosum62_aligned_ratio_mean"] = sum(aligned) / len(aligned)
            raw = [blosum_aligned(sample, native) for sample in samples]
            raw = [value for value in raw if value is not None]
            out["blosum62_aligned_mean"] = sum(raw) / len(raw)
        if matching:
            positional = [blosum_positional(sample, native) for sample in matching]
            positional = [value for value in positional if value is not None]
            if positional:
                out["blosum62_mean_at_length"] = sum(positional) / len(positional)
                ratios = [blosum_identity_ratio(sample, native) for sample in matching]
                ratios = [value for value in ratios if value is not None]
                out["blosum62_ratio_at_length_mean"] = sum(ratios) / len(ratios)
            aars = [amino_acid_recovery(sample, native) for sample in matching]
            mean_aar = sum(aars) / len(aars)
            out["aar_length_matched_mean"] = mean_aar
            out["aar_mean"] = mean_aar
            out["aar_max"] = max(aars)
            out["aar_min"] = min(aars)
            # Positional AAR is conditioned on correct length; unequal lengths use edit similarity.
            out["novelty_mean"] = 1.0 - mean_aar

    return out
