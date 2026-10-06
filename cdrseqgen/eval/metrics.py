"""Sequence metrics shared by generated-CDR evaluation and benchmark scoring.

Functions: 
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
  * confidence_metrics      -- rank correlation and top-k recovery from candidate scores
  * calculate               -- aggregate all sequence metrics into one dict

Notes:
  * The `native` in the input args refer to the ground-truth loop string. Required for recovery,
    edit similarity, and BLOSUM metrics; optional for diversity, uniqueness, and entropy.
  * Sequence strings should already be uppercase and stripped; only amino_acid_recovery
    normalizes its inputs.
  * BLOSUM62 distinguishes residue substitutions that AAR counts as mismatches:
      substitution: I -> V  I -> P
      AAR:               0       0
      BLOSUM62:          3      -3
    These are evolutionary substitution scores, not measurements of antigen binding
    or structure. Interpret them alongside recovery, judge perplexity and length metrics.
"""
import math
import statistics
from collections import Counter
from itertools import combinations

# sort loop lengths for h3
LENGTH_BUCKETS = (("short", 0, 11), ("medium", 12, 17), ("long", 18, 10**6))


def amino_acid_recovery(pred, native):
    """Return positional identity divided by the native sequence length.

    Example:
      pred  : A-C-D-D-B-E-F
      native: A-B-C-D-E-F
      match : 1-0-0-1-0-0
      AAR   : sum(match) / len(native) = 2/6
    """
    native = (native or "").upper().strip()
    if not native:
        return None
    pred = pred.upper().strip()

    overlap = min(len(pred), len(native))
    correct = sum(1 for i in range(overlap) if pred[i] == native[i])
    aar = correct / len(native)
    return aar


def core_anchor_indices(length, anchor_trim):
    """Apply core/anchor index labels for a CDR loop. Returns core [list], anchors [list].

    Example: length=8, anchor_trim=2
      position: 0 1 2 3 4 5 6 7
      region:   A A C C C C A A     (A=anchor, C=core)
      core:     [2, 3, 4, 5]
      anchors:  [0, 1, 6, 7]

    Notes:
      * Cases where all are anchors (and core list empty) is possible, 
        eg. length <= 2 * anchor_trim
    """
    core_idxs = list(range(anchor_trim, length - anchor_trim)) if length > 2 * anchor_trim else []
    core_set = set(core_idxs)
    anchor_idxs = [i for i in range(length) if i not in core_set]
    return core_idxs, anchor_idxs


def recovery_at_indices(sample, native, indices):
    """Return positional recovery over the supplied indices.

    Example: indices=[2, 3]
      position: 0 1 2 3 4 5
      sample:   A C D Q F G
      native:   A C D E F G
      selected:     1 0          (D==D, Q!=E)
      recovery: (1 + 0) / 2 = 0.5

    Notes:
      * Indices can be any list of valid positions, they are not necessarily consecutive 
        or sorted.
    """
    if not indices:
        return None
    return sum(sample[i] == native[i] for i in indices) / len(indices)


def position_recovery(samples, native):
    """Return one recovery dict per native position, using length-matched draws.

    Example: 
      position:          0   1   2   3   4
      relative_position: 0 0.25 0.5 0.75 1
      native:            A   B   C   D   E
      sample 1:          A   P   C   D   E
      sample 2:          A   Q   D   C   E
      sample 3:          A   C   E          (excluded)
      recovery:          1   0  0.5 0.5  1      

    Notes:
      * Each dict has position, relative_position and recovery.
      * relative_position = position / (native length - 1);
        a one-residue native uses 0.5, and an empty native returns [].
      * With no length-matched draws, each recovery is None.
      * Duplicate draws count separately; characters are compared as supplied.
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
    """Return (positional mismatches + extra residues) / longer sequence length.

    Example:
      a:        A C D
      b:        A Q D E
      distance: (1 mismatch + 1 extra residue) / 4 = 0.5

    Notes:
      * Compare the overlapping prefix directly; no alignment is performed.
      * Result is in [0, 1]. Two empty strings score 0.0;
        exactly one empty string scores 1.0.
    """
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0

    overlap = min(len(a), len(b))
    diff = sum(1 for i in range(overlap) if a[i] != b[i]) + (longest - overlap)
    return diff / longest


def mean_pairwise_diversity(samples):
    """Return mean hamming_fraction across all unordered pairs of draws.

    Example: samples=["AC", "AC", "AQ"]
      pair (0, 1): AC vs AC -> 0.0
      pair (0, 2): AC vs AQ -> 0.5
      pair (1, 2): AC vs AQ -> 0.5
      diversity:   (0.0 + 0.5 + 0.5) / 3 = 1/3

    Notes:
      * Duplicate sequences remain separate draws, including zero-distance pairs.
      * Result is in [0, 1]; fewer than two draws return 0.0.
    """
    if len(samples) < 2:
        return 0.0

    dists = [hamming_fraction(a, b) for a, b in combinations(samples, 2)]
    return sum(dists) / len(dists)


def unique_fraction(samples):
    """Return distinct sequence count / total draw count.
    Eg. ["AC", "AC", "AQ"] -> {"AC", "AQ"}, fraction: 2 / 3
    """
    return len(set(samples)) / len(samples) if samples else 0.0


def per_position_entropy(samples):
    """Return mean Shannon entropy in bits across sequence positions.

    Example: samples=["AC", "AT"]
      position 0: A=1.0          -> entropy=0 bits
      position 1: C=0.5, T=0.5   -> entropy=1 bit
      mean entropy: (0 + 1) / 2 = 0.5 bits

    Notes:
      * At each position, entropy = -sum(p * log2(p)) over character frequencies.
      * Duplicate draws affect frequencies; identical non-empty draws score 0.0.
      * Return None for no draws, unequal lengths, or zero-length draws.
      * Inputs are compared as supplied, without alignment or normalization.
    """
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
    """Return 1 - edit distance / longer sequence length, in [0, 1].

    Example:
      sample:     "ACDF"
      reference:  "ACDEF"
      edit:       insert E before F -> distance=1
      similarity: 1 - 1/5 = 0.8

    Notes:
      * Levenshtein distance: insertion, deletion and substitution each cost 1.
      * Characters are compared as supplied; shifts are handled through edits.
      * Two empty strings score 1.0; exactly one empty string scores 0.0.
    """
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
    """Return ascending ranks in input order, averaging tied ranks.

    Example:
      values: [30,  10,  10,  20]
      ranks:  [4.0, 1.5, 1.5, 3.0]
      tie:    the two 10s share ranks 1 and 2 -> (1 + 2) / 2 = 1.5

    Notes:
      * Ranks start at 1; the smallest value receives the lowest rank.
      * Missing or non-finite values must be handled by the caller.
    """
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
    """Return Spearman correlation: Pearson correlation of the two rank lists.

    Example:
      xs:       [1,  2,  3]     -> ranks [1, 2, 3]
      ys:       [10, 20, 30]    -> ranks [1, 2, 3]
      rho:      1.0            (same ordering)
      reversed ys: [30, 20, 10] -> rho=-1.0

    Notes:
      * Ties receive average ranks; a defined result is in [-1, 1].
      * Unequal input lengths raise ValueError.
      * Return None for fewer than three pairs (this project's reporting minimum)
        or when either input is constant.
      * Missing or non-finite values must be handled by the caller.
    """
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
    """Return candidate count, rank correlation, baseline and top-k recovery.

    Example: lower_is_better=True, prefix="conf"
      candidate:   0    1    2    3
      recovery: 0.25 0.50 0.75 1.00
      score:       4    3    2    1     (e.g. perplexity)
      confidence order: [3, 2, 1, 0]
      conf_n:                 4
      conf_spearman:          1.0
      conf_baseline_recovery: (0.25 + 0.50 + 0.75 + 1.00) / 4 = 0.625
      conf_top1_recovery:     1.0
      conf_top3_recovery:     (1.00 + 0.75 + 0.50) / 3 = 0.75

    Notes:
      * Inputs must be equally long and in candidate order; unequal lengths raise ValueError.
      * Exclude pairs containing None, NaN or infinity before counting or ranking.
      * With fewer than three pairs, return only {prefix}_n.
      * Baseline and top-k use the same eligible pairs; top-k values are recoveries.
      * Use lower_is_better=False when larger scores mean greater confidence.
      * Top-k ties keep input order; Spearman averages tied ranks and is None
        when either input has constant ranks.
    """
    if len(recoveries) != len(scores):
        raise ValueError("confidence_metrics needs equal-length recoveries and scores")
    pairs = [
        (r, s) for r, s in zip(recoveries, scores)
        if r is not None and s is not None and math.isfinite(r) and math.isfinite(s)
    ]
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
# Rows and columns follow _BLOSUM62_ORDER.
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
    """Parse the inlined BLOSUM62 rows into a residue-pair score dict.

    Example:
      residue order: A  R ...
      first row:     4 -1 ...
      table:         {('A', 'A'): 4, ('A', 'R'): -1, ...}

    Notes:
      * The 20x20 table is inlined to avoid a matrix-loading dependency.
      * Called at import; incorrect dimensions, non-integer scores or asymmetry
        raise ValueError.
      * Symmetry requires table[(a, b)] == table[(b, a)].
    """
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
        if table[(b, a)] != value:
            raise ValueError(f"BLOSUM62 is asymmetric at {a}/{b}")
    return table


BLOSUM62 = _build_blosum62()
# Alignment defaults recorded by blosum_provenance().
BLOSUM62_GAP_OPEN = -11
BLOSUM62_GAP_EXTEND = -1
# Fallback for pairs outside the stored alphabet.
BLOSUM62_UNKNOWN = -4


def blosum_pair(a, b):
    """Return the BLOSUM62 substitution score for two uppercase residues.

    Example:
      I vs V ->  3
      I vs P -> -3
      B vs B -> -4     (B is outside the stored 20-residue alphabet)

    Notes:
      * Any pair containing an unknown/non-canonical residue scores -4,
        including identical unknown characters.
      * Unknown residues are scored rather than skipped, preserving positional
        averaging denominators.
      * No case normalization is performed; lowercase residues also use -4.
    """
    return BLOSUM62.get((a, b), BLOSUM62_UNKNOWN)


def blosum_positional(sample, native, indices=None):
    """Return mean BLOSUM62 score over selected positions of equal-length strings.

    Example:
      sample: I V
      native: I I
      scores: 4 3
      mean:   (4 + 3) / 2 = 3.5
      with indices=[1]: score=3.0

    Notes:
      * indices=None scores all positions; explicit indices are zero-based.
      * Return None for an absent/empty native, unequal lengths or an empty subset.
      * Out-of-range positions raise IndexError.
      * No alignment or case normalization is performed; scores can be negative.
    """
    if not native or len(sample) != len(native):
        return None
    positions = range(len(native)) if indices is None else indices
    positions = list(positions)
    if not positions:
        return None
    return sum(blosum_pair(sample[i], native[i]) for i in positions) / len(positions)


def blosum_self_score(native, indices=None):
    """Return the native's mean BLOSUM62 score against itself.

    Example: native="IW"
      pairs:  I vs I, W vs W
      scores:      4,     11
      mean:   (4 + 11) / 2 = 7.5

    Notes:
      * Calls blosum_positional(native, native, indices), with the same None rules.
      * Used as the denominator for positional ratios; canonical residues give
        a positive score that depends on the native's composition.
    """
    return blosum_positional(native, native, indices)


def blosum_identity_ratio(sample, native, indices=None):
    """Return positional BLOSUM mean / native positional self-score.

    Example: sample="IV", native="II"
      sample vs native: (4 + 3) / 2 = 3.5
      native vs native: (4 + 4) / 2 = 4.0
      ratio:            3.5 / 4.0 = 0.875

    Notes:
      * Both scores use the same selected indices.
      * Return None if the comparison is unavailable or the denominator is zero.
      * Identical strings score 1.0 when the denominator is nonzero.
      * Ratios can be negative; they are not probabilities.
    """
    score = blosum_positional(sample, native, indices)
    reference = blosum_self_score(native, indices)
    if score is None or not reference:
        return None
    return score / reference


def blosum_aligned(sample, native, gap_open=BLOSUM62_GAP_OPEN, gap_extend=BLOSUM62_GAP_EXTEND):
    """Return a global BLOSUM62 alignment score with affine gap penalties.

    Example: sample="AC", native="ADC", default gap scores
      sample: A - C
      native: A D C
      score:  4 - 11 + 9 = 2.0

    Gap cost for k consecutive gaps:
      gap_open + (k - 1) * gap_extend
      defaults: one gap=-11, two gaps=-12, three gaps=-13

    Notes:
      * Return the best score under this recurrence, without the aligned strings.
      * States: M=residue pair, X=gap in native, Y=gap in sample.
        Direct X <-> Y transitions are excluded (no adjacent opposing gaps).
      * Terminal gaps use the same penalties; unequal input lengths are supported.
      * Absent/empty input returns None. Inputs are not case-normalized;
        unknown residue pairs score -4, and alignment scores can be negative.
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
    """Return global alignment score / native self-alignment score.

    Example: sample="AC", native="ADC", default gap scores
      sample vs native: 4 - 11 + 9 = 2.0
      native vs native: 4 + 6 + 9  = 19.0
      ratio:            2 / 19

    Notes:
      * Gap-score kwargs are forwarded to both blosum_aligned calls.
      * Return None if the comparison is unavailable or the denominator is zero.
      * Unequal lengths are supported; identical strings score 1.0 when defined.
      * Ratios can be negative; they are not probabilities.
    """
    score = blosum_aligned(sample, native, **kwargs)
    reference = blosum_aligned(native, native, **kwargs)
    if score is None or not reference:
        return None
    return score / reference


def blosum_provenance():
    """Return BLOSUM defaults and normalization formulas as an artifact metadata dict.

    Alignment version 1 uses gap_open + (length - 1) * gap_extend and forbids
    adjacent gaps in opposite sequences. Record custom gap scores separately.
    """
    return {
        "matrix": BLOSUM62_MATRIX_NAME,
        "gap_open": BLOSUM62_GAP_OPEN,
        "gap_extend": BLOSUM62_GAP_EXTEND,
        "unknown_residue_score": BLOSUM62_UNKNOWN,
        "alignment": "needleman-wunsch global, affine gaps (Gotoh)",
        "alignment_version": 1,
        "adjacent_opposing_gaps": False,
        "gap_cost_formula": "gap_open + (length - 1) * gap_extend",
        "positional_normalisation": "mean_per_position_score / native_self_score",
        "aligned_normalisation": "alignment_score / native_self_alignment_score",
    }


def calculate(samples, native=None):
    """Return sequence metrics for one framework's candidate draws as a dict.

    BLOSUM ratios are omitted when the native self-score is zero.
    """

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
        # Reuse candidate alignments and one native self-score for both summaries.
        raw_scores = [blosum_aligned(sample, native) for sample in samples]
        raw_scores = [score for score in raw_scores if score is not None]
        if raw_scores:
            native_self_score = blosum_aligned(native, native)
            if native_self_score:
                aligned_ratios = [score / native_self_score for score in raw_scores]
                out["blosum62_aligned_ratio_mean"] = sum(aligned_ratios) / len(aligned_ratios)
                out["blosum62_aligned_mean"] = sum(raw_scores) / len(raw_scores)
        if matching:
            positional = [blosum_positional(sample, native) for sample in matching]
            positional = [value for value in positional if value is not None]
            if positional:
                out["blosum62_mean_at_length"] = sum(positional) / len(positional)
                ratios = [blosum_identity_ratio(sample, native) for sample in matching]
                ratios = [value for value in ratios if value is not None]
                if ratios:
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
