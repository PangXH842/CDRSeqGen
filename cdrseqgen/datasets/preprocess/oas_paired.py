"""Corpus-specific functions for: OAS paired antibody sequences.

Data:
  * Source                  -- OAS, 8 csvs from 6 donors
  * Size                    -- 152727
  * Modality                -- sequence-only
  * Paired-chain context    -- yes
  * Antigen context         -- no

Notes:
  * The default split clusters heavy CDR3 only. Same-length sequences join the first matching
    representative at the chosen identity threshold; this is not an all-pairs homology audit
    and does not prevent similarity leakage for other target loops.
"""
import csv
import json
import random
from collections import defaultdict

from tqdm import tqdm

from cdrseqgen.datasets.preprocess.common import REGIONS


def read_oas_metadata(csv_path):
    """Return normalized source metadata from an OAS CSV's one-field first row."""
    with open(csv_path, newline="") as handle:
        row = next(csv.reader(handle), None)
    if not row or len(row) != 1:
        raise ValueError(f"{csv_path} does not start with the expected one-field OAS metadata row")
    try:
        raw = json.loads(row[0])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"could not parse OAS metadata row in {csv_path}: {exc}") from exc

    species = str(raw.get("Species", "")).strip().lower()
    if not species:
        raise ValueError(f"OAS metadata row in {csv_path} has no Species value")
    run = raw.get("Run")
    if run in (None, ""):
        raise ValueError(f"OAS metadata row in {csv_path} has no Run value")
    return {
        "dataset": "OAS",
        "run": int(run),
        "species": species,
        "source_url": raw.get("Link"),
        "author": raw.get("Author"),
        "subject": raw.get("Subject"),
        "b_type": raw.get("BType"),
    }


def cluster_by_cdr3(cdr3_seqs, identity=0.85):
    """Return (cluster_ids, n_clusters) from greedy, same-length sequence clustering.

    cdr3_seqs is an ordered sequence of heavy-CDR3 strings. Within each length bucket, assign
    each sequence to the first representative with at least identity * length matching
    positions, or create a new representative. IDs correspond to the original input order.

    Only representatives are compared: clustering is order-dependent, ignores indels and is
    not a connected-components or all-pairs similarity guarantee. It does not establish that
    all clonal relatives share a cluster. Split audits below count overlap of these IDs.
    """
    by_len = defaultdict(list)
    for i, s in enumerate(cdr3_seqs):
        by_len[len(s)].append(i)

    cluster_id = [-1] * len(cdr3_seqs)
    next_cid = 0
    for L, idxs in tqdm(by_len.items(), desc="clustering CDR3 by length"):
        threshold = identity * L
        reps = []  # (representative_seq, cluster_id)
        for i in idxs:
            s = cdr3_seqs[i]
            cid = None
            for rep_seq, rcid in reps:
                matches = sum(a == b for a, b in zip(s, rep_seq))
                if matches >= threshold:
                    cid = rcid
                    break
            if cid is None:
                cid = next_cid
                next_cid += 1
                reps.append((s, cid))
            cluster_id[i] = cid
    return cluster_id, next_cid


def assign_cluster_splits(cluster_ids, test_size=0.1, validation_size=0.0, seed=42):
    """Return one train/validation/test label per supplied cluster ID.

    Shuffle distinct clusters with seed, then allocate whole clusters to test followed by
    validation until each reaches its rounded target count. Both fractions refer to all
    entries; whole-cluster allocation can overshoot a target. Remaining clusters are train.

    Fractions must be nonnegative and sum to less than one. For identical cluster_ids,
    test_size and seed, changing validation_size does not change test membership.
    """
    if test_size < 0 or validation_size < 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be non-negative and sum to less than 1")

    members = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        members[c].append(i)

    cids = list(members)
    random.Random(seed).shuffle(cids)

    n = len(cluster_ids)
    split = ["train"] * n
    cursor = 0
    for label, fraction in (("test", test_size), ("validation", validation_size)):
        target = round(fraction * n)
        assigned = 0
        while cursor < len(cids) and assigned < target:
            c = cids[cursor]
            cursor += 1
            for i in members[c]:
                split[i] = label
            assigned += len(members[c])
    return split


def add_split_field(processed_data, mode, test_size, seed, cluster_identity, validation_size=0.0):
    """Attach split labels to processed_data in place and print cluster-ID overlap.

    mode='cluster' assigns whole heavy-CDR3 clusters; mode='random' shuffles individual
    entries and allocates test then validation fractions. Both cluster the heavy CDR3s at
    cluster_identity to report shared cluster counts for each pair of partitions. This
    report is not an exhaustive cross-partition sequence-similarity search.

    mode='none' returns without assigning/removing labels or running the audit. The CLI
    restricts mode to these three choices. Fractions are relative to the complete entry list;
    seed controls allocation, and validation_size=0 gives a train/test split. Returns None.
    """
    if mode == "none":
        print("No 'split' field written — train.py will draw a random split at load time (leaky).")
        return

    cluster_ids, n_clusters = cluster_by_cdr3(
        [e["heavy"]["cdr3"] for e in processed_data], cluster_identity
    )

    if mode == "cluster":
        split = assign_cluster_splits(cluster_ids, test_size, validation_size, seed)
    else:  # random
        if test_size < 0 or validation_size < 0 or test_size + validation_size >= 1:
            raise ValueError("test_size and validation_size must be non-negative and sum to less than 1")
        rng = random.Random(seed)
        idx = list(range(len(processed_data)))
        rng.shuffle(idx)
        n_test = round(test_size * len(idx))
        n_validation = round(validation_size * len(idx))
        test_set = set(idx[:n_test])
        validation_set = set(idx[n_test:n_test + n_validation])
        split = [
            "test" if i in test_set else "validation" if i in validation_set else "train"
            for i in range(len(processed_data))
        ]

    for entry, s in zip(processed_data, split):
        entry["split"] = s

    train_clusters = {cluster_ids[i] for i, s in enumerate(split) if s == "train"}
    validation_clusters = {cluster_ids[i] for i, s in enumerate(split) if s == "validation"}
    test_clusters = {cluster_ids[i] for i, s in enumerate(split) if s == "test"}
    n_test = split.count("test")
    n_validation = split.count("validation")
    leakage = {
        "train/validation": len(train_clusters & validation_clusters),
        "train/test": len(train_clusters & test_clusters),
        "validation/test": len(validation_clusters & test_clusters),
    }
    total = len(split) or 1  # Empty corpora have zero entries in every partition.
    print(f"Split '{mode}': {n_clusters} CDR3 clusters (>={cluster_identity:.0%} identity); "
          f"train = {split.count('train')} ({split.count('train') / total:.1%}); "
          f"validation = {n_validation} ({n_validation / total:.1%}); "
          f"test = {n_test} ({n_test / total:.1%}). "
          f"Cluster overlap: {leakage}.")


def build_regions(row, suffix):
    """Return seven region values from an OAS row for suffix='heavy' or 'light'."""
    return {r: row[f"{r}_aa_{suffix}"] for r in REGIONS}


def add_source_arguments(parser):
    """Register the flags only this source understands; __main__.args imports them."""
    parser.add_argument("--split", choices=["cluster", "random", "none"], default="cluster",
                        help="Train/test 'split' field to attach. 'none' = the original behaviour, "
                             "omit it and let train.py split randomly at load; 'random' = that same "
                             "split, seeded and written to the file; 'cluster' (default) = the "
                             "leakage-safe CDR3-identity holdout added later. See the module docstring.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of entries held out as test.")
    parser.add_argument("--validation_size", type=float, default=0.1,
                        help="Fraction held out as validation after test allocation. Set 0 to reproduce "
                             "the historical two-way split.")
    parser.add_argument("--cluster_identity", type=float, default=0.85,
                        help="Min heavy-CDR3 identity to group sequences into one clonal cluster.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible cluster/random split.")
    parser.add_argument("--dedup", choices=["exact", "none"], default="exact",
                        help="Drop repeated heavy+light amino-acid pairs across the whole corpus.")
    return parser


if __name__ == "__main__":
    from cdrseqgen.datasets.preprocess.__main__ import main

    raise SystemExit(main(source_name="oas_paired"))
