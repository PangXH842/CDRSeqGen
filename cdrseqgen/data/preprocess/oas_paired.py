"""Data preprocessing script: OAS, paired chains

Data:
  * Source                  -- 1287169_1_Paired_All.csv (to be expanded)
  * Size                    -- 21106
  * Modality                -- sequence-only
  * Paired-chain context    -- yes
  * Antigen context         -- no

Key Functions:
  * build_entries        -- master function: load CSV, filter rows, parse numbering, assemble entries
  * add_split_field      -- attach the per-entry 'split' field (cluster / random / none)
  * cluster_by_cdr3      -- greedy CD-HIT-style clustering of heavy CDR3s into clonal families
  * assign_cluster_split -- hold out whole clusters until ~test_size of entries are in test
  * build_regions        -- collect the seven IMGT region columns for one chain

Notes:
  * Keeps productive heavy+light pairs and lifts each chain's seven IMGT regions straight from OAS's
    own `*_aa_heavy` / `*_aa_light` columns.
  * The split started out random — train.py simply drew one at load time, which is still what
    `--split none` gives you (`--split random` is that same split, seeded and written to the file).
    `--split cluster` was added later and is now the default. Why it was added:
      - A random split is not leakage-safe. Clonal relatives — CDR3s a few somatic mutations apart —
        land on both sides of it, so a model can score well by recognising near-duplicates instead
        of generalising.
      - Measured on the current 21,106-sequence file: random leaks 76 of 2,111 test entries (3.6%)
        back into train; the clustered split leaks 0.
      - The effect on results turned out small — re-running moved the H3 core metric by <= 0.004 —
        so this is about being able to defend the number, not about rescuing it.
  * Splitting lives here rather than in train.py because it has to respect sequence identity across
    the whole corpus. Every mode prints the same leakage count, so whichever you pick is auditable.
  * Entries carry NO antigen — OAS is unbound repertoire sequence. Antigen-conditioned runs use
    sabdab2.py, which emits the same schema plus an 'antigen' block.
"""
import ast
import argparse
import random
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from cdrseqgen.data.preprocess.common import (
    REGIONS,
    valid_regions,
    build_chain,
    add_io_args,
    print_funnel,
    write_processed,
)


def cluster_by_cdr3(cdr3_seqs, identity=0.85):
    """Greedy CD-HIT-style clustering of heavy CDR3s to catch clonally-related antibodies.

    Added to support the 'cluster' split: this is the sequence-identity grouping a leakage-safe
    holdout has to respect, and the yardstick the leakage count is measured against.

    Clonal variants share CDR3 *length* and differ only by a few somatic mutations, so we bucket by
    length and greedily assign each sequence to the first cluster representative it matches at
    >= `identity` amino-acid identity (a new representative otherwise). Same-length identity captures
    the dominant somatic-hypermutation case; for indel-aware clustering, swap in MMseqs2/CD-HIT.

    Returns (cluster_id_per_seq, n_clusters).
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


def assign_cluster_split(cluster_ids, test_size=0.1, seed=42):
    """Backward-compatible two-way clustered split."""
    return assign_cluster_splits(cluster_ids, test_size=test_size, validation_size=0.0, seed=seed)


def assign_cluster_splits(cluster_ids, test_size=0.1, validation_size=0.0, seed=42):
    """Assign whole CDR3 clusters to train/validation/test.

    Test clusters are allocated first using the historical shuffle, so adding validation does not
    change the established test set. Validation clusters then come from what was previously training
    data. Both sizes are fractions of the complete corpus.
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
    """Attach a per-entry train/test or train/validation/test split field.

    With a clustered split, test clusters are allocated first and validation clusters second. This
    preserves the historical test set when validation_size changes from zero to a positive value.
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
    print(f"Split '{mode}': {n_clusters} CDR3 clusters (>={cluster_identity:.0%} identity); "
          f"train = {split.count('train')} ({split.count('train') / len(split):.1%}); "
          f"validation = {n_validation} ({n_validation / len(split):.1%}); "
          f"test = {n_test} ({n_test / len(split):.1%}). "
          f"Cluster overlap: {leakage}.")


def build_regions(row, suffix):
    """The seven IMGT region strings for one chain: suffix='heavy' -> {'fwr1': row['fwr1_aa_heavy'], ...}"""
    return {r: row[f"{r}_aa_{suffix}"] for r in REGIONS}


def build_entries(csv_path, limit=None):
    """Load the OAS paired CSV and build one entry per usable antibody.

    Keeps productive heavy+light pairs (no stop codons / frameshifts), then drops rows whose ANARCI
    numbering won't parse or that have a missing/empty region, printing the surviving count at each
    stage. `limit` truncates the input for a quick sanity check. Returns the entry list with no
    'split' field yet; add_split_field attaches that.
    """
    # The first row is metadata, not a header.
    df = pd.read_csv(csv_path, skiprows=1, low_memory=False)
    n_raw = len(df)

    # Productive pairs only — no stop codons or frameshifts.
    df = df[(df['productive_heavy'] == 'T') & (df['productive_light'] == 'T')]
    n_productive = len(df)

    if limit:
        df = df.head(limit)

    processed_data = []
    skipped = {"malformed-numbering": 0, "missing/empty regions": 0}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # ANARCI numbering is a stringified dict in OAS. Narrow except: only swallow parse failures
        # (incl. NaN passed to literal_eval), NOT e.g. a KeyError from a renamed column.
        try:
            heavy_imgt = ast.literal_eval(row['ANARCI_numbering_heavy'])
            light_imgt = ast.literal_eval(row['ANARCI_numbering_light'])
        except (ValueError, SyntaxError, TypeError):
            skipped["malformed-numbering"] += 1
            continue

        heavy_regions = build_regions(row, "heavy")
        light_regions = build_regions(row, "light")
        if not (valid_regions(heavy_regions) and valid_regions(light_regions)):
            skipped["missing/empty regions"] += 1
            continue

        entry = {
            "sequence_id": row['sequence_id_heavy'],
            "isotype": row['Isotype_heavy'],
            "heavy": build_chain(heavy_regions, row['sequence_alignment_aa_heavy'], heavy_imgt),
            "light": build_chain(light_regions, row['sequence_alignment_aa_light'], light_imgt),
        }
        processed_data.append(entry)

    print_funnel([(n_raw, "raw"), (n_productive, "productive")], skipped, len(processed_data))
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OAS paired antibody data.")
    add_io_args(parser,
                input_default="dataset/data/raw/1287169_1_Paired_All.csv",
                output_default="dataset/data/processed/processed_oas_data.json",
                input_help="Path to the OAS paired CSV")
    parser.add_argument("--split", choices=["cluster", "random", "none"], default="cluster",
                        help="Train/test 'split' field to attach. 'none' = the original behaviour, "
                             "omit it and let train.py split randomly at load; 'random' = that same "
                             "split, seeded and written to the file; 'cluster' (default) = the "
                             "leakage-safe CDR3-identity holdout added later. See the module docstring.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of entries held out as test.")
    parser.add_argument("--validation_size", type=float, default=0.0,
                        help="Fraction held out as validation after test allocation. Set 0 to reproduce "
                             "the historical two-way split.")
    parser.add_argument("--cluster_identity", type=float, default=0.85,
                        help="Min heavy-CDR3 identity to group sequences into one clonal cluster.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible cluster/random split.")
    args = parser.parse_args()

    data = build_entries(args.input_file, limit=args.limit)
    add_split_field(data, args.split, args.test_size, args.seed, args.cluster_identity,
                    validation_size=args.validation_size)
    write_processed(data, args.output_file)