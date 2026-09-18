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
import ast
import csv

import pandas as pd
import json
import random
from collections import defaultdict

from tqdm import tqdm

from cdrseqgen.datasets.preprocess.common import REGIONS, RejectedEntry


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


def filter_rows(path, **_options):
    """Read one CSV and keep the rows this source considers usable, with its funnel stages."""
    # The first row is metadata, not a header.
    df = pd.read_csv(path, skiprows=1, low_memory=False)
    stages = [(len(df), "raw")]

    # Productive pairs only -- no stop codons or frameshifts.
    df = df[(df['productive_heavy'] == 'T') & (df['productive_light'] == 'T')]
    stages.append((len(df), "productive"))
    return df, stages, ("malformed-numbering", "missing/empty regions")


def read_metadata(path):
    """Per-file provenance hook: OAS ships a metadata row, so each CSV announces itself."""
    metadata = read_oas_metadata(path)
    print(f"\n=== {path} (run {metadata['run']}, subject {metadata.get('subject')}) ===")
    return metadata


def dedup_key(entry, options):
    """Deduplication hook: the biological identity, or None to keep every row.

    Not sequence_id -- ids are assigned per run and say nothing about whether two rows are
    the same antibody. Several donors' files become one corpus, so clones recur across them.
    """
    if options.dedup != "exact":
        return None
    return tuple((entry.get(chain) or {}).get("full_aa") for chain in ("heavy", "light"))


def finalize(entries, *, per_file, options, duplicates, funnel):
    """Assign the split across the merged corpus and return its provenance metadata."""
    donors = sorted({str(m.get("subject")) for m in per_file})
    species = sorted({m["species"] for m in per_file})
    # Calculate longest loop, used for the causal model's max answer length
    longest = {}
    for chain in ("heavy", "light"):
        for region in ("cdr1", "cdr2", "cdr3"):
            longest[f"{chain[0]}{region[-1]}"] = max(
                (len(e[chain][region]) for e in entries if e.get(chain)), default=0)

    print(f"\nCorpus: {len(entries)} entries from {len(per_file)} file(s), "
          f"{len(donors)} donor(s): {', '.join(donors)}")
    if duplicates:
        print(f"        dropped {duplicates} duplicate heavy+light pair(s) "
              f"(--dedup none keeps them)")
    if len(species) > 1:
        print(f"        WARNING: mixed species in one corpus: {', '.join(species)}")
    print("        longest loop: " + "  ".join(f"{k} {v}" for k, v in sorted(longest.items())))

    add_split_field(entries, options.split, options.test_size, options.seed,
                    options.cluster_identity, validation_size=options.validation_size)
    return {
        "dataset": "OAS paired",
        "sources": per_file,
        "donors": donors,
        "longest_loop": longest,
        "records": len(entries),
        "preprocessing": {
            "productive_pairs_only": True,
            "numbering_scheme": "IMGT",
            "split": options.split,
            "test_fraction": options.test_size,
            "validation_fraction": options.validation_size,
            "cluster_identity": (options.cluster_identity if options.split == "cluster" else None),
            "seed": options.seed,
            "limit": options.limit,
            "dedup": options.dedup,
            "duplicates_dropped": duplicates,
        },
    }


def row_parts(row, source_metadata):
    """This source's per-row values for the shared entry assembly (see __main__.build_entry)."""
    # ANARCI numbering is a stringified dict in OAS. Narrow except: only swallow parse
    # failures (incl. NaN passed to literal_eval), NOT e.g. a KeyError from a renamed column.
    try:
        heavy_numbering = ast.literal_eval(row['ANARCI_numbering_heavy'])
        light_numbering = ast.literal_eval(row['ANARCI_numbering_light'])
    except (ValueError, SyntaxError, TypeError) as exc:
        raise RejectedEntry("malformed-numbering") from exc

    return {
        "sequence_id": row['sequence_id_heavy'],
        "chain_ids": {"heavy": row['sequence_id_heavy'], "light": row['sequence_id_light']},
        "heavy_regions": build_regions(row, "heavy"),
        "light_regions": build_regions(row, "light"),
        "heavy_aa": row['sequence_alignment_aa_heavy'],
        "light_aa": row['sequence_alignment_aa_light'],
        "heavy_numbering": heavy_numbering,
        "light_numbering": light_numbering,
        "species": source_metadata["species"],
        "isotype": row['Isotype_heavy'],
        "split": None,                       # assigned once after merging, in add_split_field
        "source": {
            "dataset": source_metadata["dataset"],
            "run": source_metadata["run"],
            # The donor, not the sequencing run: one subject spans many runs, so a
            # donor-held-out split can only be built if this travels with the entry.
            "subject": source_metadata.get("subject"),
            "b_type": source_metadata.get("b_type"),
        },
    }


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
