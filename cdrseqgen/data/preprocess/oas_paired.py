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
import argparse
import ast
import csv
import json
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
    write_processed_metadata,
)


def read_oas_metadata(csv_path):
    """Read and normalize OAS's JSON metadata record from the first CSV row."""
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
    }


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


def build_entries(csv_path, limit=None, source_metadata=None):
    """Load the OAS paired CSV and build one entry per usable antibody.

    Keeps productive heavy+light pairs (no stop codons / frameshifts), then drops rows whose ANARCI
    numbering won't parse or that have a missing/empty region, printing the surviving count at each
    stage. `limit` truncates the input for a quick sanity check. Returns the entry list with no
    'split' field yet; add_split_field attaches that.
    """
    source_metadata = source_metadata or read_oas_metadata(csv_path)

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
            "chain_sequence_ids": {
                "heavy": row['sequence_id_heavy'],
                "light": row['sequence_id_light'],
            },
            "species": source_metadata["species"],
            "source": {
                "dataset": source_metadata["dataset"],
                "run": source_metadata["run"],
                # The donor, not the sequencing run: one subject spans many runs, so a
                # donor-held-out split can only be built if this travels with the entry.
                "subject": source_metadata.get("subject"),
            },
            "isotype": row['Isotype_heavy'],
            "heavy": build_chain(heavy_regions, row['sequence_alignment_aa_heavy'], heavy_imgt),
            "light": build_chain(light_regions, row['sequence_alignment_aa_light'], light_imgt),
        }
        processed_data.append(entry)

    print_funnel([(n_raw, "raw"), (n_productive, "productive")], skipped, len(processed_data))
    return processed_data


def build_corpus(csv_paths, limit=None, dedup=True):
    """Build ONE entry list from several OAS paired CSVs, each read with its own metadata.

    Concatenation happens here, before any split is assigned, and that ordering is the whole point.
    `add_split_field` clusters heavy CDR3s and allocates WHOLE clusters to train/validation/test, so
    running it per file would give each file an independent clustering — one clonal family could
    then land in train from donor A's file and in test from donor B's, and the merged JSON would
    look perfectly well-formed. Splitting once over the merged list is what makes a multi-donor
    corpus leakage-safe.

    `limit` is per file, since it exists to sanity-check a source rather than to size the corpus.

    Returns (entries, per_file, stats). `per_file` carries each source's metadata plus how many
    entries it contributed, so the sidecar can show the composition.
    """
    entries, per_file = [], []
    seen = {}
    duplicates = 0

    for path in csv_paths:
        metadata = read_oas_metadata(path)
        print(f"\n=== {path} (run {metadata['run']}, subject {metadata.get('subject')}) ===")
        built = build_entries(path, limit=limit, source_metadata=metadata)

        kept = 0
        for entry in built:
            if dedup:
                # The biological identity, not sequence_id: ids are assigned per run and say
                # nothing about whether two rows are the same antibody.
                key = (entry["heavy"]["full_aa"], entry["light"]["full_aa"])
                if key in seen:
                    duplicates += 1
                    continue
                seen[key] = True
            entries.append(entry)
            kept += 1

        per_file.append({**metadata, "path": str(path), "entries": kept})
        print(f"    contributed {kept} entr{'y' if kept == 1 else 'ies'}")

    donors = sorted({str(m.get("subject")) for m in per_file})
    species = sorted({m["species"] for m in per_file})
    # The longest loop is a hard constraint downstream, not a curiosity: the causal layout reserves
    # a fixed number of answer positions (dataset_ar.LOOP_LENGTH_RESERVE) and refuses any loop
    # longer than that at data-load time. Report it here, where a new corpus is being built, rather
    # than letting a run discover it. The single-donor corpus topped out at exactly 40 and the
    # reserve was 40; the 4-donor corpus holds one 49-residue H3.
    longest = {}
    for chain in ("heavy", "light"):
        for region in ("cdr1", "cdr2", "cdr3"):
            key = f"{chain[0]}{region[-1]}"
            longest[key] = max((len(e[chain][region]) for e in entries), default=0)
    stats = {"duplicates_dropped": duplicates, "donors": donors, "species": species,
             "longest_loop": longest}

    print(f"\nCorpus: {len(entries)} entries from {len(csv_paths)} file(s), "
          f"{len(donors)} donor(s): {', '.join(donors)}")
    if duplicates:
        print(f"        dropped {duplicates} duplicate heavy+light pair(s) "
              f"(--dedup none keeps them)")
    if len(species) > 1:
        print(f"        WARNING: mixed species in one corpus: {', '.join(species)}")
    print("        longest loop: "
          + "  ".join(f"{k} {v}" for k, v in sorted(longest.items())))
    return entries, per_file, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OAS paired antibody data.")
    add_io_args(parser,
                input_default="dataset/data/raw/1287169_1_Paired_All.csv",
                output_default="dataset/data/processed/processed_oas_data.json",
                input_help="One or more OAS paired CSVs. Several files become ONE corpus: they are "
                           "concatenated first and the split is assigned once over the merged set.",
                multiple=True)
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
    parser.add_argument("--dedup", choices=["exact", "none"], default="exact",
                        help="Drop repeated heavy+light amino-acid pairs across the whole corpus. "
                             "Public/convergent clones recur across donors, and with --split "
                             "random or none nothing else stops an identical pair straddling the "
                             "holdout. 'none' reproduces pre-2026-09-03 behaviour.")
    args = parser.parse_args()

    data, per_file, stats = build_corpus(args.input_file, limit=args.limit,
                                         dedup=args.dedup == "exact")
    # Once, over the merged corpus — see build_corpus on why this must not run per file.
    add_split_field(data, args.split, args.test_size, args.seed, args.cluster_identity,
                    validation_size=args.validation_size)
    write_processed(data, args.output_file)
    write_processed_metadata(args.output_file, {
        "dataset": "OAS paired",
        # Always a list, one element per input CSV, even for a single file: a corpus that can be
        # several donors should not change shape depending on how many it happens to hold.
        "sources": per_file,
        "donors": stats["donors"],
        "longest_loop": stats["longest_loop"],
        "records": len(data),
        "preprocessing": {
            "productive_pairs_only": True,
            "numbering_scheme": "IMGT",
            "split": args.split,
            "test_fraction": args.test_size,
            "validation_fraction": args.validation_size,
            "cluster_identity": args.cluster_identity if args.split == "cluster" else None,
            "seed": args.seed,
            "limit": args.limit,
            "dedup": args.dedup,
            "duplicates_dropped": stats["duplicates_dropped"],
        },
    })
