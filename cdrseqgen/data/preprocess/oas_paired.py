"""Data preprocessing script: OAS paired antibody sequences.

Data:
  * Source                  -- OAS, 8 csvs from 6 donors
  * Size                    -- 152727
  * Modality                -- sequence-only
  * Paired-chain context    -- yes
  * Antigen context         -- no

Key Functions:
  CLI (__main__)
  * build_corpus            -- merge sources, optionally deduplicate pairs, collect statistics
    * read_oas_metadata     -- parse each CSV's source, species, run and subject metadata
    * build_entries         -- filter one CSV and assemble paired-chain entries
      * build_regions       -- collect seven precomputed IMGT region columns for one chain
      * (common -- build_chain, valid_regions, print_funnel)
  * add_split_field         -- attach split labels in place and report cluster overlap
    * cluster_by_cdr3       -- group equal-length heavy CDR3s by representative identity
    * assign_cluster_splits -- allocate whole clusters to train/validation/test

Notes:
  * The corpus count above describes the current processed_oas_multi.metadata.json, not a
    fixed property of this script. Inputs, filters, row limits and deduplication determine size.
  * Keep rows marked productive for both chains, with parseable ANARCI numbering, string
    values for all seven regions and nonempty CDRs. Region strings come directly from OAS.
  * Merge all inputs before splitting. Exact deduplication (the default) keeps the first
    occurrence of each full heavy/light sequence pair, including its source attribution.
  * The default split clusters heavy CDR3 only. Same-length sequences join the first matching
    representative at the chosen identity threshold; this is not an all-pairs homology audit
    and does not prevent similarity leakage for other target loops.
  * Test clusters are allocated before optional validation clusters. With the same entries,
    cluster assignments and seed, adding validation preserves the historical test set.
  * Random splitting assigns entries individually and reports shared cluster IDs. Mode none
    leaves split fields untouched (new entries have none) and skips the clustering audit.
  * The CLI writes the entry JSON and a provenance sidecar with its hash, source composition,
    deduplication counts, longest CDR lengths and preprocessing/split settings.
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
    """Return normalized source metadata from an OAS CSV's one-field first row.

    Parse that field as JSON, lowercase/strip Species and convert Run to int. Return dataset,
    run, species, source_url, author and subject; optional source fields may be None. Invalid
    row/JSON content, missing required values or a non-integer run raise rather than silently
    assigning a species or run. The sequence-table rows are not loaded here.
    """
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
    print(f"Split '{mode}': {n_clusters} CDR3 clusters (>={cluster_identity:.0%} identity); "
          f"train = {split.count('train')} ({split.count('train') / len(split):.1%}); "
          f"validation = {n_validation} ({n_validation / len(split):.1%}); "
          f"test = {n_test} ({n_test / len(split):.1%}). "
          f"Cluster overlap: {leakage}.")


def build_regions(row, suffix):
    """Return seven region values from an OAS row for suffix='heavy' or 'light'.

    For example, fwr1 comes from row['fwr1_aa_heavy'] for the heavy chain. Values are copied
    without validation; a missing source column raises KeyError. valid_regions checks them.
    """
    return {r: row[f"{r}_aa_{suffix}"] for r in REGIONS}


def build_entries(csv_path, limit=None, source_metadata=None):
    """Return paired-chain entries from one OAS CSV, before deduplication or splitting.

    Use source_metadata when supplied, otherwise read the CSV's first metadata row. Load the
    table after that row and keep pairs whose two productive flags equal 'T'. Apply limit
    after this filter (None or zero leaves all rows), then parse ANARCI numbering with
    literal_eval and require valid region strings/nonempty CDRs on both chains.

    Numbering parse failures and invalid regions are counted and skipped. Missing columns or
    other schema errors propagate. Entries retain both sequence IDs, source species/run/subject,
    heavy-chain isotype, full sequences, seven regions per chain and the parsed numbering.
    No antigen or split field is added. Print the filter funnel before returning the list.
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
    """Merge OAS CSVs into one entry list and return (entries, per_file, stats).

    Read each path in order with its own metadata and build_entries filters. limit applies
    to productive rows separately in each file, not to the final corpus. With dedup=True,
    keep only the first occurrence of each exact (heavy.full_aa, light.full_aa) pair across
    all files; retain the first entry's identifiers and provenance. dedup=False keeps repeats.

    per_file records source metadata, path and retained contribution counts, including files
    that contribute no entries. stats contains duplicates_dropped, source subject labels
    (donors), species, and longest_loop lengths for h1-h3/l1-l3. A missing subject is represented
    as the string 'None' in the subject summary; this is not donor-identity validation.

    No split is assigned here. Call add_split_field once on the merged entries so the same
    clustering is used across every input file. Print source contributions and corpus statistics.
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
