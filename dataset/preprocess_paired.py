import pandas as pd
import ast
import json
import argparse
import random
from collections import defaultdict
from tqdm import tqdm

REGIONS = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]


def cluster_by_cdr3(cdr3_seqs, identity=0.85):
    """Greedy CD-HIT-style clustering of heavy CDR3s to catch clonally-related antibodies.

    Clonal variants share CDR3 *length* and differ only by a few somatic mutations, so we bucket by
    length and greedily assign each sequence to the first cluster representative it matches at
    >= `identity` amino-acid identity (a new representative otherwise). This is the sequence-identity
    grouping that a leakage-safe split must respect: without it, a random split scatters members of
    one clonal family across train and test, and the model scores well by recognising near-duplicates
    rather than generalising (homology leakage). Same-length identity captures the dominant
    somatic-hypermutation case; for indel-aware clustering, swap in MMseqs2/CD-HIT.

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
    """Hold out whole CDR3 clusters until ~test_size of entries are in test, so no clonal relative
    straddles the train/test boundary. Deterministic given seed. Returns a per-entry split list."""
    members = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        members[c].append(i)

    cids = list(members)
    random.Random(seed).shuffle(cids)

    n = len(cluster_ids)
    target = round(test_size * n)
    split = ["train"] * n
    n_test = 0
    for c in cids:
        if n_test >= target:
            break
        for i in members[c]:
            split[i] = "test"
        n_test += len(members[c])
    return split


def add_split_field(processed_data, mode, test_size, seed, cluster_identity):
    """Attach a per-entry 'split' field (train/test) that get_datasets in train.py will honor.

    mode='cluster' gives the leakage-safe CDR3-clustered holdout; mode='random' is a plain seeded
    per-entry split (still leaks clonal near-duplicates); mode='none' attaches nothing (legacy:
    train.py draws a fresh random split at load time). Prints a cross-split leakage check.
    """
    if mode == "none":
        print("No 'split' field written — train.py will draw a random split at load time (leaky).")
        return

    cluster_ids, n_clusters = cluster_by_cdr3([e["heavy"]["cdr3"] for e in processed_data], cluster_identity)

    if mode == "cluster":
        split = assign_cluster_split(cluster_ids, test_size, seed)
    else:  # random
        rng = random.Random(seed)
        idx = list(range(len(processed_data)))
        rng.shuffle(idx)
        n_test = round(test_size * len(idx))
        test_set = set(idx[:n_test])
        split = ["test" if i in test_set else "train" for i in range(len(processed_data))]

    for entry, s in zip(processed_data, split):
        entry["split"] = s

    # Leakage check: how many test entries share a CDR3 cluster with any train entry?
    # 0 for a correct cluster split; > 0 quantifies the leak a random split would carry.
    train_clusters = {cluster_ids[i] for i, s in enumerate(split) if s == "train"}
    leaked = sum(1 for i, s in enumerate(split) if s == "test" and cluster_ids[i] in train_clusters)
    n_test = split.count("test")
    print(f"Split '{mode}': {n_clusters} CDR3 clusters (>={cluster_identity:.0%} identity); "
          f"test = {n_test} entries ({n_test / len(split):.1%}). "
          f"Test entries sharing a CDR3 cluster with train (leakage): {leaked} "
          f"({leaked / max(n_test, 1):.1%}).")


def valid_regions(region_dict):
    """Guard against malformed rows before they reach dataset.py.

    Every region must be a real string (empty CSV cells arrive as float NaN, which would
    later crash the string concatenation in AntibodyInfillingDataset), and every CDR must be
    non-empty since any of the three can be the masked target loop.
    """
    if not all(isinstance(region_dict.get(r), str) for r in REGIONS):
        return False
    return all(region_dict[cdr] for cdr in ("cdr1", "cdr2", "cdr3"))


def build_regions(row, suffix):
    return {r: row[f"{r}_aa_{suffix}"] for r in REGIONS}


def preprocess_oas_paired(file_path):
    # 1. Load data, skipping the first row of metadata
    # We use engine='python' because OAS files sometimes have complex quoting
    df = pd.read_csv(file_path, skiprows=1, low_memory=False)

    # 2. Filter for quality
    # We only want productive antibodies (no stop codons or frameshifts)
    n_raw = len(df)
    df = df[(df['productive_heavy'] == 'T') & (df['productive_light'] == 'T')]
    n_productive = len(df)

    processed_data = []
    skipped = {"numbering": 0, "regions": 0}

    for _, row in tqdm(df.iterrows(), total=n_productive):
        # 3. Clean and parse ANARCI numbering (a stringified dict in OAS).
        # Narrow except: only swallow parse failures (incl. NaN passed to literal_eval),
        # NOT e.g. a KeyError from a renamed column -- that should surface loudly.
        try:
            heavy_imgt = ast.literal_eval(row['ANARCI_numbering_heavy'])
            light_imgt = ast.literal_eval(row['ANARCI_numbering_light'])
        except (ValueError, SyntaxError, TypeError):
            skipped["numbering"] += 1
            continue

        heavy_regions = build_regions(row, "heavy")
        light_regions = build_regions(row, "light")
        if not (valid_regions(heavy_regions) and valid_regions(light_regions)):
            skipped["regions"] += 1
            continue

        # 4. Construct our "Structured Biological Object"
        entry = {
            "sequence_id": row['sequence_id_heavy'],
            "isotype": row['Isotype_heavy'],
            "heavy": {**heavy_regions, "full_aa": row['sequence_alignment_aa_heavy'], "imgt_numbering": heavy_imgt},
            "light": {**light_regions, "full_aa": row['sequence_alignment_aa_light'], "imgt_numbering": light_imgt},
        }
        processed_data.append(entry)

    print(f"Rows: {n_raw} raw -> {n_productive} productive -> {len(processed_data)} kept "
          f"(skipped {skipped['numbering']} malformed-numbering, {skipped['regions']} missing/empty regions).")
    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OAS paired antibody data.")
    parser.add_argument("--input_file", help="Path to the input CSV file", default="./raw/1287169_1_Paired_All.csv")
    parser.add_argument("--output_file", help="Path to the output JSON file", default="./processed/processed_paired_data.json")
    parser.add_argument("--split", choices=["cluster", "random", "none"], default="cluster",
                        help="Train/test 'split' field to attach. 'cluster' (default) = CDR3-identity "
                             "clustered holdout (leakage-safe, recommended); 'random' = seeded per-entry "
                             "random; 'none' = omit (train.py splits randomly at load, the legacy behavior).")
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of entries held out as test.")
    parser.add_argument("--cluster_identity", type=float, default=0.85,
                        help="Min heavy-CDR3 identity to group sequences into one clonal cluster.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible cluster/random split.")
    args = parser.parse_args()

    data = preprocess_oas_paired(args.input_file)
    add_split_field(data, args.split, args.test_size, args.seed, args.cluster_identity)

    with open(args.output_file, 'w') as f:
        json.dump(data, f)

    print(f"Wrote {len(data)} entries to {args.output_file}")