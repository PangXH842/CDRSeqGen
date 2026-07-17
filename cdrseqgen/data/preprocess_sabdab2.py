import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

# Standard IMGT region boundaries (inclusive), same convention the OAS ANARCI columns use.
IMGT_BOUNDARIES = {
    "fwr1": (1, 26),
    "cdr1": (27, 38),
    "fwr2": (39, 55),
    "cdr2": (56, 65),
    "fwr3": (66, 104),
    "cdr3": (105, 117),
    "fwr4": (118, 128),
}

# type == paired heavy+light (Fab, Fv-only, or Fab+Fc). Excludes single-domain (SD-H/SD-L/VNAR).
PAIRED_TYPES = {"FAB", "FV", "FAB+FC"}
PROTEIN_AG_TYPES = {"PROTEIN", "PEPTIDE"}


def split_imgt_regions(numbering_json):
    """SAbDab2's *_numbering_list columns are JSON: [[aa, [imgt_pos, insertion_code]], ...]."""
    numbering = json.loads(numbering_json)
    regions = {name: [] for name in IMGT_BOUNDARIES}
    for aa, (pos, _ins) in numbering:
        for name, (lo, hi) in IMGT_BOUNDARIES.items():
            if lo <= pos <= hi:
                regions[name].append(aa)
                break
    return {name: "".join(res) for name, res in regions.items()}


def is_protein_antigen(agtypes):
    """Keep only protein/peptide antigens; excludes sugar/hapten/ion-only entries."""
    if not isinstance(agtypes, str):
        return False
    return all(t in PROTEIN_AG_TYPES for t in agtypes.split("/"))


def valid_regions(region_dict):
    """Every CDR (any of which can be the masked target) must be non-empty."""
    return all(region_dict.get(cdr) for cdr in ("cdr1", "cdr2", "cdr3"))


def preprocess_sabdab2(csv_path, limit=None):
    df = pd.read_csv(csv_path, low_memory=False)
    n_raw = len(df)

    # Paired heavy+light structures with a resolved protein/peptide antigen only.
    df = df[df["type"].isin(PAIRED_TYPES)]
    n_paired = len(df)
    df = df[df["agtypes"].apply(is_protein_antigen)]
    n_protein_ag = len(df)
    df = df[df["agresolvedseqs"].notna() & (df["agresolvedseqs"].astype(str).str.len() > 0)]
    n_resolved = len(df)

    if limit:
        df = df.head(limit)

    processed_data = []
    skipped = {"numbering": 0, "regions": 0}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            heavy_regions = split_imgt_regions(row["VH_numbering_list"])
            light_regions = split_imgt_regions(row["VL_numbering_list"])
        except (ValueError, TypeError):
            skipped["numbering"] += 1
            continue

        if not (valid_regions(heavy_regions) and valid_regions(light_regions)):
            skipped["regions"] += 1
            continue

        # Multiple antigen chains are '/'-joined by SAbDab2; the amino-acid tokenizer has no
        # representation for '/', so use 'X' (the standard "unknown residue" convention) as a
        # soft chain-break marker instead.
        antigen_seq = "X".join(row["agresolvedseqs"].split("/"))

        entry = {
            "sequence_id": row["INSTANCE"],
            "heavy": {
                **heavy_regions,
                "full_aa": row["VH_numerable_seq"],
                "imgt_numbering": json.loads(row["VH_numbering_list"]),
            },
            "light": {
                **light_regions,
                "full_aa": row["VL_numerable_seq"],
                "imgt_numbering": json.loads(row["VL_numbering_list"]),
            },
            "antigen": {
                "full_aa": antigen_seq,
                "chains": row["agchains"].split("/") if isinstance(row["agchains"], str) else [],
                "type": row["agtypes"],
            },
            "split": row["ab_ag_split"],
        }
        processed_data.append(entry)

    print(f"Rows: {n_raw} raw -> {n_paired} paired-chain -> {n_protein_ag} protein/peptide-antigen "
          f"-> {n_resolved} with resolved antigen seq -> {len(processed_data)} kept "
          f"(skipped {skipped['numbering']} malformed-numbering, {skipped['regions']} missing/empty regions).")
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SAbDab2 paired antibody-antigen data.")
    # Defaults are relative to the REPO ROOT, which is where this is run from
    # (python -m cdrseqgen.data.preprocess_sabdab2). The data blob lives outside the package.
    parser.add_argument("--input_file", help="Path to SAbDab2's abag_split.csv",
                         default="dataset/data/raw/sabdab2/splits_final/abag_split.csv")
    parser.add_argument("--output_file", help="Path to the output JSON file",
                         default="dataset/data/processed/processed_sabdab2_data.json")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N rows (for sanity-checking).")
    args = parser.parse_args()

    data = preprocess_sabdab2(args.input_file, limit=args.limit)

    # dataset/data/ is gitignored, so the output tree does not exist on a fresh clone.
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(data, f)

    print(f"Wrote {len(data)} entries to {args.output_file}")
