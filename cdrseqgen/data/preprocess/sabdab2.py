"""Data preprocessing script: SAbDab2, antigen-aware (abag_split.csv)

Data:
  * Source                  -- abag_split.csv (.cif available but not used yet)
  * Size                    -- 3984
  * Modality                -- sequence and structure coordinates
  * Paired-chain context    -- yes
  * Antigen context         -- yes

Key Functions:
  * build_entries       -- master function: load CSV, filter rows, split regions, assemble entries
  * split_imgt_regions  -- parsed per-residue numbering -> the seven IMGT region strings
  * is_protein_antigen  -- keep protein/peptide antigens, drop sugar/hapten/ion-only entries

Notes:
  * Regions are derived from SAbDab2's per-residue IMGT numbering rather than ready-made columns.
  * The 'split' field is SAbDab2's own `ab_ag_split` column, NOT computed here — unlike the OAS
    script, which has to cluster CDR3s itself. SAbDab2 ships an antibody-antigen-aware split.
  * Structure-derived, so this is the antigen/binding dataset; OAS remains the large unbound
    sequence one.
  * Multiple antigen chains are joined with 'X', not '/': no amino-acid vocabulary has a token for
    '/' (it would encode to <unk>), while 'X' is the standard unknown-residue convention.
  * Reason for 15641 CSV rows -> 3984 entries (no SAbDab2 official figure):
    The funnel: 15641 raw -> 12160 paired (drops 3350 SD-H + 67 VNAR + 64 SD-L single-domain)
    -> 3985 protein/peptide antigen (drops 8175 with no antigen or a non-protein one)
    -> 3985 with a resolved antigen sequence (drops 0) -> 3984 kept. The final row lost is
    pdb_000011sz_C_D, whose heavy CDR1 is empty. Every run reprints this funnel.
"""
import argparse
import json

import pandas as pd
from tqdm import tqdm

from cdrseqgen.data.preprocess.common import (
    valid_regions,
    build_chain,
    add_io_args,
    optional_text,
    print_funnel,
    write_processed,
    write_processed_metadata,
)

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


def split_imgt_regions(numbering):
    """Fold parsed numbering into the seven region strings.

    `numbering` is a decoded *_numbering_list column: [[aa, [imgt_pos, insertion_code]], ...].
    Taking it already-parsed (rather than the raw JSON string) mirrors the OAS script, where the
    numbering is parsed once in the loop and reused for the entry's imgt_numbering field.
    """
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


def build_entries(csv_path, limit=None):
    """Load SAbDab2's abag_split.csv and build one entry per usable antibody-antigen complex.

    Keeps paired heavy+light structures (Fab / Fv / Fab+Fc — single-domain excluded) whose antigen is
    a resolved protein or peptide, then drops rows whose numbering won't parse or that have an empty
    CDR, printing the surviving count at each stage. `limit` truncates the input for a quick
    sanity check. Returns the entry list; 'split' comes from the CSV, so no split step follows.
    """
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
    skipped = {"malformed-numbering": 0, "missing/empty regions": 0}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Parse the numbering once and reuse it for both the regions and the entry (as OAS does).
        try:
            heavy_imgt = json.loads(row["VH_numbering_list"])
            light_imgt = json.loads(row["VL_numbering_list"])
        except (ValueError, TypeError):
            skipped["malformed-numbering"] += 1
            continue

        heavy_regions = split_imgt_regions(heavy_imgt)
        light_regions = split_imgt_regions(light_imgt)
        if not (valid_regions(heavy_regions) and valid_regions(light_regions)):
            skipped["missing/empty regions"] += 1
            continue

        # Multiple antigen chains are '/'-joined by SAbDab2; the amino-acid tokenizer has no
        # representation for '/', so use 'X' (the standard "unknown residue" convention) as a
        # soft chain-break marker instead.
        antigen_seq = "X".join(row["agresolvedseqs"].split("/"))

        entry = {
            "sequence_id": row["INSTANCE"],
            "chain_sequence_ids": {
                "heavy": optional_text(row["Hchain"]),
                "light": optional_text(row["Lchain"]),
            },
            # This source currently exposes a species column with no populated values. Preserve
            # unknown as JSON null rather than silently conditioning downstream models as human.
            "species": optional_text(row["species"]),
            "source": {
                "dataset": "SAbDab2",
                "pdb_id": optional_text(row["PDB_ID"]),
                "sabdab_id": optional_text(row["SABDAB_ID"]),
            },
            "heavy": build_chain(heavy_regions, row["VH_numerable_seq"], heavy_imgt),
            "light": build_chain(light_regions, row["VL_numerable_seq"], light_imgt),
            "antigen": {
                "full_aa": antigen_seq,
                "chains": row["agchains"].split("/") if isinstance(row["agchains"], str) else [],
                "type": row["agtypes"],
            },
            "split": row["ab_ag_split"],
        }
        processed_data.append(entry)

    print_funnel([(n_raw, "raw"), (n_paired, "paired-chain"), (n_protein_ag, "protein/peptide-antigen"),
                  (n_resolved, "with resolved antigen seq")], skipped, len(processed_data))
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SAbDab2 paired antibody-antigen data.")
    add_io_args(parser,
                input_default="dataset/data/raw/sabdab2/splits_final/abag_split.csv",
                output_default="dataset/data/processed/processed_sabdab2_data.json",
                input_help="Path to SAbDab2's abag_split.csv")
    args = parser.parse_args()

    data = build_entries(args.input_file, limit=args.limit)
    write_processed(data, args.output_file)
    species_known = sum(entry["species"] is not None for entry in data)
    write_processed_metadata(args.output_file, {
        "dataset": "SAbDab2 antibody-antigen",
        "source": {"dataset": "SAbDab2", "file": args.input_file},
        "records": len(data),
        "species": {
            "known_records": species_known,
            "unknown_records": len(data) - species_known,
        },
        "preprocessing": {
            "paired_chains_only": True,
            "protein_or_peptide_antigen_only": True,
            "resolved_antigen_only": True,
            "numbering_scheme": "IMGT",
            "split": "source ab_ag_split",
            "limit": args.limit,
        },
    })
