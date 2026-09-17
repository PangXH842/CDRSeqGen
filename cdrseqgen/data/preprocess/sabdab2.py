"""Data preprocessing script: SAbDab2 paired antibody-antigen sequences.

Data:
  * Source                  -- abag_split.csv; associated CIF files are not read here
  * Current processed set   -- 3,984 retained entries from the recorded 15,641-row CSV
  * Modality                -- structure-derived sequences and IMGT numbering; no coordinates
  * Paired-chain context    -- yes
  * Antigen context         -- resolved protein/peptide sequences

Key Functions:
  CLI (__main__)
  * build_entries           -- filter one CSV and assemble paired-chain entries
    * is_protein_antigen    -- require every listed antigen type to be protein or peptide
    * split_imgt_regions    -- collect seven IMGT region strings from parsed residue numbering
    * (common -- build_chain, valid_regions, print_funnel, optional_text)

Notes:
  * Keep paired FAB/FV/FAB+FC entries with protein/peptide antigen types and a nonempty
    resolved antigen sequence. Parse both numbering columns and require nonempty CDRs.
  * Regions are derived from residue numbering. Full chain sequences retain the source's
    VH_numerable_seq/VL_numerable_seq values; numbering is also stored in each chain object.
  * Copy ab_ag_split directly into split. This script performs no clustering, deduplication,
    validation-partition construction or independent verification of the source split.
  * Join multiple antigen sequences with 'X' in place of '/'; retain antigen chain IDs and
    the source antigen-type string. Missing species stays None rather than becoming human.
  * The recorded filtering funnel was 15,641 raw -> 12,160 paired -> 3,985 protein/peptide
    antigen -> 3,985 with resolved antigen sequence -> 3,984 kept. The final rejected row,
    pdb_000011sz_C_D, had an empty heavy CDR1. Counts depend on the input version and limit.
  * The CLI writes the entry JSON plus a hashed provenance sidecar with source path,
    species completeness and filtering/split settings. Structure coordinates are not exported.
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
    """Return seven IMGT region strings from parsed residue-numbering records.

    numbering contains [amino_acid, [position, insertion_code]] records decoded from a
    *_numbering_list column. Inclusive IMGT_BOUNDARIES select each region by numeric position;
    insertion codes do not change membership, and residues retain their input order. Positions
    outside all boundaries are ignored; regions with no residues become empty strings.

    No sorting or numbering-schema validation is performed. Malformed record shapes/types
    can raise during iteration, even when their enclosing JSON parsed successfully.
    """
    regions = {name: [] for name in IMGT_BOUNDARIES}
    for aa, (pos, _ins) in numbering:
        for name, (lo, hi) in IMGT_BOUNDARIES.items():
            if lo <= pos <= hi:
                regions[name].append(aa)
                break
    return {name: "".join(res) for name, res in regions.items()}


def is_protein_antigen(agtypes):
    """Return True only when every slash-separated type is PROTEIN or PEPTIDE.

    The input must be a string. Matching is case-sensitive without whitespace normalization;
    mixed protein/non-protein lists, empty strings and missing values return False.
    """
    if not isinstance(agtypes, str):
        return False
    return all(t in PROTEIN_AG_TYPES for t in agtypes.split("/"))


def build_entries(csv_path, limit=None):
    """Return paired antibody-antigen entries from SAbDab2's abag_split.csv.

    Filter to PAIRED_TYPES, exclusively protein/peptide antigen types, and nonmissing,
    nonempty resolved antigen sequences. Apply limit after these filters (None or zero
    leaves all rows). Parse VH/VL numbering JSON, derive regions and skip JSON parse failures
    or invalid regions/empty CDRs, printing the filtering funnel and rejection counts.

    Entries preserve INSTANCE, heavy/light chain IDs, nullable species, PDB/SAbDab provenance,
    full chain sequences and numbering. Join antigen sequences with 'X', retain chain IDs/types,
    and copy ab_ag_split unchanged. No split assignment or deduplication follows here.

    Missing columns and invalid decoded numbering structures propagate as errors; only the
    explicit JSON parse failures and region checks are handled as skipped rows. No CIF files
    are read and no coordinates are included in the returned list.
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
