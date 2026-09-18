"""Corpus-specific functions for: SAbDab2 paired antibody-antigen sequences.

Data:
  * Source                  -- abag_split.csv (not including CIFs yet)
  * Size                    -- 3984 (out of 15641)
  * Modality                -- structure-derived sequences and IMGT numbering; no coordinates
  * Paired-chain context    -- yes
  * Antigen context         -- yes, protein/peptide only

Notes:
  * Current processing funnel: 15,641 raw -> 12,160 paired -> 3,985 protein/peptide
    antigen -> 3,985 with resolved antigen sequence -> 3,984 kept. 
  * ab_ag_split.csv comes with a train/test split; only the validation partition is needed. 
  * Antigen context is added by joining the antigen chains with 'X' in place of '/'. Retains 
    antigen chain IDs and the source antigen-type string. 
  * Source does not specify species for each entry, "species" field left as None.
"""
import json
import random

import pandas as pd

from cdrseqgen.datasets.preprocess.common import REGIONS, RejectedEntry, optional_text

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

if list(IMGT_BOUNDARIES) != REGIONS:
    raise ValueError(f"IMGT_BOUNDARIES must name exactly {REGIONS}, in that order; "
                     f"got {list(IMGT_BOUNDARIES)}")

# type == paired heavy+light (Fab, Fv-only, or Fab+Fc). Excludes single-domain (SD-H/SD-L/VNAR).
PAIRED_TYPES = {"FAB", "FV", "FAB+FC"}
PROTEIN_AG_TYPES = {"PROTEIN", "PEPTIDE"}


def split_imgt_regions(numbering):
    """Return seven IMGT region strings from parsed residue-numbering records."""
    regions = {name: [] for name in IMGT_BOUNDARIES}
    for aa, (pos, _ins) in numbering:
        for name, (lo, hi) in IMGT_BOUNDARIES.items():
            if lo <= pos <= hi:
                regions[name].append(aa)
                break
    return {name: "".join(res) for name, res in regions.items()}


def is_protein_antigen(agtypes):
    """Return True only when every slash-separated type is PROTEIN or PEPTIDE."""
    if not isinstance(agtypes, str):
        return False
    return all(t in PROTEIN_AG_TYPES for t in agtypes.split("/"))


def filter_rows(path, *, require_antigen=True, **_options):
    """Read the CSV and keep the rows this source considers usable, with its funnel stages."""
    df = pd.read_csv(path, low_memory=False)
    stages = [(len(df), "raw")]

    # Paired heavy+light structures, by default with a resolved protein/peptide antigen.
    df = df[df["type"].isin(PAIRED_TYPES)]
    stages.append((len(df), "paired-chain"))
    if require_antigen:
        df = df[df["agtypes"].apply(is_protein_antigen)]
        stages.append((len(df), "protein/peptide-antigen"))
        df = df[df["agresolvedseqs"].notna() & (df["agresolvedseqs"].astype(str).str.len() > 0)]
        stages.append((len(df), "with resolved antigen seq"))
    return df, stages, ("malformed-numbering", "missing/empty regions", "ambiguous-chain-break")


def finalize(entries, *, per_file, options, duplicates, funnel):
    """Carve validation out of the shipped split if asked, and return provenance metadata."""
    carve = carve_validation_split(entries, validation_size=options.validation_size,
                                   seed=options.seed)
    splits = {}
    for entry in entries:
        splits[entry.get("split")] = splits.get(entry.get("split"), 0) + 1
    species_known = sum(entry["species"] is not None for entry in entries)
    require_antigen = not options.allow_missing_antigen
    paths = options.input_file if isinstance(options.input_file, list) else [options.input_file]
    return {
        "dataset": "SAbDab2 antibody-antigen",
        "source": {"dataset": "SAbDab2", "file": str(paths[0])},
        "records": len(entries),
        "species": {"known_records": species_known,
                    "unknown_records": len(entries) - species_known},
        "splits": splits,
        "filtering": funnel,
        "preprocessing": {
            "paired_chains_only": True,
            "protein_or_peptide_antigen_only": require_antigen,
            "resolved_antigen_only": require_antigen,
            "numbering_scheme": "IMGT",
            "split": "source ab_ag_split",
            "validation_carve": carve,
            "limit": options.limit,
        },
    }


def row_parts(row, source_metadata=None):
    """This source's per-row values for the shared entry assembly (see __main__.build_entry)."""
    # Parse the numbering once and reuse it for both the regions and the entry (as OAS does).
    try:
        heavy_numbering = json.loads(row["VH_numbering_list"])
        light_numbering = json.loads(row["VL_numbering_list"])
    except (ValueError, TypeError) as exc:
        raise RejectedEntry("malformed-numbering") from exc

    return {
        "sequence_id": row["INSTANCE"],
        "chain_ids": {"heavy": optional_text(row["Hchain"]), "light": optional_text(row["Lchain"])},
        "heavy_regions": split_imgt_regions(heavy_numbering),
        "light_regions": split_imgt_regions(light_numbering),
        "heavy_aa": row["VH_numerable_seq"],
        "light_aa": row["VL_numerable_seq"],
        "heavy_numbering": heavy_numbering,
        "light_numbering": light_numbering,
        # This source currently exposes a species column with no populated values. Preserve
        # unknown as JSON null rather than silently conditioning downstream models as human.
        "species": optional_text(row["species"]),
        "isotype": None,
        "split": row["ab_ag_split"],
        "source": {
            "dataset": "SAbDab2",
            "pdb_id": optional_text(row["PDB_ID"]),
            "sabdab_id": optional_text(row["SABDAB_ID"]),
        },
    }


def finish_entry(entry, row):
    """Optional per-source finishing step, run AFTER region validation (see __main__)."""
    return add_antigen(entry, row)


def add_antigen(entry, row):
    """Attach the row's antigen to entry in place and return the same entry.

    Missing or blank resolved sequence gives antigen=None. Otherwise join source
    chains with X and retain their IDs and types. Reject real X residues because
    they would be ambiguous with this chain separator. Source-level antigen
    eligibility is decided by build_entries, before row conversion.
    """
    antigen = optional_text(row["agresolvedseqs"])
    if antigen is None:
        entry["antigen"] = None
    else:
        chains = antigen.split("/")
        if any("X" in chain for chain in chains):
            raise RejectedEntry("ambiguous-chain-break")
        entry["antigen"] = {
            "full_aa": "X".join(chains),
            "chains": row["agchains"].split("/") if isinstance(row["agchains"], str) else [],
            "type": row["agtypes"],
        }

    return entry


def carve_validation_split(entries, *, validation_size=0.15, seed=42):
    """Move whole antigen/CDR-H3 groups out of `train` to form a `validation` split."""
    train = [e for e in entries if e.get("split") == "train"]
    if validation_size <= 0 or not train:
        return {"requested_fraction": validation_size, "moved_records": 0, "moved_groups": 0}

    parent = {}

    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            key = parent[key] = parent[parent[key]]
        return key

    def keys(entry):
        antigen = (entry.get("antigen") or {}).get("full_aa")
        return [("h3", entry["heavy"]["cdr3"])] + ([("ag", antigen)] if antigen else [])

    groups = {}
    for entry in train:
        first, *rest = keys(entry)
        for key in rest:
            parent[root(key)] = root(first)
    for entry in train:
        groups.setdefault(root(keys(entry)[0]), []).append(entry)

    order = sorted(groups)
    random.Random(seed).shuffle(order)
    target, moved, used = round(len(train) * validation_size), 0, 0
    for group in (groups[r] for r in order):
        if moved >= target:
            break
        for entry in group:
            entry["split"] = "validation"
        moved, used = moved + len(group), used + 1

    # The whole point of grouping. Assert it rather than trust the union-find.
    def values(split, field):
        return {e["heavy"]["cdr3"] if field == "h3" else (e.get("antigen") or {}).get("full_aa")
                for e in train if e["split"] == split} - {None}

    for field in ("h3", "ag"):
        if values("train", field) & values("validation", field):
            raise ValueError(f"validation carve leaked a {field} value into both partitions; "
                             "the antigen/CDR-H3 grouping is wrong")
    return {"requested_fraction": validation_size, "seed": seed, "target_records": target,
            "moved_records": moved, "moved_groups": used, "train_records_before": len(train),
            "train_groups": len(groups)}


def add_source_arguments(parser):
    """Register the flags only this source understands; __main__.args imports them."""
    parser.add_argument("--allow_missing_antigen", action="store_true",
                        help="Keep paired antibodies whose antigen is absent or not a "
                             "protein/peptide. The antigen filter currently discards about "
                             "two thirds of the paired rows, which the *_ag_lconly configs "
                             "do not use; those entries carry antigen=null.")
    parser.add_argument("--validation_size", type=float, default=0.0,
                        help="Fraction of the TRAIN partition to relabel as validation, "
                             "holding out whole antigen/CDR-H3 groups. SAbDab2 ships no "
                             "validation split, so 0.0 keeps the historical behaviour of "
                             "selecting on test. The test partition is never touched.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the validation carve-out's group shuffle.")
    return parser


if __name__ == "__main__":
    from cdrseqgen.datasets.preprocess.__main__ import main

    raise SystemExit(main(source_name="sabdab2"))
