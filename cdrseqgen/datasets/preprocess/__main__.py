"""Main data preprocessing pipeline entry point for. Currently supports: OAS (paired), SAbDab2

Workflow:
* main
  * args                                   io flags + the source's own imported flags
  * build_corpus
    * read_oas_metadata                    (OAS only, once per CSV)
    * build_entries                        source filters, then the row limit
      * build_entry
        * build_regions / split_imgt_regions   (OAS / SAbDab2)
        * build_chain
        * valid_regions
        * add_antigen                      (SAbDab2 only)
      * print_funnel                       the funnel this corpus records
    * add_split_field                      (OAS only, after merging/deduplication)
      * cluster_by_cdr3
      * assign_cluster_splits
    * carve_validation_split               (SAbDab2 only, optional)
  * write_processed
  * write_processed_metadata

General policy:
  * __main__.py : Defines the core pipeline, corpus registry, and CLI entry point. 
  * common.py   : Stores functions for the main pipeline, compatible across sources.
  * [corpus].py : Stores functions specific to one source, including its own CLI flags. 
"""
import argparse
import ast
import json
import sys
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from cdrseqgen.datasets.preprocess import oas_paired, sabdab2
from cdrseqgen.datasets.preprocess.common import (
    RejectedEntry,
    add_io_args,
    assemble_entry,
    build_chain,
    print_funnel,
    optional_text,
    valid_regions,
    write_processed,
    write_processed_metadata,
)


@dataclass(frozen=True)
class CorpusSource:
    """Where a source's raw input lives and where its corpus goes, for the CLI defaults."""
    name: str
    input_default: str
    output_default: str
    input_help: str
    multiple: bool = False      # whether --input_file accepts several paths


SOURCES = {s.name: s for s in (
    CorpusSource(
        "oas_paired",
        "data/raw/oas/1287169_1_Paired_All.csv", "data/processed/processed_oas_data.json",
        "One or more OAS paired CSVs. Several files become ONE corpus: they are concatenated "
        "first and the split is assigned once over the merged set.", multiple=True),
    CorpusSource(
        "sabdab2",
        "data/raw/sabdab2/splits_final/abag_split.csv",
        "data/processed/processed_sabdab2_data.json", "Path to SAbDab2's abag_split.csv"),
)}


def args(argv=None, *, source_name=None):
    """Arguments parser"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if source_name is None:
        pre = argparse.ArgumentParser(prog="preprocess", add_help=False)
        pre.add_argument("--source", choices=list(SOURCES))
        chosen, argv = pre.parse_known_args(argv)
        if chosen.source is None:
            print(f"usage: preprocess --source {{{','.join(SOURCES)}}} [flags]")
            return None
        source_name = chosen.source

    source = SOURCES[source_name]
    parser = argparse.ArgumentParser(prog=f"preprocess --source {source.name}",
                                     description=f"Build the {source.name} corpus.")
    add_io_args(parser, input_default=source.input_default,
                output_default=source.output_default, input_help=source.input_help,
                multiple=source.multiple)

    if source.name == "oas_paired":
        oas_paired.add_source_arguments(parser)
    elif source.name == "sabdab2":
        sabdab2.add_source_arguments(parser)
    else: 
        raise ValueError(f"unknown source {source.name!r}; add it to the registry and import its flags")

    parsed = parser.parse_args(argv)
    parsed.source = source.name
    return parsed


def build_entry(row, source, source_metadata=None):
    """Convert one filtered row into the entry every source emits."""
    if source == "oas_paired":
        # ANARCI numbering is a stringified dict in OAS. Narrow except: only swallow parse
        # failures (incl. NaN passed to literal_eval), NOT e.g. a KeyError from a renamed column.
        try:
            heavy_numbering = ast.literal_eval(row['ANARCI_numbering_heavy'])
            light_numbering = ast.literal_eval(row['ANARCI_numbering_light'])
        except (ValueError, SyntaxError, TypeError) as exc:
            raise RejectedEntry("malformed-numbering") from exc

        heavy_regions = oas_paired.build_regions(row, "heavy")
        light_regions = oas_paired.build_regions(row, "light")
        heavy_aa = row['sequence_alignment_aa_heavy']
        light_aa = row['sequence_alignment_aa_light']

        sequence_id = row['sequence_id_heavy']
        chain_ids = {"heavy": row['sequence_id_heavy'], "light": row['sequence_id_light']}
        species = source_metadata["species"]
        isotype = row['Isotype_heavy']
        split = None
        provenance = {
            "dataset": source_metadata["dataset"],
            "run": source_metadata["run"],
            # The donor, not the sequencing run: one subject spans many runs, so a
            # donor-held-out split can only be built if this travels with the entry.
            "subject": source_metadata.get("subject"),
            "b_type": source_metadata.get("b_type"),
        }
    else:
        # Parse the numbering once and reuse it for both the regions and the entry (as OAS does).
        try:
            heavy_numbering = json.loads(row["VH_numbering_list"])
            light_numbering = json.loads(row["VL_numbering_list"])
        except (ValueError, TypeError) as exc:
            raise RejectedEntry("malformed-numbering") from exc

        heavy_regions = sabdab2.split_imgt_regions(heavy_numbering)
        light_regions = sabdab2.split_imgt_regions(light_numbering)
        heavy_aa = row["VH_numerable_seq"]
        light_aa = row["VL_numerable_seq"]

        sequence_id = row["INSTANCE"]
        chain_ids = {"heavy": optional_text(row["Hchain"]), "light": optional_text(row["Lchain"])}
        # This source currently exposes a species column with no populated values. Preserve
        # unknown as JSON null rather than silently conditioning downstream models as human.
        species = optional_text(row["species"])
        isotype = None
        split = row["ab_ag_split"]
        provenance = {
            "dataset": "SAbDab2",
            "pdb_id": optional_text(row["PDB_ID"]),
            "sabdab_id": optional_text(row["SABDAB_ID"]),
        }

    heavy = build_chain(heavy_regions, heavy_aa, heavy_numbering)
    light = build_chain(light_regions, light_aa, light_numbering)
    if not (valid_regions(heavy_regions) and valid_regions(light_regions)):
        raise RejectedEntry("missing/empty regions")

    entry = assemble_entry(sequence_id, chain_ids=chain_ids, heavy=heavy, light=light,
                           species=species, isotype=isotype, source=provenance, split=split)
    if source == "sabdab2":
        sabdab2.add_antigen(entry, row)
    return entry


def build_entries(path, source, limit=None, *, source_metadata=None, require_antigen=True):
    """Return (entries, funnel) from one raw file, before deduplication or splitting.
    
    Runs build_entry on each row and keeps valid entries.
    """
    if source == "oas_paired":
        # The first row is metadata, not a header.
        df = pd.read_csv(path, skiprows=1, low_memory=False)
        stages = [(len(df), "raw")]

        # Productive pairs only -- no stop codons or frameshifts.
        df = df[(df['productive_heavy'] == 'T') & (df['productive_light'] == 'T')]
        stages.append((len(df), "productive"))
        reasons = ("malformed-numbering", "missing/empty regions")
    else:
        df = pd.read_csv(path, low_memory=False)
        stages = [(len(df), "raw")]

        # Paired heavy+light structures, by default with a resolved protein/peptide antigen.
        df = df[df["type"].isin(sabdab2.PAIRED_TYPES)]
        stages.append((len(df), "paired-chain"))
        if require_antigen:
            df = df[df["agtypes"].apply(sabdab2.is_protein_antigen)]
            stages.append((len(df), "protein/peptide-antigen"))
            df = df[df["agresolvedseqs"].notna()
                    & (df["agresolvedseqs"].astype(str).str.len() > 0)]
            stages.append((len(df), "with resolved antigen seq"))
        reasons = ("malformed-numbering", "missing/empty regions", "ambiguous-chain-break")

    if limit:
        df = df.head(limit)

    kept, skipped = [], {reason: 0 for reason in reasons}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            kept.append(build_entry(row, source, source_metadata))
        except RejectedEntry as exc:
            if exc.reason not in skipped:
                raise ValueError(f"row rejected for unregistered reason {exc.reason!r}; add it "
                                 "to this source's reason list so the funnel counts it") from exc
            skipped[exc.reason] += 1
    return kept, print_funnel(stages, skipped, len(kept))


def build_corpus(options):
    """Return (entries, metadata) ready for writing, from one parsed command line.
    
    Runs build_entries on each input file and merges the results. 
    """
    source = options.source
    paths = options.input_file if isinstance(options.input_file, list) else [options.input_file]

    entries, per_file, funnel = [], [], None
    seen, duplicates = set(), 0

    for path in paths:
        source_metadata = None
        if source == "oas_paired":
            source_metadata = oas_paired.read_oas_metadata(path)
            print(f"\n=== {path} (run {source_metadata['run']}, "
                  f"subject {source_metadata.get('subject')}) ===")

        built, funnel = build_entries(
            path, source, limit=options.limit, source_metadata=source_metadata,
            require_antigen=not getattr(options, "allow_missing_antigen", False))

        kept = 0
        for entry in built:
            if source == "oas_paired" and options.dedup == "exact":
                # The biological identity, not sequence_id: ids are assigned per run and say
                # nothing about whether two rows are the same antibody.
                key = tuple((entry.get(chain) or {}).get("full_aa")
                            for chain in ("heavy", "light"))
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
            entries.append(entry)
            kept += 1

        if source == "oas_paired":
            per_file.append({**source_metadata, "path": str(path), "entries": kept,
                             "filtering": funnel})
            print(f"    contributed {kept} entr{'y' if kept == 1 else 'ies'}")

    if source == "oas_paired":
        donors = sorted({str(m.get("subject")) for m in per_file})
        species = sorted({m["species"] for m in per_file})
        # Calculate longest loop, used for the causal model's max answer length
        longest = {}
        for chain in ("heavy", "light"):
            for region in ("cdr1", "cdr2", "cdr3"):
                longest[f"{chain[0]}{region[-1]}"] = max(
                    (len(e[chain][region]) for e in entries if e.get(chain)), default=0)

        print(f"\nCorpus: {len(entries)} entries from {len(paths)} file(s), "
              f"{len(donors)} donor(s): {', '.join(donors)}")
        if duplicates:
            print(f"        dropped {duplicates} duplicate heavy+light pair(s) "
                  f"(--dedup none keeps them)")
        if len(species) > 1:
            print(f"        WARNING: mixed species in one corpus: {', '.join(species)}")
        print("        longest loop: " + "  ".join(f"{k} {v}" for k, v in sorted(longest.items())))

        oas_paired.add_split_field(entries, options.split, options.test_size, options.seed,
                                   options.cluster_identity,
                                   validation_size=options.validation_size)
        metadata = {
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
                "cluster_identity": (options.cluster_identity
                                     if options.split == "cluster" else None),
                "seed": options.seed,
                "limit": options.limit,
                "dedup": options.dedup,
                "duplicates_dropped": duplicates,
            },
        }
    else:
        carve = sabdab2.carve_validation_split(entries, validation_size=options.validation_size,
                                               seed=options.seed)
        splits = {}
        for entry in entries:
            splits[entry.get("split")] = splits.get(entry.get("split"), 0) + 1
        species_known = sum(entry["species"] is not None for entry in entries)
        require_antigen = not options.allow_missing_antigen
        metadata = {
            "dataset": "SAbDab2 antibody-antigen",
            "source": {"dataset": "SAbDab2", "file": str(paths[0])},
            "records": len(entries),
            "species": {
                "known_records": species_known,
                "unknown_records": len(entries) - species_known,
            },
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
    return entries, metadata


def main(argv=None, *, source_name=None):
    """Build and write one corpus. Without source_name, read it from ``--source``."""
    options = args(argv, source_name=source_name)
    if options is None:
        return 2
    data, metadata = build_corpus(options)
    write_processed(data, options.output_file)
    write_processed_metadata(options.output_file, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
