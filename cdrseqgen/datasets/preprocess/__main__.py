"""Main data preprocessing pipeline entry point. Currently supports: OAS (paired), SAbDab2

Nothing here names a source: a corpus is reached only through the hooks marked [*] below.
Adding one is a SOURCES entry plus a module implementing the three required hooks -- a missing
one raises at its first use rather than silently skipping a step someone forgot to branch on.

Workflow:                                    [*] = source hook   (opt) = optional hook
* main
  * args                                     io flags + [*] add_source_arguments (opt)
  * build_corpus
    * [*] read_metadata            (opt)     per-file provenance; OAS ships one row per CSV
    * build_entries
      * [*] filter_rows                      source row filters, funnel stages, reason list
      * build_entry                          (--limit applies after the filters, before this)
        * [*] row_parts                      regions, numbering, ids, species, provenance
        * valid_regions / build_chain / assemble_entry
        * [*] finish_entry         (opt)     after validation, so reasons keep precedence;
                                             SAbDab2 attaches its antigen here
      * print_funnel                         the funnel this corpus records
    * [*] dedup_key                (opt)     cross-file identity; OAS merges several donors
    * [*] finalize                           split assignment + the metadata sidecar
        (OAS)     add_split_field -> cluster_by_cdr3, assign_cluster_splits
        (SAbDab2) carve_validation_split
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
    module: object              # the module implementing this source's protocol
    input_default: str
    output_default: str
    input_help: str
    multiple: bool = False      # whether --input_file accepts several paths


SOURCES = {s.name: s for s in (
    CorpusSource(
        "oas_paired", oas_paired,
        "data/raw/oas/1287169_1_Paired_All.csv", "data/processed/processed_oas_data.json",
        "One or more OAS paired CSVs. Several files become ONE corpus: they are concatenated "
        "first and the split is assigned once over the merged set.", multiple=True),
    CorpusSource(
        "sabdab2", sabdab2,
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

    # Optional, like the other hooks: a source with no extra flags need not define it.
    add_flags = getattr(source.module, "add_source_arguments", None)
    if add_flags is not None:
        add_flags(parser)

    parsed = parser.parse_args(argv)
    parsed.source = source.name
    return parsed


def build_entry(row, source, source_metadata=None):
    """Convert one filtered row into the entry every source emits.

    The source supplies only what it alone can know -- how to reach its regions, numbering,
    ids and provenance -- through `row_parts`. Validation, chain assembly and the schema are
    shared, so a new source implements one function rather than a branch here.
    """
    module = SOURCES[source].module
    parts = module.row_parts(row, source_metadata)
    heavy_regions, light_regions = parts["heavy_regions"], parts["light_regions"]
    if not (valid_regions(heavy_regions) and valid_regions(light_regions)):
        raise RejectedEntry("missing/empty regions")

    entry = assemble_entry(
        parts["sequence_id"], chain_ids=parts["chain_ids"],
        heavy=build_chain(heavy_regions, parts["heavy_aa"], parts["heavy_numbering"]),
        light=build_chain(light_regions, parts["light_aa"], parts["light_numbering"]),
        species=parts["species"], isotype=parts["isotype"],
        source=parts["source"], split=parts["split"])
    # Runs after region validation so rejection reasons keep their precedence.
    finish = getattr(module, "finish_entry", None)
    return finish(entry, row) if finish else entry


def build_entries(path, source, limit=None, *, source_metadata=None, require_antigen=True):
    """Return (entries, funnel) from one raw file, before deduplication or splitting.

    The source owns reading and filtering its own table -- productive pairs for OAS,
    paired-chain types and optionally a resolved antigen for SAbDab2 -- and declares which
    rejection reasons its row conversion can raise. `limit` applies only after those filters,
    so a limited run is a prefix of the real corpus rather than a prefix of the raw file.

    Rejections are counted against the declared reasons, so the funnel reports a zero rather
    than omitting an untriggered reason, and an unregistered reason raises instead of
    vanishing. Missing columns and other schema errors propagate.
    """
    df, stages, reasons = SOURCES[source].module.filter_rows(
        path, require_antigen=require_antigen)
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

    The shape is the same for every source: read each input, convert its rows, optionally
    deduplicate across the merged set, then let the source finalise. What differs is delegated
    -- `read_metadata` for per-file provenance, `dedup_key` for cross-file identity, and
    `finalize` for split assignment and the metadata sidecar. The first two are optional; a
    source that ships one complete file and its own split implements neither.
    """
    source = options.source
    module = SOURCES[source].module
    paths = options.input_file if isinstance(options.input_file, list) else [options.input_file]
    read_metadata = getattr(module, "read_metadata", None)
    dedup_key = getattr(module, "dedup_key", None)

    entries, per_file, funnel = [], [], None
    seen, duplicates = set(), 0

    for path in paths:
        source_metadata = read_metadata(path) if read_metadata else None
        built, funnel = build_entries(
            path, source, limit=options.limit, source_metadata=source_metadata,
            require_antigen=not getattr(options, "allow_missing_antigen", False))

        kept = 0
        for entry in built:
            key = dedup_key(entry, options) if dedup_key else None
            if key is not None:
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
            entries.append(entry)
            kept += 1

        if source_metadata is not None:
            per_file.append({**source_metadata, "path": str(path), "entries": kept,
                             "filtering": funnel})
            print(f"    contributed {kept} entr{'y' if kept == 1 else 'ies'}")

    return entries, module.finalize(entries, per_file=per_file, options=options,
                                    duplicates=duplicates, funnel=funnel)


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
