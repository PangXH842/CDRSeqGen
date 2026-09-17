"""Functions shared across every preprocessing source.

Entities:
  * REGIONS, CDRS            -- the seven IMGT region keys, and the three that must be nonempty
  * RejectedEntry            -- an expected row rejection, carrying its funnel counter key
  * assemble_entry           -- the entry schema: every top-level key, in one place
  * valid_regions            -- check region types and nonempty CDRs
  * build_chain              -- combine region strings, full sequence and numbering
  * optional_text            -- normalize optional source fields to text or None
  * add_io_args              -- input/output paths and the post-filter row limit
  * print_funnel             -- report and return what each filter stage dropped
  * write_processed          -- create parent directories and write the entry list
  * write_processed_metadata -- hash the processed file and write its provenance sidecar

Notes:
  * Validation checks region presence/types and nonempty CDRs, not amino-acid alphabets,
    numbering structure, or agreement between regions and the full sequence.
  * Write the processed JSON before its metadata: the sidecar hashes the bytes on disk.
  * REGIONS is the one region key set; a source deriving regions its own way must match it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from tqdm import tqdm

from cdrseqgen.artifacts import processed_metadata_path

REGIONS = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]
CDRS = ("cdr1", "cdr2", "cdr3")


class RejectedEntry(ValueError):
    """Catch invalid entry rows"""
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def valid_regions(region_dict):
    """Check if all regions are valid (without missing keys or non-string values).

    Note: Empty framework strings are allowed. Residue characters and numbering are not checked here.
    """
    if not all(isinstance(region_dict.get(r), str) for r in REGIONS):
        return False
    return all(region_dict[cdr] for cdr in CDRS)


def build_chain(regions, full_aa, numbering):
    """Return a new chain dictionary containing regions, full_aa and imgt_numbering."""
    return {**regions, "full_aa": full_aa, "imgt_numbering": numbering}


def optional_text(value):
    """Return stripped text, or None for missing or blank values."""
    if value is None:
        return None
    try:
        if value != value:  # == NaN 
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def add_io_args(parser, *, input_default, output_default, input_help, multiple=False):
    """Add --input_file, --output_file and --limit to an argparse parser; return it.

    input_default, output_default and input_help define the source-specific paths/help.
    With multiple=True, --input_file accepts one or more paths and its default is a list;
    otherwise it is a single value. The caller applies --limit after its source filters.
    """
    if multiple:
        default = [input_default] if isinstance(input_default, str) else list(input_default)
        parser.add_argument("--input_file", nargs="+", default=default, help=input_help)
    else:
        parser.add_argument("--input_file", default=input_default, help=input_help)
    parser.add_argument("--output_file", default=output_default, help="Path to the output JSON file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows after filtering (for sanity-checking).")
    return parser


def assemble_entry(sequence_id, *, chain_ids, heavy=None, light=None, species=None,
                   source, antigen=None, isotype=None, split=None):
    """Build the entry every source emits."""
    if heavy is None and light is None:
        raise RejectedEntry("no-chain")
    return {
        "sequence_id": sequence_id,
        "chain_sequence_ids": chain_ids,
        "heavy": heavy,
        "light": light,
        "antigen": antigen,
        "species": species,
        "isotype": isotype,
        "source": source,
        "split": split,
    }


def print_funnel(stages, skipped, kept):
    """Print filtering stages and the kept count, and return them as a JSON-ready dict."""
    chain = " -> ".join(f"{n} {label}" for n, label in stages)
    detail = ", ".join(f"{n} {label}" for label, n in skipped.items())
    print(f"Rows: {chain} -> {kept} kept" + (f" (skipped {detail})." if detail else "."))
    return {"stages": [{"label": label, "rows": int(n)} for n, label in stages],
            "rejected_rows": {reason: int(n) for reason, n in skipped.items()},
            "kept": int(kept)}


def write_processed(data, output_file):
    """Write entries as JSON to output_file and print their count."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f)
    print(f"Wrote {len(data)} entries to {output_file}")


def write_processed_metadata(output_file, metadata):
    """Hash an existing processed JSON, write its metadata sidecar, and return its Path."""
    output_path = Path(output_file)
    digest = hashlib.sha256()
    with output_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    payload = {
        "schema_version": "cdrseqgen.processed_metadata.v1",
        "processed_file": output_path.name,
        "processed_file_sha256": digest.hexdigest(),
        **metadata,
    }
    metadata_path = processed_metadata_path(output_path)
    with metadata_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote preprocessing metadata to {metadata_path}")
    return metadata_path
