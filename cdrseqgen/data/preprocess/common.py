"""Shared scaffolding for the data preprocessing scripts (csv -> json)

Entities:
  * REGIONS         -- the 7 IMGT region names, in order
  * CDRS            -- the 3 CDR keys
  * valid_regions   -- filter out invalid regions (empty, NaN, etc)
  * build_chain     -- assemble full aa sequence and H/L chain numbering with 7 regions dict
  * add_io_args     -- --input_file / --output_file / --limit
  * print_funnel    -- one-line "N raw -> ... -> N kept (skipped ...)" summary
  * write_processed -- create the output tree and write the JSON

Notes:
  * Each script keeps its own extraction step, because the sources differ in kind: OAS ships
    ready-made `*_aa_heavy` columns, while SAbDab2 must be folded up from per-residue numbering.
  * dataset/data/ is gitignored, so the output tree never exists on a fresh clone — write_processed
    creates it rather than failing.
"""
import hashlib
import json
import os
from pathlib import Path
from cdrseqgen.artifacts import processed_metadata_path

REGIONS = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]
CDRS = ("cdr1", "cdr2", "cdr3")


def valid_regions(region_dict):
    """Boolean to filter out invalid regions (empty, NaN, etc)"""
    if not all(isinstance(region_dict.get(r), str) for r in REGIONS):
        return False
    return all(region_dict[cdr] for cdr in CDRS)


def build_chain(regions, full_aa, numbering):
    """Takes the region dict (the 7 regions) and add the full aa sequence and the corresponding
    IMGT numbering (heavy/light) to the dict"""
    return {**regions, "full_aa": full_aa, "imgt_numbering": numbering}


def optional_text(value):
    """Return a stripped string or None for pandas/JSON missing values."""
    if value is None:
        return None
    try:
        if value != value:  # NaN is the common missing value unequal to itself.
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def add_io_args(parser, *, input_default, output_default, input_help, multiple=False):
    """Attach common io args.

    `multiple` makes --input_file variadic, for a preprocessor that builds ONE corpus from several
    source files (OAS paired, where a corpus is several donors' repertoires). It stays off by
    default so single-source preprocessors keep a plain string.
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


def print_funnel(stages, skipped, kept):
    """One-line row funnel: 'Rows: 100 raw -> 80 productive -> 78 kept (skipped 1 x, 1 y).'"""
    chain = " -> ".join(f"{n} {label}" for n, label in stages)
    detail = ", ".join(f"{n} {label}" for label, n in skipped.items())
    print(f"Rows: {chain} -> {kept} kept" + (f" (skipped {detail})." if detail else "."))


def write_processed(data, output_file):
    """Write the processed entries as JSON, creating the output tree if it doesn't exist"""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f)
    print(f"Wrote {len(data)} entries to {output_file}")


def write_processed_metadata(output_file, metadata):
    """Write a provenance sidecar for a processed JSON file."""
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
