"""Shared validation, chain assembly, CLI arguments and output for preprocessing.

Entities:
  * REGIONS                  -- the seven IMGT region keys, in sequence order
  * CDRS                     -- the three CDR keys that must have nonempty strings
  * valid_regions            -- check region types and nonempty CDRs
  * build_chain              -- combine region strings, full sequence and numbering
  * optional_text            -- normalize optional source fields to text or None
  * add_io_args              -- add input/output paths and an optional row limit
  * print_funnel             -- report filtering stages and skipped-row counts
  * write_processed          -- create parent directories and write the entry list
  * write_processed_metadata -- hash the processed file and write its provenance sidecar

Notes:
  * Source-specific extraction stays in oas_paired.py and sabdab2.py. Both emit heavy/light
    chain objects with the same region keys; numbering retains each source's representation.
  * Validation checks region presence/types and nonempty CDRs, not amino-acid alphabets,
    numbering structure, or agreement between regions and the full sequence.
  * Write the processed JSON before its metadata: the sidecar hashes the bytes on disk.
"""
import hashlib
import json
import os
from pathlib import Path
from cdrseqgen.artifacts import processed_metadata_path

REGIONS = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]
CDRS = ("cdr1", "cdr2", "cdr3")


def valid_regions(region_dict):
    """Return whether all seven regions are strings and all three CDRs are nonempty.

    Missing keys and non-string values, including NaN, fail validation. Empty framework
    strings are allowed. Residue characters and numbering are not checked here.
    """
    if not all(isinstance(region_dict.get(r), str) for r in REGIONS):
        return False
    return all(region_dict[cdr] for cdr in CDRS)


def build_chain(regions, full_aa, numbering):
    """Return a new chain dictionary containing regions, full_aa and imgt_numbering.

    The supplied region dictionary is not modified. Full sequence and numbering are stored
    as supplied, without validation or conversion of the source-specific numbering format.
    """
    return {**regions, "full_aa": full_aa, "imgt_numbering": numbering}


def optional_text(value):
    """Return stripped text, or None for missing or blank values.

    None, NaN and pandas missing scalars become None; other values are converted with str.
    Literal strings such as "nan" are retained rather than interpreted as missing values.
    """
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


def print_funnel(stages, skipped, kept):
    """Print filtering stages, the final kept count and optional skipped-row counts.

    stages is an ordered iterable of (count, label) pairs. skipped maps reasons to counts;
    kept is the final number of entries. Counts are supplied by the caller, not recomputed.
    """
    chain = " -> ".join(f"{n} {label}" for n, label in stages)
    detail = ", ".join(f"{n} {label}" for label, n in skipped.items())
    print(f"Rows: {chain} -> {kept} kept" + (f" (skipped {detail})." if detail else "."))


def write_processed(data, output_file):
    """Write entries as JSON to output_file and print their count.

    Create missing parent directories and overwrite an existing file. Provenance is written
    separately by write_processed_metadata after this output exists.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f)
    print(f"Wrote {len(data)} entries to {output_file}")


def write_processed_metadata(output_file, metadata):
    """Hash an existing processed JSON, write its metadata sidecar, and return its Path.

    The sidecar path comes from processed_metadata_path. Its defaults record a schema
    version, the processed filename and SHA-256 of its bytes. Caller-supplied metadata adds
    source/filter/split details and takes precedence if it repeats a default key. Existing
    sidecar content is overwritten; output_file must already exist.
    """
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
