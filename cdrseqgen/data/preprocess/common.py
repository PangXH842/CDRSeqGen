"""Shared scaffolding for the data preprocessing scripts (csv -> json)

Entities:
  * REGIONS         -- the 7 IMGT region names, in order
  * CDRS            -- the 3 CDR keys 
  * valid_regions   -- reject a chain with a malformed region or an empty CDR
  * build_chain     -- assemble one chain's entry block (regions + full_aa + imgt_numbering)
  * add_io_args     -- the --input_file / --output_file / --limit CLI trio
  * print_funnel    -- one-line "N raw -> ... -> N kept (skipped ...)" summary
  * write_processed -- create the output tree and write the JSON

Notes:
  * Each script keeps its own extraction step, because the sources differ in kind: OAS ships
    ready-made `*_aa_heavy` columns, while SAbDab2 must be folded up from per-residue numbering.
  * dataset/data/ is gitignored, so the output tree never exists on a fresh clone — write_processed
    creates it rather than failing.
"""
import os
import json

REGIONS = ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"]
CDRS = ("cdr1", "cdr2", "cdr3")


def valid_regions(region_dict):
    """Boolean to filter out invalid regions (empty, NaN, etc)"""
    if not all(isinstance(region_dict.get(r), str) for r in REGIONS):
        return False
    return all(region_dict[cdr] for cdr in CDRS)


def build_chain(regions, full_aa, numbering):
    """One chain's entry block: the 7 regions plus the full sequence and its IMGT numbering"""
    return {**regions, "full_aa": full_aa, "imgt_numbering": numbering}


def add_io_args(parser, *, input_default, output_default, input_help):
    """Attach common io args"""
    parser.add_argument("--input_file", default=input_default, help=input_help)
    parser.add_argument("--output_file", default=output_default, help="Path to the output JSON file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows after filtering (for sanity-checking).")
    return parser


def print_funnel(stages, skipped, kept):
    """One-line row funnel: 'Rows: 100 raw -> 80 productive -> 78 kept (skipped 1 x, 1 y).'

    `stages` is [(count, label), ...] from the raw row count down through each dataframe filter;
    `skipped` maps a reason to how many rows the per-row loop dropped for it. Printed by every run
    so an unexpected collapse is visible instead of silent.
    """
    chain = " -> ".join(f"{n} {label}" for n, label in stages)
    detail = ", ".join(f"{n} {label}" for label, n in skipped.items())
    print(f"Rows: {chain} -> {kept} kept" + (f" (skipped {detail})." if detail else "."))


def write_processed(data, output_file):
    """Write the processed entries as JSON, creating the output tree if it doesn't exist"""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f)
    print(f"Wrote {len(data)} entries to {output_file}")
