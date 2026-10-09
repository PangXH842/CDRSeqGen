#!/usr/bin/env python3
"""Export the IMGT flank regions of a manifest's frameworks, for generators that need regions, not strings.

The manifest carries the heavy chain as `pt1 + native + pt2`. IgCraft encodes IMGT regions (each padded to
its own capacity), so it needs pt1 split into FWR1/CDR1/FWR2/CDR2/FWR3 and pt2 as FWR4. Numbering the strings
at run time would add an ANARCI dependency to inference, and reading the corpus by `source_index` would let
an adapter reach the answer. This writes the split once, from the processed corpus the manifest was built
from, keyed by `example_id` and holding flanks only: no CDR3 and nothing derived from it. Every record is
checked to rebuild the manifest's `pt1` and `pt2`, so an adapter can re-verify them without the corpus.

  python benchmark/export_regions.py --manifest benchmark/manifests/X.json --corpus data/processed/Y.json \
      --output benchmark/manifests/X_regions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark._common import atomic_write_json, sha256_file

SCHEMA_VERSION = 'cdrseqgen.flank_regions.v1'
PT1_REGIONS = ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3')
FLANK_REGIONS = PT1_REGIONS + ('fwr4',)


def flank_regions(record, regions):
    """The flank regions as a dict, after proving they rebuild the record's pt1 and pt2."""
    if ''.join(regions[name] for name in PT1_REGIONS) != record['pt1'] or regions['fwr4'] != record['pt2']:
        raise ValueError(f"{record['example_id']} flank regions do not rebuild pt1/pt2")
    return {name: regions[name] for name in FLANK_REGIONS}


def build(manifest, corpus, corpus_sha256):
    """The sidecar payload for `manifest`, built from the corpus entries its records point to."""
    if corpus_sha256 != manifest['source']['processed_file_sha256']:
        raise ValueError('Corpus differs from the one the manifest was built from')
    records = {}
    for record in manifest['records']:
        entry = corpus[record['source_index']]
        if entry['sequence_id'] != record['sequence_id']:
            raise ValueError(f"{record['example_id']} points at a different corpus entry")
        records[record['example_id']] = flank_regions(record, entry['heavy'])
    return dict(schema_version=SCHEMA_VERSION, benchmark_id=manifest['benchmark_id'],
                source=dict(processed_file=manifest['source']['processed_file'], processed_file_sha256=corpus_sha256),
                regions=list(FLANK_REGIONS), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'output already exists: {args.output}')
    manifest = json.loads(args.manifest.read_text())
    payload = build(manifest, json.loads(args.corpus.read_text()), sha256_file(args.corpus))
    atomic_write_json(args.output, payload)
    args.output.with_name(args.output.name + '.sha256').write_text(f'{sha256_file(args.output)}  {args.output.name}\n')
    print(f'Wrote {len(payload["records"])} frameworks to {args.output}')


if __name__ == '__main__':
    main()
