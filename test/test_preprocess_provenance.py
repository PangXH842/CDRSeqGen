import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cdrseqgen.data.preprocess.common import optional_text, write_processed, write_processed_metadata
from cdrseqgen.data.preprocess.oas_paired import build_entries, read_oas_metadata


class PreprocessProvenanceTests(unittest.TestCase):
    def test_reads_oas_source_metadata(self):
        metadata = {
            "Run": 1287169,
            "Species": "Human",
            "Link": "https://doi.org/10.1038/s41586-022-05371-z",
            "Author": "Jaffe et al., 2022",
            "Subject": "Donor-4",
        }
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv") as handle:
            csv.writer(handle).writerow([json.dumps(metadata)])
            handle.flush()
            parsed = read_oas_metadata(handle.name)

        self.assertEqual(parsed["run"], 1287169)
        self.assertEqual(parsed["species"], "human")
        self.assertEqual(parsed["subject"], "Donor-4")

    def test_oas_entries_keep_species_source_and_both_chain_ids(self):
        row = {
            "productive_heavy": "T",
            "productive_light": "T",
            "ANARCI_numbering_heavy": "{}",
            "ANARCI_numbering_light": "{}",
            "sequence_id_heavy": "heavy-1",
            "sequence_id_light": "light-1",
            "Isotype_heavy": "IGHM",
            "sequence_alignment_aa_heavy": "HSEQ",
            "sequence_alignment_aa_light": "LSEQ",
        }
        for chain in ("heavy", "light"):
            for region in ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"):
                row[f"{region}_aa_{chain}"] = region[0].upper()

        source = {"dataset": "OAS", "run": 7, "species": "human"}
        with patch("cdrseqgen.data.preprocess.oas_paired.pd.read_csv", return_value=pd.DataFrame([row])):
            entries = build_entries("unused.csv", source_metadata=source)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["species"], "human")
        self.assertEqual(entries[0]["source"], {"dataset": "OAS", "run": 7})
        self.assertEqual(
            entries[0]["chain_sequence_ids"], {"heavy": "heavy-1", "light": "light-1"}
        )

    def test_sidecar_hashes_the_processed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "processed.json"
            write_processed([{"sequence_id": "x"}], output)
            sidecar = write_processed_metadata(output, {"dataset": "test", "records": 1})
            payload = json.loads(sidecar.read_text())

            expected = hashlib.sha256(output.read_bytes()).hexdigest()
            self.assertEqual(payload["processed_file_sha256"], expected)
            self.assertEqual(payload["schema_version"], "cdrseqgen.processed_metadata.v1")

    def test_optional_text_preserves_unknown_as_none(self):
        self.assertIsNone(optional_text(float("nan")))
        self.assertIsNone(optional_text("  "))
        self.assertEqual(optional_text(" human "), "human")


if __name__ == "__main__":
    unittest.main()
