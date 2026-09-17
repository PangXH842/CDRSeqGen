import csv
import re
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cdrseqgen.datasets.preprocess.common import optional_text, write_processed, write_processed_metadata
from cdrseqgen.datasets.preprocess.oas_paired import build_corpus, build_entries, read_oas_metadata


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

        source = {"dataset": "OAS", "run": 7, "species": "human", "subject": "Donor-4"}
        with patch("cdrseqgen.datasets.preprocess.oas_paired.pd.read_csv", return_value=pd.DataFrame([row])):
            entries = build_entries("unused.csv", source_metadata=source)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["species"], "human")
        # The donor rides on the entry, not just on the sidecar: a corpus merged from several
        # donors' files can only be split donor-wise if every entry knows whose it is.
        self.assertEqual(entries[0]["source"],
                         {"dataset": "OAS", "run": 7, "subject": "Donor-4"})
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


class DownloaderPathTests(unittest.TestCase):
    """The downloaders resolve the repo root from their own location; the depth must be right.

    `../..` from cdrseqgen/datasets/preprocess/ reaches <repo>/cdrseqgen, not <repo>. That shipped, and
    silently wrote 154 MB into <repo>/cdrseqgen/data/raw/ — a path .gitignore's anchored
    `data/` pattern does not cover, so the blobs sat staged-able in the working tree.
    """

    SCRIPTS = ("download_oas_paired.sh", "download_sabdab2.sh")

    def test_the_scripts_resolve_the_actual_repo_root(self):
        # Match the ROOT= line and read its relative component, rather than substring-testing:
        # "/../../.." contains "/../..", so a naive assertNotIn can never fail.
        pattern = re.compile(r'ROOT="\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/(\S+)" && pwd\)"')
        here = Path("cdrseqgen/datasets/preprocess").resolve()
        for name in self.SCRIPTS:
            with self.subTest(script=name):
                match = pattern.search((here / name).read_text())
                self.assertIsNotNone(match, f"{name} has no recognisable ROOT= line")
                self.assertEqual(match.group(1), "../../..")

    def test_the_resolved_root_is_the_repo(self):
        # Resolve it the way bash does, and check it is the tree that holds train.py.
        root = (Path("cdrseqgen/datasets/preprocess") / ".." / ".." / "..").resolve()
        self.assertTrue((root / "train.py").is_file(), f"{root} is not the repo root")
        self.assertTrue((root / "cdrseqgen").is_dir())

    def test_the_scripts_refuse_a_root_that_is_not_the_repo(self):
        here = Path("cdrseqgen/datasets/preprocess").resolve()
        for name in self.SCRIPTS:
            with self.subTest(script=name):
                text = (here / name).read_text()
                self.assertIn('! -f "$ROOT/train.py"', text)
                self.assertIn("Refusing to create a data tree there.", text)


class MultiDonorCorpusTests(unittest.TestCase):
    """`build_corpus` merges several donors' CSVs into ONE corpus before any split is assigned."""

    def _entry(self, heavy, light, subject, run):
        # Mirror the real shape: build_chain always emits the seven regions beside full_aa, and
        # build_corpus reads the CDRs to report the corpus's longest loop.
        def chain(seq):
            return {"full_aa": seq, "imgt_numbering": {},
                    **{r: seq[:2] for r in ("fwr1", "cdr1", "fwr2", "cdr2",
                                            "fwr3", "cdr3", "fwr4")}}
        return {"heavy": chain(heavy), "light": chain(light),
                "source": {"dataset": "OAS", "run": run, "subject": subject}}

    def _patched(self, files, dedup=True):
        """files: {path: (subject, run, [(heavy, light), ...])}"""
        meta = {path: {"dataset": "OAS", "run": run, "species": "human", "subject": subject}
                for path, (subject, run, _) in files.items()}

        def fake_entries(path, limit=None, source_metadata=None):
            subject, run, pairs = files[path]
            return [self._entry(h, l, subject, run) for h, l in pairs]

        with patch("cdrseqgen.datasets.preprocess.oas_paired.read_oas_metadata",
                   side_effect=lambda path: meta[path]), \
             patch("cdrseqgen.datasets.preprocess.oas_paired.build_entries", side_effect=fake_entries):
            return build_corpus(list(files), dedup=dedup)

    def test_files_become_one_corpus_that_remembers_each_donor(self):
        entries, per_file, stats = self._patched({
            "a.csv": ("Donor-A", 101, [("H1", "L1"), ("H2", "L2")]),
            "b.csv": ("Donor-B", 202, [("H3", "L3")]),
        })
        self.assertEqual(len(entries), 3)
        self.assertEqual([e["source"]["subject"] for e in entries],
                         ["Donor-A", "Donor-A", "Donor-B"])
        self.assertEqual(stats["donors"], ["Donor-A", "Donor-B"])
        self.assertEqual([(m["subject"], m["entries"]) for m in per_file],
                         [("Donor-A", 2), ("Donor-B", 1)])

    def test_a_clone_shared_by_two_donors_is_kept_once(self):
        # A public clone recurs across donors. Kept twice it would inflate the corpus, and under
        # --split random/none the copies could land on both sides of the holdout.
        entries, per_file, stats = self._patched({
            "a.csv": ("Donor-A", 101, [("H1", "L1"), ("H2", "L2")]),
            "b.csv": ("Donor-B", 202, [("H1", "L1"), ("H3", "L3")]),
        })
        self.assertEqual(len(entries), 3)
        self.assertEqual(stats["duplicates_dropped"], 1)
        # The first donor to carry it keeps it, so provenance stays deterministic.
        self.assertEqual(per_file[0]["entries"], 2)
        self.assertEqual(per_file[1]["entries"], 1)

    def test_dedup_can_be_turned_off(self):
        entries, _, stats = self._patched({
            "a.csv": ("Donor-A", 101, [("H1", "L1")]),
            "b.csv": ("Donor-B", 202, [("H1", "L1")]),
        }, dedup=False)
        self.assertEqual(len(entries), 2)
        self.assertEqual(stats["duplicates_dropped"], 0)

    def test_identity_is_the_sequence_pair_not_one_chain(self):
        # Same heavy, different light = different antibody. Keying on the heavy chain alone would
        # silently discard real pairs.
        entries, _, stats = self._patched({
            "a.csv": ("Donor-A", 101, [("H1", "L1")]),
            "b.csv": ("Donor-B", 202, [("H1", "L2")]),
        })
        self.assertEqual(len(entries), 2)
        self.assertEqual(stats["duplicates_dropped"], 0)


if __name__ == "__main__":
    unittest.main()
