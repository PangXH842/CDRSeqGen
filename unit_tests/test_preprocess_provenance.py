import csv
import io
from contextlib import redirect_stderr, redirect_stdout
import re
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from cdrseqgen.datasets.preprocess.common import (RejectedEntry, optional_text,
                                                 write_processed, write_processed_metadata)
from cdrseqgen.datasets.preprocess import oas_paired, sabdab2
from cdrseqgen.datasets.preprocess import __main__ as pipeline
from cdrseqgen.datasets.preprocess.__main__ import build_corpus, build_entries, build_entry
from cdrseqgen.datasets.preprocess.oas_paired import read_oas_metadata


def options(source, *flags):
    """The parsed command line build_corpus takes, so the tests drive the real CLI."""
    return pipeline.args(list(flags), source_name=source)


class PreprocessProvenanceTests(unittest.TestCase):
    def test_reads_oas_source_metadata(self):
        metadata = {
            "Run": 1287169,
            "Species": "Human",
            "Link": "https://doi.org/10.1038/s41586-022-05371-z",
            "Author": "Jaffe et al., 2022",
            "Subject": "Donor-4",
            "BType": "Naive-B-Cells",
        }
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv") as handle:
            csv.writer(handle).writerow([json.dumps(metadata)])
            handle.flush()
            parsed = read_oas_metadata(handle.name)

        self.assertEqual(parsed["run"], 1287169)
        self.assertEqual(parsed["species"], "human")
        self.assertEqual(parsed["subject"], "Donor-4")
        self.assertEqual(parsed["b_type"], "Naive-B-Cells")

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

        source = {"dataset": "OAS", "run": 7, "species": "human", "subject": "Donor-4",
                  "b_type": "Naive-B-Cells"}
        with patch.object(pipeline.pd, "read_csv", return_value=pd.DataFrame([row])):
            entries, _ = build_entries("unused.csv", "oas_paired", source_metadata=source)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["species"], "human")
        # The donor rides on the entry, not just on the sidecar: a corpus merged from several
        # donors' files can only be split donor-wise if every entry knows whose it is.
        # B-cell type rides along for the same reason: a corpus merged past these four
        # naive donors would otherwise hide that it mixes naive with memory/plasmablast.
        self.assertEqual(entries[0]["source"],
                         {"dataset": "OAS", "run": 7, "subject": "Donor-4",
                          "b_type": "Naive-B-Cells"})
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

    def _patched(self, files, dedup=True, **split_options):
        """files: {path: (subject, run, [(heavy, light), ...])}"""
        meta = {path: {"dataset": "OAS", "run": run, "species": "human", "subject": subject}
                for path, (subject, run, _) in files.items()}

        def fake_entries(path, source, limit=None, source_metadata=None, require_antigen=True):
            subject, run, pairs = files[path]
            rows = [self._entry(h, l, subject, run) for h, l in pairs]
            return rows, {"stages": [], "rejected_rows": {}, "kept": len(rows)}

        flags = ["--input_file", *files, "--dedup", "exact" if dedup else "none"]
        for name, value in split_options.items():
            flags += [f"--{name}", str(value)]
        with patch("cdrseqgen.datasets.preprocess.oas_paired.read_oas_metadata",
                   side_effect=lambda path: meta[path]), \
             patch.object(pipeline, "build_entries", side_effect=fake_entries):
            return build_corpus(options("oas_paired", *flags))

    def test_files_become_one_corpus_that_remembers_each_donor(self):
        entries, metadata = self._patched({
            "a.csv": ("Donor-A", 101, [("H1", "L1"), ("H2", "L2")]),
            "b.csv": ("Donor-B", 202, [("H3", "L3")]),
        })
        self.assertEqual(len(entries), 3)
        self.assertEqual([e["source"]["subject"] for e in entries],
                         ["Donor-A", "Donor-A", "Donor-B"])
        self.assertEqual(metadata["donors"], ["Donor-A", "Donor-B"])
        self.assertEqual([(m["subject"], m["entries"]) for m in metadata["sources"]],
                         [("Donor-A", 2), ("Donor-B", 1)])

    def test_a_clone_shared_by_two_donors_is_kept_once(self):
        # A public clone recurs across donors. Kept twice it would inflate the corpus, and under
        # --split random/none the copies could land on both sides of the holdout.
        entries, metadata = self._patched({
            "a.csv": ("Donor-A", 101, [("H1", "L1"), ("H2", "L2")]),
            "b.csv": ("Donor-B", 202, [("H1", "L1"), ("H3", "L3")]),
        })
        self.assertEqual(len(entries), 3)
        self.assertEqual(metadata["preprocessing"]["duplicates_dropped"], 1)
        # The first donor to carry it keeps it, so provenance stays deterministic.
        self.assertEqual(metadata["sources"][0]["entries"], 2)
        self.assertEqual(metadata["sources"][1]["entries"], 1)

    def test_dedup_can_be_turned_off(self):
        entries, metadata = self._patched({
            "a.csv": ("Donor-A", 101, [("H1", "L1")]),
            "b.csv": ("Donor-B", 202, [("H1", "L1")]),
        }, dedup=False)
        self.assertEqual(len(entries), 2)
        self.assertEqual(metadata["preprocessing"]["duplicates_dropped"], 0)

    def test_identity_is_the_sequence_pair_not_one_chain(self):
        # Same heavy, different light = different antibody. Keying on the heavy chain alone would
        # silently discard real pairs.
        entries, metadata = self._patched({
            "a.csv": ("Donor-A", 101, [("H1", "L1")]),
            "b.csv": ("Donor-B", 202, [("H1", "L2")]),
        })
        self.assertEqual(len(entries), 2)
        self.assertEqual(metadata["preprocessing"]["duplicates_dropped"], 0)

    def test_split_is_assigned_across_sources_and_preserves_test_membership(self):
        files = {
            "a.csv": ("Donor-A", 101, [("AA", "L1"), ("CC", "L2"), ("DD", "L3")]),
            "b.csv": ("Donor-B", 202, [("AA", "L4"), ("EE", "L5"), ("FF", "L6")]),
        }
        two_way, _ = self._patched(files, test_size=0.3, validation_size=0)
        three_way, metadata = self._patched(files, test_size=0.3, validation_size=0.3)
        self.assertEqual([e["split"] == "test" for e in two_way],
                         [e["split"] == "test" for e in three_way])
        self.assertEqual(three_way[0]["split"], three_way[3]["split"])
        self.assertEqual({e["split"] for e in three_way}, {"train", "validation", "test"})
        self.assertEqual(metadata["preprocessing"]["validation_fraction"], 0.3)

    def test_empty_corpus_and_no_split(self):
        entries, metadata = self._patched({"a.csv": ("Donor-A", 101, [])})
        self.assertEqual(entries, [])
        self.assertEqual(metadata["records"], 0)
        entries, metadata = self._patched(
            {"a.csv": ("Donor-A", 101, [("AA", "LL")])}, split="none")
        self.assertNotIn("split", entries[0])
        self.assertEqual(metadata["preprocessing"]["split"], "none")


class EntryBuilderTests(unittest.TestCase):
    SOURCE = {"dataset": "OAS", "species": "human", "run": 7, "subject": "Donor-A"}

    def oas_row(self, **changes):
        row = {
            "productive_heavy": "T", "productive_light": "T",
            "ANARCI_numbering_heavy": "{}", "ANARCI_numbering_light": "{}",
            "sequence_id_heavy": "h", "sequence_id_light": "l", "Isotype_heavy": "IGHM",
            "sequence_alignment_aa_heavy": "AAAAAAA", "sequence_alignment_aa_light": "AAAAAAA",
            **{f"{r}_aa_{c}": "A" for c in ("heavy", "light") for r in oas_paired.REGIONS},
        }
        return {**row, **changes}

    def sabdab_row(self, **changes):
        numbering = json.dumps([["A", [lo, ""]] for lo, _ in sabdab2.IMGT_BOUNDARIES.values()])
        row = {
            "type": "FAB", "agtypes": "PROTEIN/PEPTIDE", "agresolvedseqs": "AA/CC",
            "VH_numbering_list": numbering, "VL_numbering_list": numbering,
            "INSTANCE": "complex-1", "Hchain": "H", "Lchain": "L", "species": float("nan"),
            "PDB_ID": "pdb-1", "SABDAB_ID": "ab-1", "VH_numerable_seq": "AAAAAAA",
            "VL_numerable_seq": "AAAAAAA", "agchains": "A/B", "ab_ag_split": "test",
        }
        return {**row, **changes}

    def test_a_real_X_residue_is_refused_rather_than_read_as_a_chain_break(self):
        """'X' joins antigen chains, so an X INSIDE a chain would silently move a boundary."""
        with self.assertRaisesRegex(RejectedEntry, "ambiguous-chain-break"):
            build_entry(self.sabdab_row(agresolvedseqs="AXA/CC"), "sabdab2")
        # The unambiguous case still builds, and one X per break survives.
        entry = build_entry(self.sabdab_row(agresolvedseqs="AA/CC/DD"), "sabdab2")
        self.assertEqual(entry["antigen"]["full_aa"], "AAXCCXDD")
        self.assertEqual(entry["antigen"]["full_aa"].count("X"), 2)

    def test_the_antigen_requirement_can_be_lifted_and_is_recorded(self):
        """The *_ag_lconly configs ignore the antigen, so they should not pay its filter."""
        rows = [self.sabdab_row(),
                self.sabdab_row(INSTANCE="no-antigen", agtypes="HAPTEN",
                                agresolvedseqs=float("nan"))]
        with patch.object(pipeline.pd, "read_csv", return_value=pd.DataFrame(rows)), \
             redirect_stdout(io.StringIO()):
            strict, _ = build_corpus(options("sabdab2", "--input_file", "s.csv"))
            loose, metadata = build_corpus(options("sabdab2", "--input_file", "s.csv",
                                                   "--allow_missing_antigen"))
        self.assertEqual(len(strict), 1)
        self.assertEqual(len(loose), 2)
        self.assertIsNone(loose[1]["antigen"])
        self.assertFalse(metadata["preprocessing"]["protein_or_peptide_antigen_only"])

    def test_the_filtering_funnel_reaches_the_metadata_not_just_stdout(self):
        """A sidecar that records WHICH filters ran but not what they dropped cannot be audited."""
        rows = [self.sabdab_row(), self.sabdab_row(INSTANCE="bad", VH_numbering_list="nope")]
        with patch.object(pipeline.pd, "read_csv", return_value=pd.DataFrame(rows)), \
             redirect_stdout(io.StringIO()):
            _, metadata = build_corpus(options("sabdab2", "--input_file", "s.csv"))
        funnel = metadata["filtering"]
        self.assertEqual(funnel["kept"], 1)
        self.assertEqual(funnel["rejected_rows"]["malformed-numbering"], 1)
        self.assertEqual([stage["label"] for stage in funnel["stages"]][0], "raw")

    def test_validation_is_carved_by_group_so_the_split_stays_disjoint(self):
        """A row-wise carve would split one antigen across train and validation."""
        entries = []
        for index in range(20):
            entry = {"split": "train", "heavy": {"cdr3": f"H{index}"},
                     "antigen": {"full_aa": f"AG{index // 2}"}}   # each antigen spans two rows
            entries.append(entry)
        entries.append({"split": "test", "heavy": {"cdr3": "HELD"},
                        "antigen": {"full_aa": "AGHELD"}})

        report = sabdab2.carve_validation_split(entries, validation_size=0.3, seed=42)
        by_split = {}
        for entry in entries:
            by_split.setdefault(entry["split"], []).append(entry)
        self.assertEqual(len(by_split["test"]), 1, "the held-out partition must never be touched")
        self.assertGreaterEqual(report["moved_records"], report["target_records"])
        # Every antigen group moved whole: no antigen appears on both sides.
        def antigens(split):
            return {e["antigen"]["full_aa"] for e in by_split.get(split, [])}
        self.assertFalse(antigens("train") & antigens("validation"))
        # Deterministic under a seed, and off by default.
        self.assertEqual(sabdab2.carve_validation_split([], validation_size=0.3)["moved_records"], 0)

    def test_expected_rejections_are_distinct_from_schema_errors(self):
        cases = (
            ("oas_paired", self.oas_row(), "ANARCI_numbering_heavy", "cdr1_aa_heavy"),
            ("sabdab2", self.sabdab_row(), "VH_numbering_list", None),
        )
        for source, row, numbering_key, region_key in cases:
            metadata = self.SOURCE if source == "oas_paired" else None
            with self.subTest(source=source):
                with self.assertRaisesRegex(RejectedEntry, "malformed-numbering"):
                    build_entry({**row, numbering_key: "bad-numbering"}, source, metadata)
                invalid = {**row, region_key: ""} if region_key else {**row, numbering_key: "[]"}
                with self.assertRaisesRegex(RejectedEntry, "missing/empty regions"):
                    build_entry(invalid, source, metadata)
                missing = dict(row)
                del missing[numbering_key]
                with self.assertRaises(KeyError):
                    build_entry(missing, source, metadata)
        with self.assertRaises(ValueError) as raised:
            build_entry(self.sabdab_row(VH_numbering_list='[["A"]]'), "sabdab2")
        self.assertNotIsInstance(raised.exception, RejectedEntry)

    def test_filters_precede_limit_and_rejected_rows_count_towards_limit(self):
        cases = (
            ("oas_paired", [self.oas_row(productive_heavy="F"),
                            self.oas_row(ANARCI_numbering_heavy="bad"), self.oas_row(),
                            self.oas_row(sequence_id_heavy="later")]),
            ("sabdab2", [self.sabdab_row(type="SD-H"),
                         self.sabdab_row(VH_numbering_list="bad"), self.sabdab_row(),
                         self.sabdab_row(INSTANCE="later")]),
        )
        for source, rows in cases:
            metadata = self.SOURCE if source == "oas_paired" else None
            with self.subTest(source=source), \
                 patch.object(pipeline.pd, "read_csv", return_value=pd.DataFrame(rows)), \
                 redirect_stdout(io.StringIO()) as output:
                entries, _ = build_entries("unused.csv", source, limit=2,
                                           source_metadata=metadata)
                self.assertEqual(len(entries), 1)
                self.assertNotEqual(entries[0]["sequence_id"], "later")
                self.assertIn("1 malformed-numbering", output.getvalue())

    def test_sabdab_corpus_preserves_complexes_and_source_splits(self):
        rows = [self.sabdab_row(), self.sabdab_row(INSTANCE="complex-2", ab_ag_split="train",
                                                 agresolvedseqs="DD", species=" human ")]
        with patch.object(pipeline.pd, "read_csv", return_value=pd.DataFrame(rows)), \
             redirect_stdout(io.StringIO()):
            entries, metadata = build_corpus(options("sabdab2", "--input_file", "source.csv"))
        self.assertEqual([e["split"] for e in entries], ["test", "train"])
        self.assertEqual(entries[0]["antigen"]["full_aa"], "AAXCC")
        self.assertEqual(entries[0]["antigen"]["chains"], ["A", "B"])
        self.assertEqual(entries[0]["heavy"], entries[1]["heavy"])
        self.assertEqual(metadata["records"], 2)
        self.assertEqual(metadata["species"], {"known_records": 1, "unknown_records": 1})
        self.assertEqual(metadata["source"]["file"], "source.csv")
        self.assertEqual(metadata["preprocessing"]["split"], "source ab_ag_split")


class CorpusSourceTests(unittest.TestCase):
    """Adding a source must be one SOURCES entry plus its branches -- nothing else to remember."""

    def test_every_source_has_defaults_and_flags_that_parse(self):
        self.assertTrue(pipeline.SOURCES)
        for name, source in pipeline.SOURCES.items():
            with self.subTest(source=name):
                self.assertTrue(source.input_default and source.output_default)
                parsed = pipeline.args([], source_name=name)
                # Its own flags reach the same namespace build_corpus reads, so a flag a source
                # registers but never consumes shows up here rather than at the end of a build.
                self.assertEqual(parsed.source, name)
                self.assertEqual(parsed.output_file, source.output_default)
                self.assertEqual(parsed.input_file, [source.input_default] if source.multiple
                                 else source.input_default)

    def test_no_source_prints_the_registered_ones_instead_of_building(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(pipeline.args([]))
        for name in pipeline.SOURCES:
            self.assertIn(name, output.getvalue())

    def test_an_unknown_source_is_refused(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.args(["--source", "oas_unpaired"])


if __name__ == "__main__":
    unittest.main()
