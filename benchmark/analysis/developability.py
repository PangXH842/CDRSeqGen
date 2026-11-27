#!/usr/bin/env python3
"""Pure-Python sequence developability & chemical liability profiler for CDR-H3 benchmark outputs.

Computes standard therapeutic antibody CDR-H3 sequence flags and physicochemical properties
directly from finished benchmark `generations/*.json` files and the frozen manifest's native loops
(zero GPU, zero external dependencies):

  * N-linked glycosylation motif: N[^P][ST]
  * Unpaired cysteines: odd count of 'C' in CDR-H3 (aggregation / misfolding risk)
  * Deamidation motifs: NG (high risk), NS / NT (moderate risk)
  * Aspartate isomerization motifs: DG (high risk), DS (moderate risk)
  * Oxidation-prone residues: M, W counts and presence
  * Hydrophobic patch: longest consecutive run in {A, V, I, L, M, F, W, Y} >= 5
  * Net charge at pH 7.4: K + R + 0.1*H - D - E
  * Kyte-Doolittle GRAVY (mean hydropathy) and aromatic fraction (F, W, Y)
  * Major-liability-free rate: no N-glyc, no unpaired Cys, no NG, and no DG

Usage:
    python benchmark/analysis/developability.py \\
        --batch results/benchmark/20261124_181422_091012_oas4_h3_test_final_n1000_full \\
        --batch results/benchmark/20261126_155819_938596_oas4_h3_test_s2s_esm2_n1000_full \\
        --output-json results/benchmark/.../developability_n1000.json \\
        --output-md results/benchmark/.../developability_n1000.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark._common import atomic_write_json
from cdrseqgen.eval.developability import (
    AROMATIC,
    HYDROPHOBIC_RUN_RE,
    KYTE_DOOLITTLE,
    NGLYC_RE,
    SCALAR_METRICS,
    score_sequence,
)

SCHEMA_VERSION = "cdrseqgen.developability.v1"


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average per-framework candidate scores (framework = statistical unit)."""
    per_fw: dict[str, list[float]] = {k: [] for k in SCALAR_METRICS}
    n_frameworks = 0
    n_candidates = 0
    for rec in records:
        samples = [s for s in (rec.get("samples") or []) if s]
        if not samples:
            continue
        n_frameworks += 1
        n_candidates += len(samples)
        scored = [score_sequence(s) for s in samples]
        for key in SCALAR_METRICS:
            per_fw[key].append(statistics.mean(item[key] for item in scored))
    summary = {}
    for key in SCALAR_METRICS:
        vals = per_fw[key]
        summary[key] = {
            "macro_mean": statistics.mean(vals) if vals else None,
            "between_example_sd": statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None),
        }
    return {
        "frameworks": n_frameworks,
        "candidates": n_candidates,
        "metrics": summary,
    }


def summarize_native_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Score the native CDR-H3 sequences in a frozen benchmark manifest as a reference distribution."""
    fake_records = [{"samples": [r["native"]]} for r in manifest.get("records") or []]
    return summarize_records(fake_records)


def family_name(label: str) -> str:
    """Group multi-seed arms (e.g., mlm_s42..s48 -> mlm_scratch) while preserving single arms."""
    m = re.match(r"^(mlm|ar|s2s)_s\d+$", label)
    if m:
        return f"{m.group(1)}_scratch"
    m = re.match(r"^(s2s_esm2|mlm_esm2)_s\d+$", label)
    if m:
        return m.group(1)
    m = re.match(r"^(iglm_s_lr3e-5|iglm_s_random_lr3e-5|iglm_s_random_lr1e-4)_s\d+$", label)
    if m:
        return m.group(1)
    return label


def group_families(arms: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-arm macro means across seeds within each model family."""
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for label, entry in arms.items():
        buckets[family_name(label)].append((label, entry))
    grouped = {}
    for fam, members in buckets.items():
        metrics = {}
        for key in SCALAR_METRICS:
            vals = [m["metrics"][key]["macro_mean"] for _, m in members if m["metrics"][key]["macro_mean"] is not None]
            metrics[key] = {
                "mean": statistics.mean(vals) if vals else None,
                "seed_sd": statistics.stdev(vals) if len(vals) > 1 else None,
            }
        grouped[fam] = {
            "seeds": len(members),
            "arms": [lbl for lbl, _ in members],
            "frameworks": members[0][1]["frameworks"],
            "candidates": sum(m["candidates"] for _, m in members),
            "metrics": metrics,
        }
    return grouped


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CDR-H3 Sequence Developability & Liability Profile",
        "",
        f"- **Manifest:** `{result['manifest']['path']}` (`n = {result['manifest']['frameworks']}` frameworks)",
        f"- **Major-liability-free definition:** No N-glycosylation (`N[^P][ST]`), no unpaired Cys (odd `C`), no `NG` deamidation, no `DG` isomerization.",
        "",
        "| Family / Reference | Seeds | Major-free (%) | N-glyc (%) | Unpaired C (%) | NG (%) | DG (%) | Hydro5 (%) | Charge | GRAVY | Arom (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows: list[tuple[str, str, dict[str, float | None]]] = []
    nat = result["native_reference"]["metrics"]
    rows.append(("**Native OAS4 test H3**", "ref", {k: nat[k]["macro_mean"] for k in SCALAR_METRICS}))
    for fam, entry in result["families"].items():
        rows.append((fam, str(entry["seeds"]), {k: entry["metrics"][k]["mean"] for k in SCALAR_METRICS}))

    for name, seeds, m in rows:
        pct = lambda k: f"{100.0 * m[k]:.2f}" if m[k] is not None else "—"
        num = lambda k: f"{m[k]:+.3f}" if m[k] is not None else "—"
        lines.append(
            f"| {name} | {seeds} | {pct('major_liability_free_rate')} | {pct('nglyc_rate')} | "
            f"{pct('unpaired_cys_rate')} | {pct('deamidation_ng_rate')} | {pct('isomerization_dg_rate')} | "
            f"{pct('hydrophobic_patch5_rate')} | {num('net_charge_ph74_mean')} | {num('gravy_mean')} | "
            f"{pct('aromatic_fraction_mean')} |"
        )
    lines.append("")
    return "\n".join(lines)


def collect_from_batches(batch_dirs: list[Path]) -> tuple[Path, dict[str, Path]]:
    manifest_path: Path | None = None
    arms: dict[str, Path] = {}
    for bdir in batch_dirs:
        plan = json.loads((bdir / "plan.json").read_text())
        mpath = (ROOT / plan["manifest"]).resolve()
        if manifest_path is None:
            manifest_path = mpath
        elif manifest_path != mpath:
            raise ValueError(f"batch {bdir} uses manifest {mpath}, expected {manifest_path}")
        gen_dir = bdir / "generations"
        for gpath in sorted(gen_dir.glob("*.json")):
            label = gpath.stem
            if label in arms:
                raise ValueError(f"duplicate arm label {label!r} across batches")
            arms[label] = gpath
    if manifest_path is None:
        raise ValueError("at least one --batch or --manifest must be supplied")
    return manifest_path, arms


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, action="append", default=[],
                   help="Completed benchmark batch directory (may be passed multiple times).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Explicit frozen manifest path (inferred from --batch when omitted).")
    p.add_argument("--input", action="append", default=[],
                   help="Optional extra LABEL=PATH generation JSON(s).")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-md", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    arms_paths: dict[str, Path] = {}
    manifest_path = args.manifest
    if args.batch:
        inferred_manifest, batch_arms = collect_from_batches(args.batch)
        manifest_path = manifest_path or inferred_manifest
        arms_paths.update(batch_arms)
    for spec in args.input:
        label, sep, path_str = spec.partition("=")
        if not (label and sep and path_str):
            raise ValueError(f"expected LABEL=PATH for --input, got {spec!r}")
        arms_paths[label] = Path(path_str)
    if manifest_path is None:
        raise ValueError("provide --batch or --manifest")

    manifest = json.loads(manifest_path.read_text())
    native_summary = summarize_native_manifest(manifest)

    arm_summaries = {}
    for label, gpath in arms_paths.items():
        payload = json.loads(gpath.read_text())
        arm_summaries[label] = summarize_records(payload.get("records") or [])

    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest": {
            "path": str(manifest_path),
            "frameworks": len(manifest.get("records") or []),
        },
        "native_reference": native_summary,
        "arms": arm_summaries,
        "families": group_families(arm_summaries),
    }

    md = format_markdown(result)
    if args.output_json:
        atomic_write_json(args.output_json, result, allow_nan=False)
        print(f"Wrote JSON: {args.output_json}")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md)
        print(f"Wrote Markdown: {args.output_md}")
    if not args.output_md:
        print(md)


if __name__ == "__main__":
    main()
