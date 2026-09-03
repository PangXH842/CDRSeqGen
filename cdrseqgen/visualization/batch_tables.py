"""Generate validated Markdown comparison tables from a finished training batch.

This is the tabular counterpart to ``linegraph_plotter``: callers select an analysis, while one
renderer owns Markdown layout and row-width validation. The standard tables read existing run and
dataset artifacts only; they do not rerun inference or require a GPU.

Usage:
    python -m cdrseqgen.visualization.batch_tables                          # results/runs/latest
    python -m cdrseqgen.visualization.batch_tables --batch results/runs/20260729_202635
    python -m cdrseqgen.visualization.batch_tables --only regimes,loops

Metrics:
  * AAR / Amino acid recovery (whole, core)
  * Diversity (pairwise Hamming distance)
  * Perplexity (self, judge)
  * Novelty (uniqueness, relative to train set)
  * Entropy (per-position, loop length)

Named tables:
  * regimes   -- AAR by regime (dataset; loop), across models (eg. scratch & ESM-2)
  * loops     -- AAR by CDR loop (OAS only), w/ or w/o partner-chain context
  * antigen   -- AAR by H CDR loop (SAbDab2 only), w/ or w/o antigen context
  * retrieval -- train/test CDR overlap for OAS and SAbDab2
  * design    -- AAR, diversity, entropy, judge-ppl by loop and model
  * length    -- loop-length distributions and entropy supplied by fixing the target length

The optional decoding table compares two ``generation_eval.json`` files passed with
``--decoding-json``; it is skipped when the pair is absent or unreadable.

Main functions:
  * discover_runs    -- parse a batch into {(loop, dataset, context, model): run_dir}.
  * recovery_summary -- choose a coverage-valid recovery metric and its best epoch for one run.
  * markdown_table   -- validate and render headers and rows through one Markdown path.
  * build_table      -- generate any named table from definitions and row providers.
  * main             -- CLI: discover runs and print the selected tables.

Notes:
  * Run dirs are parsed, not hardcoded, so a different ESM-2 variant or an added context flag is
    picked up automatically. Naming comes from train.py: <loop>_<DS>[_LC|_HC][_AG]_<model>.
  * Recovery reporting is driven by each run's recorded eval_core_coverage, not by a hard-coded
    loop name. Core is used only when coverage reaches MIN_REPORT_CORE_COVERAGE; every table labels
    the selected metric and coverage. Old runs without coverage fall back visibly to whole-span.
  * Standard tables read ``epoch_metrics.csv``, ``generation_eval.json``, the shared run-config
    filename aliases, and processed dataset JSONs.
  * A batch mixing training objectives is flagged: a core-weighted run optimises the number these
    tables report, so its column is not like-for-like with a cross-entropy run.
"""
import os
import re
import json
import math
import glob
import argparse
import statistics
from collections import Counter, defaultdict

import pandas as pd

from cdrseqgen.artifacts import find_run_config
from cdrseqgen.eval import MIN_CORE_COVERAGE
from cdrseqgen.train.logging import DATASET_TAGS, UNKNOWN_DATASET_TAG
from cdrseqgen import artifacts

# <loop>_<DS>[_LC|_HC][_AG]_<model>, e.g. h3_OAS_LC_esm2_t12_35M_UR50D, h1_SD2_LC_AG_scratch
#
# The dataset alternation is built from DATASET_TAGS rather than written out, because it was
# written out: adding the OAS4 tag for the four-donor corpus left this regex matching only
# OAS|SD2|DS, so every OAS4 run failed to parse, discover_runs returned nothing, and a finished
# batch rendered empty tables and dashboards without an error anywhere. Longest first so OAS4
# cannot be shadowed by OAS.
_DS_ALTERNATION = "|".join(sorted(
    {tag for _, tag in DATASET_TAGS} | {UNKNOWN_DATASET_TAG}, key=lambda t: (-len(t), t)))
RUN_RE = re.compile(
    rf"^(?P<loop>[hl][123])_(?P<ds>{_DS_ALTERNATION})(?P<ctx>(?:_LC|_HC|_AG)*)_(?P<model>.+)$")
LOOPS = ["h1", "h2", "h3", "l1", "l2", "l3"]
DEFAULT_DATA = {"OAS": "dataset/data/processed/processed_oas_data.json",
                "OAS4": "dataset/data/processed/processed_oas_multi.json",
                "SD2": "dataset/data/processed/processed_sabdab2_data.json"}


def dataset_alias(runs):
    """Map the tables' canonical "OAS" onto whichever OAS-family tag this batch actually used.

    A corpus from a different donor set takes its own tag (OAS -> OAS4) so two test splits can
    never share a run name. Every table below is written against the canonical "OAS", so without
    this an OAS4 batch parses correctly and then renders every OAS row blank. A batch uses one
    corpus per round, so resolving to the single OAS-family tag present is unambiguous.
    """
    present = {key[1] for key in runs}
    if "OAS" in present:
        return {}
    family = sorted(tag for tag in present if tag.startswith("OAS"))
    return {"OAS": family[0]} if len(family) == 1 else {}
MIN_REPORT_CORE_COVERAGE = MIN_CORE_COVERAGE
# Written into the batch directory itself, beside the runs the tables describe.
BATCH_TABLES_FILENAME = "batch_tables.md"


def _epoch_metrics(run_dir):
    """Read one run's epoch metrics, returning None for missing or empty historical artifacts"""
    csv = os.path.join(run_dir, artifacts.EPOCH_METRICS_FILENAME)
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    return None if df.empty else df


def model_family(model):
    """Collapse a model_type/run suffix to the family used in the tables"""
    m = model.lower()
    return "scratch" if m == "scratch" else ("esm2" if "esm2" in m else m)


def discover_runs(batch):
    """{(loop, ds, ctx, family): run_dir} for every parseable run in a batch directory"""
    runs = {}
    for path in sorted(glob.glob(os.path.join(batch, "*"))):
        if not os.path.isdir(path):
            continue
        m = RUN_RE.match(os.path.basename(path))
        if not m:
            continue
        runs[(m["loop"], m["ds"], m["ctx"], model_family(m["model"]))] = path
    return runs


def recovery_summary(run_dir, min_core_coverage=MIN_REPORT_CORE_COVERAGE):
    """Return coverage-aware recovery at one consistently selected epoch.

    Core recovery is eligible only when every recorded evaluation epoch has at least the declared
    coverage. Otherwise whole-span recovery selects the epoch and core is reported as unavailable.
    This makes the rule portable across loops, datasets, numbering schemes, and anchor widths.
    """
    if not 0 <= min_core_coverage <= 1:
        raise ValueError("min_core_coverage must be in [0, 1]")
    df = _epoch_metrics(run_dir)
    if df is None or "eval_accuracy" not in df.columns:
        return None

    coverage = None
    if "eval_core_coverage" in df.columns:
        coverage_values = pd.to_numeric(df["eval_core_coverage"], errors="coerce").dropna()
        if not coverage_values.empty:
            if ((coverage_values < 0) | (coverage_values > 1)).any():
                raise ValueError(f"invalid eval_core_coverage in {run_dir}")
            # Coverage should be invariant across epochs. Taking the minimum is conservative if a
            # malformed/historical artifact varies, and never upgrades a partly covered metric.
            coverage = float(coverage_values.min())

    core_values = (
        pd.to_numeric(df["eval_core_accuracy"], errors="coerce")
        if "eval_core_accuracy" in df.columns else None
    )
    core_usable = (
        coverage is not None
        and coverage >= min_core_coverage
        and core_values is not None
        and core_values.notna().any()
    )
    selection_metric = "eval_core_accuracy" if core_usable else "eval_accuracy"
    selection_values = pd.to_numeric(df[selection_metric], errors="coerce")
    if not selection_values.notna().any():
        return None
    i = selection_values.idxmax()
    row = df.loc[i]

    if core_usable:
        reason = "core coverage is adequate"
    elif coverage is None:
        reason = "core coverage unavailable"
    else:
        reason = f"core coverage below {min_core_coverage:.0%} threshold"
    return {
        "primary_metric": "core" if core_usable else "whole",
        "primary_value": float(row[selection_metric]),
        "selection_metric": selection_metric,
        "epoch": int(row["epoch"]),
        "whole_accuracy": float(row["eval_accuracy"]),
        "core_accuracy": float(row["eval_core_accuracy"]) if core_usable else None,
        "core_coverage": coverage,
        "reason": reason,
    }


def _common_recovery_policy(summaries):
    """(metric label, minimum coverage) shared by summaries, or 'mixed' when incomparable."""
    present = [summary for summary in summaries if summary is not None]
    if not present:
        return "—", None
    metrics = {summary["primary_metric"] for summary in present}
    label = metrics.pop() if len(metrics) == 1 else "mixed"
    coverages = [summary["core_coverage"] for summary in present]
    coverage = min(coverages) if all(value is not None for value in coverages) else None
    return label, coverage


def _coverage_text(coverage):
    return f"{coverage:.1%}" if coverage is not None else "unknown"


def _primary_text(summary, *, label=False):
    if summary is None:
        return "—"
    suffix = f" {summary['primary_metric']}" if label else ""
    return f"{summary['primary_value']:.4f}{suffix}"


def _load(path):
    with open(path) as f:
        return json.load(f)


def data_path_for(runs, ds):
    """Prefer the data_path a run recorded in run_config.json; fall back to the standard location"""
    for (_, d, _, _), run_dir in runs.items():
        if d != ds:
            continue
        # run_config.json was run_meta.json before 2026-07-29; older batches still use the old name.
        meta = find_run_config(run_dir)
        if meta is not None:
            p = _load(str(meta)).get("data_path")
            if p and os.path.exists(p):
                return p
    return DEFAULT_DATA.get(ds)


# --- table definitions and rendering -----------------------------------------------------------

ALIGNMENT_RULES = {"left": "---", "right": "---:", "center": ":---:"}


def markdown_table(title, headers, rows, alignments=None):
    """Render validated rows as one Markdown table."""
    alignments = alignments or ("left",) * len(headers)
    if len(alignments) != len(headers):
        raise ValueError("one alignment is required per table header")

    rendered_rows = []
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(
                f"table row has {len(row)} cells but {len(headers)} headers: {row!r}"
            )
        rendered_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")

    rules = [ALIGNMENT_RULES[alignment] for alignment in alignments]
    return "\n".join([
        f"#### {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(rules) + " |",
        *rendered_rows,
    ])


def unseen_fraction(entries, cdr, chain="heavy"):
    """(n_unique, n_total, n_test, n_test_with_unseen_cdr) for one CDR under the entries' split."""
    vals = [entry[chain][cdr] for entry in entries]
    train = {entry[chain][cdr] for entry in entries if entry.get("split") == "train"}
    test = [entry for entry in entries if entry.get("split") not in (None, "train")]
    unseen = sum(1 for entry in test if entry[chain][cdr] not in train)
    return len(set(vals)), len(vals), len(test), unseen


# Recovery tables differ only in cases, run columns, and optional annotations/deltas. A context
# selector is either the case's base context, the partner-chain context for that loop, or a literal
# run-name suffix such as "_LC_AG".
BASE_CONTEXT = "base"
PARTNER_CONTEXT = "partner"

RECOVERY_TABLES = {
    "regimes": {
        "title": "Findings §1 — pretraining pays by regime",
        "label_header": "case",
        "annotation_header": "unseen in train",
        "cases": [
            ("OAS H1", "h1", "OAS", ""),
            ("SD2 H1", "h1", "SD2", "_LC"),
            ("SD2 H2", "h2", "SD2", "_LC"),
            ("OAS H3", "h3", "OAS", ""),
        ],
        "columns": [
            ("scratch", "scratch", (BASE_CONTEXT,)),
            ("ESM-2", "esm2", (BASE_CONTEXT,)),
        ],
        "delta": True,
    },
    "loops": {
        "title": "Findings §2 — best-epoch recovery, OAS (coverage-aware)",
        "label_header": "CDR",
        "cases": [(loop.upper(), loop, "OAS", "") for loop in LOOPS],
        "columns": [
            ("scratch", "scratch", (BASE_CONTEXT,)),
            ("+ctx", "scratch", (PARTNER_CONTEXT,)),
            ("ESM-2", "esm2", (BASE_CONTEXT,)),
            ("+ctx", "esm2", (PARTNER_CONTEXT,)),
        ],
    },
    "antigen": {
        "title": "Findings §4 — antigen ablation, SAbDab2 (coverage-aware)",
        "label_header": "CDR",
        "cases": [(loop.upper(), loop, "SD2", "_LC") for loop in ("h1", "h2", "h3")],
        "columns": [
            ("scratch lc-only → +ag", "scratch", ("_LC", "_LC_AG")),
            ("ESM-2 lc-only → +ag", "esm2", ("_LC", "_LC_AG")),
        ],
    },
    "methods": {
        "title": "Causal integration — H3 teacher-forced recovery (length policies differ)",
        "label_header": "case",
        "cases": [
            ("OAS H3", "h3", "OAS", ""),
            ("OAS H3 + partner", "h3", "OAS", "_LC"),
        ],
        "columns": [
            ("scratch (masked)", "scratch", (BASE_CONTEXT,)),
            ("scratch (causal)", "ar_scratch", (BASE_CONTEXT,)),
        ],
        "delta": True,
    },
}


def _resolved_context(case, selector):
    _, loop, _, base_context = case
    if selector == BASE_CONTEXT:
        return base_context
    if selector == PARTNER_CONTEXT:
        return "_HC" if loop.startswith("l") else "_LC"
    return selector


def _unseen_text(case, runs, data_cache):
    _, loop, dataset, _ = case
    path = data_path_for(runs, dataset)
    if not path or not os.path.exists(path):
        return "—"
    if dataset not in data_cache:
        data_cache[dataset] = _load(path)
    _, _, n_test, unseen = unseen_fraction(data_cache[dataset], "cdr" + loop[1])
    return f"{unseen / n_test:.1%}" if n_test else "—"


def _recovery_table(name, runs):
    """Build any recovery comparison from its declarative case/column specification."""
    spec = RECOVERY_TABLES[name]
    headers = [spec["label_header"]]
    if spec.get("annotation_header"):
        headers.append(spec["annotation_header"])
    headers.extend(("metric", "min core coverage"))
    headers.extend(column[0] for column in spec["columns"])
    if spec.get("delta"):
        headers.append("Δ")

    rows = []
    data_cache = {}
    alias = dataset_alias(runs)
    for case in spec["cases"]:
        label, loop, dataset, _ = case
        dataset = alias.get(dataset, dataset)
        column_summaries = []
        for _, family, context_selectors in spec["columns"]:
            summaries = []
            for selector in context_selectors:
                key = (loop, dataset, _resolved_context(case, selector), family)
                run_dir = runs.get(key)
                summaries.append(recovery_summary(run_dir) if run_dir else None)
            column_summaries.append(summaries)

        all_summaries = [summary for column in column_summaries for summary in column]
        metric, coverage = _common_recovery_policy(all_summaries)
        label_values = metric == "mixed"

        row = [label]
        if spec.get("annotation_header"):
            row.append(_unseen_text(case, runs, data_cache))
        row.extend((metric, _coverage_text(coverage)))
        for summaries in column_summaries:
            if any(summary is None for summary in summaries):
                row.append("—")
            else:
                row.append(" → ".join(
                    _primary_text(summary, label=label_values) for summary in summaries
                ))

        if spec.get("delta"):
            scratch = column_summaries[0][0]
            pretrained = column_summaries[1][0]
            comparable = scratch is not None and pretrained is not None and metric != "mixed"
            delta = (
                f"{pretrained['primary_value'] - scratch['primary_value']:+.3f}"
                if comparable else "—"
            )
            row.append(f"**{delta}**")
        rows.append(row)

    alignments = ["left"] * len(headers)
    alignments[headers.index("min core coverage")] = "right"
    for index in range(headers.index("min core coverage") + 1, len(headers)):
        alignments[index] = "right"
    return markdown_table(spec["title"], headers, rows, alignments)


# Model families a batch can hold, in display order. `ar_scratch` is how a causal run's name
# parses (h3_OAS_ar_scratch -> model "ar_scratch"); `batch_plots` imports this so a family added
# here shows up in both the tables and the dashboards.
FAMILY_LABELS = {
    "scratch": "scratch (masked)",
    "esm2": "ESM-2 (masked)",
    "ar_scratch": "scratch (causal)",
}

GEN_KEYS = [
    ("aar_mean", "recovery"),
    ("diversity_hamming", "diversity"),
    ("per_position_entropy_bits", "entropy(b)"),
    ("unique_fraction", "unique"),
    ("self_perplexity_mean", "self-ppl"),
    ("judge_perplexity_mean", "judge-ppl"),
]


def _gen_means(path):
    """Mean of each generative metric over the records of one generation_eval.json."""
    if not path or not os.path.exists(path):
        return None
    records = _load(path).get("records", [])
    means = {}
    for key, _ in GEN_KEYS:
        values = [
            record["metrics"][key]
            for record in records
            if record["metrics"].get(key) is not None
        ]
        means[key] = statistics.mean(values) if values else float("nan")
    means["_n"] = len(records)
    return means


def _retrieval_rows(runs):
    rows = []
    alias = dataset_alias(runs)
    for dataset in (alias.get("OAS", "OAS"), "SD2"):
        path = data_path_for(runs, dataset)
        if not path or not os.path.exists(path):
            rows.append((dataset, "(data not found)", "", "", ""))
            continue
        entries = _load(path)
        unique_h1, total, n_test, unseen_h1 = unseen_fraction(entries, "cdr1")
        _, _, _, unseen_h2 = unseen_fraction(entries, "cdr2")
        _, _, _, unseen_h3 = unseen_fraction(entries, "cdr3")
        rows.append((
            dataset,
            f"{unique_h1} / {total:,} ({unique_h1 / total:.1%})",
            f"{unseen_h1}/{n_test} (**{unseen_h1 / n_test:.1%}**)",
            f"{unseen_h2}/{n_test} ({unseen_h2 / n_test:.1%})",
            f"{unseen_h3}/{n_test} ({unseen_h3 / n_test:.1%})",
        ))
    return rows


def _design_rows(runs):
    """H1 against H3, every family that left a generative eval in this batch.

    H1 and H3 are the two poles of Findings §6/§7 -- near-retrieval against genuine design -- so the
    table shows those two loops rather than all six. Families come from `FAMILY_LABELS`, which is
    what lets a causal run appear here at all: `ar_scratch` writes the same `generation_eval.json`
    as a masked run, and before it was listed the AR batches rendered this table empty.
    """
    rows = []
    for loop in ("h1", "h3"):
        for family, family_label in FAMILY_LABELS.items():
            run_dir = runs.get((loop, dataset_alias(runs).get("OAS", "OAS"), "", family))
            means = _gen_means(os.path.join(run_dir, artifacts.GENERATION_EVAL_FILENAME)) if run_dir else None
            if not means:
                continue
            label = f"{loop.upper()} {family_label}"
            rows.append((
                label,
                f"{means['aar_mean']:.3f}",
                f"{means['diversity_hamming']:.3f}",
                f"{means['per_position_entropy_bits']:.3f}",
                f"{means['judge_perplexity_mean']:.3f}",
            ))
    return rows


def _length_rows(runs):
    path = data_path_for(runs, dataset_alias(runs).get("OAS", "OAS"))
    if not path or not os.path.exists(path):
        return None

    entries = _load(path)
    rows = []
    for loop in LOOPS:
        chain = "heavy" if loop.startswith("h") else "light"
        counts = Counter(len(entry[chain]["cdr" + loop[1]]) for entry in entries)
        total = sum(counts.values())
        entropy = -sum(
            (count / total) * math.log2(count / total) for count in counts.values()
        )
        rows.append((
            entropy,
            loop.upper(),
            len(counts),
            f"{counts.most_common(1)[0][1] / total:.1%}",
            f"{entropy:.2f} bits",
        ))
    return [row[1:] for row in sorted(rows, reverse=True)]


TABLE_DEFINITIONS = {
    "retrieval": {
        "title": "Findings §6 — test CDRs unseen in train (the split clusters heavy CDR3 only)",
        "headers": ("dataset", "unique H1", "test H1 unseen", "test H2 unseen", "test H3 unseen"),
        "alignments": ("left", "left", "left", "left", "left"),
        "rows": _retrieval_rows,
    },
    "design": {
        "title": "Findings §7 — recovery vs design quality (post-training generative eval)",
        "headers": ("", "recovery", "diversity", "entropy (bits)", "judge-ppl"),
        "alignments": ("left", "right", "right", "right", "right"),
        "rows": _design_rows,
    },
    "length": {
        "title": "How to read the numbers — information given away by fixing loop length",
        "missing_title": "Length entropy — (OAS data not found)",
        "headers": ("loop", "distinct lengths", "mode freq", "free info from knowing L"),
        "alignments": ("left", "right", "right", "right"),
        "rows": _length_rows,
    },
}

TABLES = (*RECOVERY_TABLES, *TABLE_DEFINITIONS)


def _decoding_table_data(paths):
    got = {}
    for path in paths or []:
        means = _gen_means(path)
        if means:
            mode = _load(path).get("decoding", {}).get("mode", os.path.basename(path))
            got[mode] = means
    if len(got) < 2:
        return None

    first, second = "iterative", "independent"
    if first not in got or second not in got:
        first, second = list(got)[:2]
    rows = [
        (
            label,
            f"{got[first][key]:.4f}",
            f"{got[second][key]:.4f}",
            f"{got[second][key] - got[first][key]:+.4f}",
        )
        for key, label in GEN_KEYS
    ]
    return {
        "title": f"Step 4 — decoding comparison ({first} vs {second}, n={got[first]['_n']})",
        "headers": ("metric", first, second, "Δ"),
        "alignments": ("left", "right", "right", "right"),
        "rows": rows,
    }


def _loss_types(runs):
    """Distinct training objectives across a batch's runs, from each run_config.json.

    A batch that mixes objectives cannot be read as one comparison: a core-weighted run is
    optimising the very number the recovery tables report, so its column is not like-for-like with
    a cross-entropy run. Old runs predate the field and report as "cross_entropy", which is what
    they trained with.
    """
    found = set()
    for run_dir in runs.values():
        meta = find_run_config(run_dir)
        found.add(_load(str(meta)).get("loss_type", "cross_entropy") if meta else "cross_entropy")
    return sorted(found)


def loss_warning(runs):
    """A banner when a batch mixes objectives, or None when every run shares one."""
    losses = _loss_types(runs)
    if len(losses) < 2:
        return None
    return (f"> ⚠️ **This batch mixes training objectives ({', '.join(losses)}).** A core-weighted "
            "run optimises the metric these tables report, so its numbers are not like-for-like "
            "with a cross-entropy run. Compare within one objective.")


def build_table(name, runs, *, decoding_paths=None):
    """Build one named comparison table through the shared Markdown renderer."""
    if name in RECOVERY_TABLES:
        return _recovery_table(name, runs)
    if name == "decoding":
        data = _decoding_table_data(decoding_paths)
        return markdown_table(**data) if data else None
    if name not in TABLE_DEFINITIONS:
        raise ValueError(f"unknown table {name!r}; choose from {','.join(TABLES)}")

    spec = TABLE_DEFINITIONS[name]
    rows = spec["rows"](runs)
    if rows is None:
        return f"#### {spec['missing_title']}"
    return markdown_table(
        spec["title"],
        spec["headers"],
        rows,
        spec["alignments"],
    )


def render_tables(batch, runs, wanted=None, *, decoding_paths=None, source=None):
    """Every requested table as one Markdown document, ready to print or write.

    Returned rather than printed so the same rendering serves the CLI, a file written automatically
    at the end of a batch, and any caller that wants the text.
    """
    wanted = list(TABLES) if wanted is None else wanted
    command = source or f"python -m cdrseqgen.visualization.batch_tables --batch {batch}"
    blocks = [
        f"<!-- generated by `{command}` -->",
        f"<!-- batch: {batch} — {len(runs)} runs -->\n",
    ]

    warning = loss_warning(runs)
    if warning:
        blocks.append(warning + "\n")

    for name in wanted:
        blocks.append(build_table(name, runs) + "\n")

    decoding = build_table("decoding", runs, decoding_paths=decoding_paths)
    if decoding:
        blocks.append(decoding + "\n")
    elif decoding_paths:
        blocks.append("<!-- decoding table skipped: need two readable generation_eval.json files -->\n")
    return "\n".join(blocks)


def write_batch_tables(batch, output_path=None, *, wanted=None, decoding_paths=None):
    """Render a batch's tables to a file. Returns the path, or None when the batch has no runs.

    Used by scripts/run.sh so a finished round leaves its tables beside its runs, the way each run
    already leaves its own curves beside its epoch_metrics.csv. Never raises on an unparseable
    batch: a reporting step must not be able to fail a matrix that already trained.
    """
    batch = os.path.realpath(batch) if os.path.islink(batch) else batch
    runs = discover_runs(batch)
    if not runs:
        print(f"⚠️ No parseable run directories under {batch} — no tables written")
        return None
    output_path = output_path or os.path.join(batch, BATCH_TABLES_FILENAME)
    text = render_tables(batch, runs, wanted, decoding_paths=decoding_paths,
                         source=f"python -m cdrseqgen.visualization.batch_tables --batch {batch}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write(text)
    print(f"✅ Tables successfully saved as: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown comparison tables from a batch.")
    parser.add_argument(
        "--batch",
        default="results/runs/latest",
        help="Batch directory holding the run folders (default: results/runs/latest).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=f"Comma-separated subset of: {','.join(TABLES)} (default: all).",
    )
    parser.add_argument(
        "--decoding-json",
        nargs="*",
        default=None,
        help="Two generation_eval.json files differing only in --decoding, for the Step 4 table.",
    )
    parser.add_argument(
        "--output",
        nargs="?",
        const="",
        default=None,
        help=f"Write to a file instead of stdout. Bare --output uses <batch>/{BATCH_TABLES_FILENAME}.",
    )
    args = parser.parse_args()

    wanted = [name.strip() for name in args.only.split(",")] if args.only else list(TABLES)
    unknown = [name for name in wanted if name not in TABLES]
    if unknown:
        raise SystemExit(f"Unknown table(s) {unknown}; choose from {','.join(TABLES)}")

    batch = os.path.realpath(args.batch) if os.path.islink(args.batch) else args.batch

    if args.output is not None:
        written = write_batch_tables(batch, args.output or None, wanted=wanted,
                                     decoding_paths=args.decoding_json)
        if written is None:
            raise SystemExit(1)
        return

    runs = discover_runs(batch)
    if not runs:
        raise SystemExit(
            f"No parseable run directories under {batch!r}. "
            "Expected names like h3_OAS_scratch (see train.py's run naming)."
        )
    print(render_tables(batch, runs, wanted, decoding_paths=args.decoding_json,
                        source=f"python -m cdrseqgen.visualization.batch_tables "
                               f"--batch {args.batch}"))


if __name__ == "__main__":
    main()
