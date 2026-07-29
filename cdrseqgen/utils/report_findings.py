"""Regenerate every Findings table in the README from a finished batch.

The numbers in README "Findings" and "How to read the numbers" are derived, not hand-written. This
script is the derivation: point it at a batch and it reprints each table as markdown, ready to paste.

    python -m cdrseqgen.utils.report_findings                          # results/latest
    python -m cdrseqgen.utils.report_findings --batch results/20260724_211745
    python -m cdrseqgen.utils.report_findings --only loops,design

  * discover_runs   -- parse a batch dir into {(loop, dataset, context, model): run_dir}
  * best_epoch      -- (value, epoch) at the best epoch of one run, on the right metric
  * unseen_fraction -- how much of a test split has a CDR never seen in train
  * table_*         -- one function per README table, each returning markdown
  * main            -- CLI: pick a batch, print the tables

Notes:
  * Run dirs are parsed, not hardcoded, so a different ESM-2 variant or an added context flag is
    picked up automatically. Naming comes from train.py: <loop>_<DS>[_LC|_HC][_AG]_<model>.
  * L2 is reported on WHOLE-SPAN, every other loop on core. ANCHOR_TRIM=2 leaves L2 (98.8% of its
    loops are 3 residues) with core over ~1.4% of eval rows, which is noise — see the README.
  * Reads only artifacts a finished run already writes (epoch_metrics.csv, generation_eval.json,
    run_meta.json) plus the processed dataset JSONs. Nothing is recomputed on a GPU.
  * The decoding table is the one thing that needs an extra step: two gen_eval.py runs differing
    only in --decoding, passed via --decoding-json. Skipped when absent.
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

# <loop>_<DS>[_LC|_HC][_AG]_<model>, e.g. h3_OAS_LC_esm2_t12_35M_UR50D, h1_SD2_LC_AG_scratch
RUN_RE = re.compile(r"^(?P<loop>[hl][123])_(?P<ds>OAS|SD2|DS)(?P<ctx>(?:_LC|_HC|_AG)*)_(?P<model>.+)$")
LOOPS = ["h1", "h2", "h3", "l1", "l2", "l3"]
DEFAULT_DATA = {"OAS": "dataset/data/processed/processed_oas_data.json",
                "SD2": "dataset/data/processed/processed_sabdab2_data.json"}


def metric_for(loop):
    """Which recovery column to report for a loop: core everywhere except L2 (there it is noise)"""
    return "eval_accuracy" if loop == "l2" else "eval_core_accuracy"


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


def best_epoch(run_dir, metric):
    """(value, epoch) at the row maximising `metric`; (None, None) if the run has no CSV/column"""
    csv = os.path.join(run_dir, "epoch_metrics.csv")
    if not os.path.exists(csv):
        return None, None
    df = pd.read_csv(csv)
    if metric not in df.columns or df.empty:
        return None, None
    i = df[metric].idxmax()
    return float(df[metric][i]), int(df["epoch"][i])


def _load(path):
    with open(path) as f:
        return json.load(f)


def data_path_for(runs, ds):
    """Prefer the data_path a run recorded in run_meta.json; fall back to the standard location"""
    for (_, d, _, _), run_dir in runs.items():
        if d != ds:
            continue
        meta = os.path.join(run_dir, "run_meta.json")
        if os.path.exists(meta):
            p = _load(meta).get("data_path")
            if p and os.path.exists(p):
                return p
    return DEFAULT_DATA.get(ds)


# --- tables -----------------------------------------------------------------------------------

def table_loops(runs):
    """README Findings §2 — best-epoch recovery for all six loops, with and without partner chain"""
    out = ["#### Findings §2 — best-epoch recovery, OAS (core; L2 whole-span)", "",
           "| | scratch | +ctx | ESM-2 | +ctx |", "|---|---|---|---|---|"]
    for loop in LOOPS:
        ctx = "_HC" if loop.startswith("l") else "_LC"
        cells = []
        for c, fam in [("", "scratch"), (ctx, "scratch"), ("", "esm2"), (ctx, "esm2")]:
            r = runs.get((loop, "OAS", c, fam))
            v, _ = best_epoch(r, metric_for(loop)) if r else (None, None)
            cells.append(f"{v:.4f}" if v is not None else "—")
        out.append(f"| {loop.upper()} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_antigen(runs):
    """README Findings §4 — the same-data antigen ablation on SAbDab2 (_LC vs _LC_AG)"""
    out = ["#### Findings §4 — antigen ablation, SAbDab2 (core @ best epoch)", "",
           "| CDR | scratch lc-only → +ag | ESM-2 lc-only → +ag |", "|---|---|---|"]
    for loop in ["h1", "h2", "h3"]:
        row = []
        for fam in ["scratch", "esm2"]:
            a, _ = best_epoch(runs.get((loop, "SD2", "_LC", fam)) or "", metric_for(loop))
            b, _ = best_epoch(runs.get((loop, "SD2", "_LC_AG", fam)) or "", metric_for(loop))
            row.append(f"{a:.4f} → {b:.4f}" if a is not None and b is not None else "—")
        out.append(f"| {loop.upper()} | " + " | ".join(row) + " |")
    return "\n".join(out)


def unseen_fraction(entries, cdr, chain="heavy"):
    """(n_unique, n_total, n_test, n_test_with_unseen_cdr) for one CDR under the entries' split"""
    vals = [e[chain][cdr] for e in entries]
    train = {e[chain][cdr] for e in entries if e.get("split") == "train"}
    test = [e for e in entries if e.get("split") not in (None, "train")]
    unseen = sum(1 for e in test if e[chain][cdr] not in train)
    return len(set(vals)), len(vals), len(test), unseen


def table_retrieval(runs):
    """README Findings §6 — how much of each test split is answerable by lookup"""
    out = ["#### Findings §6 — test CDRs unseen in train (the split clusters heavy CDR3 only)", "",
           "| dataset | unique H1 | test H1 unseen | test H2 unseen | test H3 unseen |",
           "|---|---|---|---|---|"]
    for ds in ["OAS", "SD2"]:
        p = data_path_for(runs, ds)
        if not p or not os.path.exists(p):
            out.append(f"| {ds} | (data not found) | | | |")
            continue
        d = _load(p)
        u1, n, n_te, un1 = unseen_fraction(d, "cdr1")
        _, _, _, un2 = unseen_fraction(d, "cdr2")
        _, _, _, un3 = unseen_fraction(d, "cdr3")
        out.append(f"| {ds} | {u1} / {n:,} ({u1/n:.1%}) | {un1}/{n_te} (**{un1/n_te:.1%}**) "
                   f"| {un2}/{n_te} ({un2/n_te:.1%}) | {un3}/{n_te} ({un3/n_te:.1%}) |")
    return "\n".join(out)


def table_regimes(runs):
    """README Findings §1 — does pretraining pay? keyed on how memorisable the test split is"""
    rows = [("OAS H1", "h1", "OAS", ""), ("SD2 H1", "h1", "SD2", "_LC"),
            ("SD2 H2", "h2", "SD2", "_LC"), ("OAS H3", "h3", "OAS", "")]
    out = ["#### Findings §1 — pretraining pays by regime", "",
           "| case | unseen in train | scratch | ESM-2 | Δ |", "|---|---|---|---|---|"]
    unseen_cache = {}
    for label, loop, ds, ctx in rows:
        p = data_path_for(runs, ds)
        if p and os.path.exists(p):
            if ds not in unseen_cache:
                unseen_cache[ds] = _load(p)
            _, _, n_te, un = unseen_fraction(unseen_cache[ds], "cdr" + loop[1])
            frac = f"{un/n_te:.1%}" if n_te else "—"
        else:
            frac = "—"
        s, _ = best_epoch(runs.get((loop, ds, ctx, "scratch")) or "", metric_for(loop))
        e, _ = best_epoch(runs.get((loop, ds, ctx, "esm2")) or "", metric_for(loop))
        d = f"{e-s:+.3f}" if s is not None and e is not None else "—"
        out.append(f"| {label} | {frac} | {s:.4f} | {e:.4f} | **{d}** |"
                   if s is not None and e is not None else f"| {label} | {frac} | — | — | — |")
    return "\n".join(out)


GEN_KEYS = [("aar_mean", "recovery"), ("diversity_hamming", "diversity"),
            ("per_position_entropy_bits", "entropy(b)"), ("unique_fraction", "unique"),
            ("self_perplexity_mean", "self-ppl"), ("judge_perplexity_mean", "judge-ppl")]


def _gen_means(path):
    """Mean of each generative metric over the records of one generation_eval.json"""
    if not path or not os.path.exists(path):
        return None
    recs = _load(path).get("records", [])
    means = {}
    for k, _ in GEN_KEYS:
        v = [r["metrics"][k] for r in recs if r["metrics"].get(k) is not None]
        means[k] = statistics.mean(v) if v else float("nan")
    means["_n"] = len(recs)
    return means


def table_design(runs, loops=("h1", "h3")):
    """README Findings §7 — recovery vs design quality, from the post-training generative eval"""
    out = ["#### Findings §7 — recovery vs design quality (post-training generative eval)", "",
           "| | recovery | diversity | entropy (bits) | judge-ppl |", "|---|---|---|---|---|"]
    for loop in loops:
        for fam in ["scratch", "esm2"]:
            r = runs.get((loop, "OAS", "", fam))
            m = _gen_means(os.path.join(r, "generation_eval.json")) if r else None
            if not m:
                continue
            name = f"{loop.upper()} {'scratch' if fam=='scratch' else 'ESM-2'}"
            out.append(f"| {name} | {m['aar_mean']:.3f} | {m['diversity_hamming']:.3f} "
                       f"| {m['per_position_entropy_bits']:.3f} | {m['judge_perplexity_mean']:.3f} |")
    return "\n".join(out)


def table_length_entropy(runs):
    """README "How to read the numbers" — free information handed over by fixing the loop length"""
    p = data_path_for(runs, "OAS")
    if not p or not os.path.exists(p):
        return "#### Length entropy — (OAS data not found)"
    d = _load(p)
    out = ["#### How to read the numbers — information given away by fixing loop length", "",
           "| loop | distinct lengths | mode freq | free info from knowing L |", "|---|---|---|---|"]
    rows = []
    for loop in LOOPS:
        chain = "heavy" if loop.startswith("h") else "light"
        c = Counter(len(e[chain]["cdr" + loop[1]]) for e in d)
        n = sum(c.values())
        H = -sum((v / n) * math.log2(v / n) for v in c.values())
        rows.append((H, loop, len(c), c.most_common(1)[0][1] / n))
    for H, loop, ndist, modef in sorted(rows, reverse=True):
        out.append(f"| {loop.upper()} | {ndist} | {modef:.1%} | {H:.2f} bits |")
    return "\n".join(out)


def table_decoding(paths):
    """README Step 4 — iterative vs independent, from two gen_eval.py runs that differ only there"""
    got = {}
    for p in paths or []:
        m = _gen_means(p)
        if m:
            got[_load(p).get("decoding", {}).get("mode", os.path.basename(p))] = m
    if len(got) < 2:
        return None
    a, b = "iterative", "independent"
    if a not in got or b not in got:
        a, b = list(got)[:2]
    out = [f"#### Step 4 — decoding comparison ({a} vs {b}, n={got[a]['_n']})", "",
           f"| metric | {a} | {b} | Δ |", "|---|---|---|---|"]
    for k, lab in GEN_KEYS:
        out.append(f"| {lab} | {got[a][k]:.4f} | {got[b][k]:.4f} | {got[b][k]-got[a][k]:+.4f} |")
    return "\n".join(out)


TABLES = {
    "regimes": table_regimes,
    "loops": table_loops,
    "antigen": table_antigen,
    "retrieval": table_retrieval,
    "design": table_design,
    "length": table_length_entropy,
}


def main():
    ap = argparse.ArgumentParser(description="Regenerate the README Findings tables from a batch.")
    ap.add_argument("--batch", default="results/latest",
                    help="Batch directory holding the run folders (default: results/latest).")
    ap.add_argument("--only", default=None,
                    help=f"Comma-separated subset of: {','.join(TABLES)} (default: all).")
    ap.add_argument("--decoding-json", nargs="*", default=None,
                    help="Two generation_eval.json files differing only in --decoding, for the Step 4 table.")
    args = ap.parse_args()

    # Validate before doing any work, so a typo fails immediately instead of after a page of output.
    wanted = [t.strip() for t in args.only.split(",")] if args.only else list(TABLES)
    unknown = [t for t in wanted if t not in TABLES]
    if unknown:
        raise SystemExit(f"Unknown table(s) {unknown}; choose from {','.join(TABLES)}")

    batch = os.path.realpath(args.batch) if os.path.islink(args.batch) else args.batch
    runs = discover_runs(batch)
    if not runs:
        raise SystemExit(f"No parseable run directories under {batch!r}. "
                         f"Expected names like h3_OAS_scratch (see train.py's run naming).")

    print(f"<!-- generated by `python -m cdrseqgen.utils.report_findings --batch {args.batch}` -->")
    print(f"<!-- batch: {batch} — {len(runs)} runs -->\n")

    for name in wanted:
        print(TABLES[name](runs), end="\n\n")

    dec = table_decoding(args.decoding_json)
    if dec:
        print(dec, end="\n\n")
    elif args.decoding_json:
        print("<!-- decoding table skipped: need two readable generation_eval.json files -->\n")


if __name__ == "__main__":
    main()
