#!/usr/bin/env python3
"""Tier 1 held-out n=1,000 context_dropout confirmation runner (partner-present vs. partner-absent).

Evaluates the existing partner-trained `cd00` (context_dropout=0.0) and `cd50` (context_dropout=0.5)
checkpoints across MLM scratch, AR scratch, S2S scratch, and S2S ESM-2 35M on the EXACT frozen
n=1,000 leakage-filtered held-out test frameworks from
`benchmark/manifests/oas4_h3_test_length_free_n1000_v1.json`, pairing each record with its exact
`light.full_aa` from `data/processed/processed_oas_multi.json` (verified against `source_index` and
`heavy.full_aa == pt1 + native + pt2`).

Every arm is evaluated under BOTH regimes:
  * `with`    -- generator receives heavy flanks + paired light chain
  * `without` -- generator receives heavy flanks only (missing-partner ablation)

All outputs are scored by one frozen partner-present IgBert judge (`Exscientia/IgBert`, max_len=384)
so partner-present vs. partner-absent deltas share one fixed judge context (matching §4.4).

Usage:
    # CPU-only dry-run verifying all 20 checkpoints, manifest, corpus hash, and light-chain pairing:
    python scripts/run_tier1_context_dropout_n1000.py --mode plan

    # 5-framework smoke check on GPU 0:
    python scripts/run_tier1_context_dropout_n1000.py --mode smoke --gpu 0

    # Full n=1,000 held-out confirmation on GPU 0 (resumable per arm/regime):
    python scripts/run_tier1_context_dropout_n1000.py --mode full --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark._common import AMINO_ACIDS, atomic_write_json, sha256_file, stable_example_seed
from benchmark.adapters.cdrseqgen import load_run_config, pretrained_input_files, resolve_decoding
from cdrseqgen.artifacts import LENGTH_CONFIG_FILENAME, LENGTH_HEAD_FILENAME
from cdrseqgen.eval.metrics import core_anchor_indices, edit_similarity, hamming_fraction, recovery_at_indices
from itertools import combinations

DEFAULT_MANIFEST = ROOT / "benchmark/manifests/oas4_h3_test_length_free_n1000_v1.json"
DEFAULT_SMOKE_MANIFEST = ROOT / "benchmark/manifests/oas4_h3_test_length_free_smoke_v1.json"
DEFAULT_CORPUS = ROOT / "data/processed/processed_oas_multi.json"

# Existing paired-trained context_dropout checkpoints (10 pairs = 20 checkpoints).
DROPOUT_ARMS: list[dict[str, Any]] = [
    # Scratch MLM (seeds 42-43)
    {"label": "mlm_scratch_cd00_s42", "family": "mlm_scratch", "dropout": 0.0, "seed": 42, "method": "mlm",
     "checkpoint": "results/runs/20261007_163535_226387_mlm_dropout_screen/h3_OAS4_LC_scratch_mlm_scratch_cd00_s42"},
    {"label": "mlm_scratch_cd50_s42", "family": "mlm_scratch", "dropout": 0.5, "seed": 42, "method": "mlm",
     "checkpoint": "results/runs/20261007_163535_226387_mlm_dropout_screen/h3_OAS4_LC_scratch_mlm_scratch_cd50_s42"},
    {"label": "mlm_scratch_cd00_s43", "family": "mlm_scratch", "dropout": 0.0, "seed": 43, "method": "mlm",
     "checkpoint": "results/runs/20261007_163535_226387_mlm_dropout_screen/h3_OAS4_LC_scratch_mlm_scratch_cd00_s43"},
    {"label": "mlm_scratch_cd50_s43", "family": "mlm_scratch", "dropout": 0.5, "seed": 43, "method": "mlm",
     "checkpoint": "results/runs/20261007_163535_226387_mlm_dropout_screen/h3_OAS4_LC_scratch_mlm_scratch_cd50_s43"},
    # Scratch AR (seeds 42-43)
    {"label": "ar_scratch_cd00_s42", "family": "ar_scratch", "dropout": 0.0, "seed": 42, "method": "ar",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_ar_scratch_ar_scratch_cd00_s42"},
    {"label": "ar_scratch_cd50_s42", "family": "ar_scratch", "dropout": 0.5, "seed": 42, "method": "ar",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_ar_scratch_ar_scratch_cd50_s42"},
    {"label": "ar_scratch_cd00_s43", "family": "ar_scratch", "dropout": 0.0, "seed": 43, "method": "ar",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_ar_scratch_ar_scratch_cd00_s43"},
    {"label": "ar_scratch_cd50_s43", "family": "ar_scratch", "dropout": 0.5, "seed": 43, "method": "ar",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_ar_scratch_ar_scratch_cd50_s43"},
    # Scratch S2S (seeds 42-43)
    {"label": "s2s_scratch_cd00_s42", "family": "s2s_scratch", "dropout": 0.0, "seed": 42, "method": "s2s",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_s2s_scratch_s2s_scratch_cd00_s42"},
    {"label": "s2s_scratch_cd50_s42", "family": "s2s_scratch", "dropout": 0.5, "seed": 42, "method": "s2s",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_s2s_scratch_s2s_scratch_cd50_s42"},
    {"label": "s2s_scratch_cd00_s43", "family": "s2s_scratch", "dropout": 0.0, "seed": 43, "method": "s2s",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_s2s_scratch_s2s_scratch_cd00_s43"},
    {"label": "s2s_scratch_cd50_s43", "family": "s2s_scratch", "dropout": 0.5, "seed": 43, "method": "s2s",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_s2s_scratch_s2s_scratch_cd50_s43"},
    # S2S ESM-2 35M (seeds 42-45)
    {"label": "s2s_esm2_cd00_s42", "family": "s2s_esm2", "dropout": 0.0, "seed": 42, "method": "s2s",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_s2s_esm2_t12_35M_UR50D_s2s_esm2_cd00_s42"},
    {"label": "s2s_esm2_cd50_s42", "family": "s2s_esm2", "dropout": 0.5, "seed": 42, "method": "s2s",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_s2s_esm2_t12_35M_UR50D_s2s_esm2_cd50_s42"},
    {"label": "s2s_esm2_cd00_s43", "family": "s2s_esm2", "dropout": 0.0, "seed": 43, "method": "s2s",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_s2s_esm2_t12_35M_UR50D_s2s_esm2_cd00_s43"},
    {"label": "s2s_esm2_cd50_s43", "family": "s2s_esm2", "dropout": 0.5, "seed": 43, "method": "s2s",
     "checkpoint": "results/runs/20260915_131629_103719_context_dropout/h3_OAS4_LC_s2s_esm2_t12_35M_UR50D_s2s_esm2_cd50_s43"},
    {"label": "s2s_esm2_cd00_s44", "family": "s2s_esm2", "dropout": 0.0, "seed": 44, "method": "s2s",
     "checkpoint": "results/runs/20261007_013541_142543_esm2_dropout_confirmation/h3_OAS4_LC_s2s_esm2_t12_35M_UR50D_s2s_esm2_cd00_s44"},
    {"label": "s2s_esm2_cd50_s44", "family": "s2s_esm2", "dropout": 0.5, "seed": 44, "method": "s2s",
     "checkpoint": "results/runs/20261007_013541_142543_esm2_dropout_confirmation/h3_OAS4_LC_s2s_esm2_t12_35M_UR50D_s2s_esm2_cd50_s44"},
    {"label": "s2s_esm2_cd00_s45", "family": "s2s_esm2", "dropout": 0.0, "seed": 45, "method": "s2s",
     "checkpoint": "results/runs/20261007_013541_142543_esm2_dropout_confirmation/h3_OAS4_LC_s2s_esm2_t12_35M_UR50D_s2s_esm2_cd00_s45"},
    {"label": "s2s_esm2_cd50_s45", "family": "s2s_esm2", "dropout": 0.5, "seed": 45, "method": "s2s",
     "checkpoint": "results/runs/20261007_013541_142543_esm2_dropout_confirmation/h3_OAS4_LC_s2s_esm2_t12_35M_UR50D_s2s_esm2_cd50_s45"},
]


def load_paired_records(manifest_path: Path, corpus_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    expected_corpus_hash = (manifest.get("source") or {}).get("processed_file_sha256")
    actual_corpus_hash = sha256_file(corpus_path)
    if expected_corpus_hash and actual_corpus_hash != expected_corpus_hash:
        raise ValueError(f"corpus hash mismatch: {actual_corpus_hash} != {expected_corpus_hash}")
    corpus = json.loads(corpus_path.read_text())

    paired = []
    for rec in manifest["records"]:
        idx = rec["source_index"]
        entry = corpus[idx]
        if entry["heavy"]["full_aa"] != rec["parent_sequence"]:
            raise ValueError(f"{rec['example_id']} source_index={idx} heavy sequence mismatch")
        light = (entry.get("light") or {}).get("full_aa")
        if not light:
            raise ValueError(f"{rec['example_id']} source_index={idx} lacks paired light.full_aa")
        paired.append({**rec, "partner_chain": light})
    return manifest, paired


def validate_arms(arms: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_hash = (manifest.get("source") or {}).get("processed_file_sha256")
    validated = []
    for arm in arms:
        ckpt = ROOT / arm["checkpoint"]
        cfg, cfg_path = load_run_config(ckpt)
        if not (ckpt / "model.safetensors").is_file() or not (ckpt / "config.json").is_file():
            raise FileNotFoundError(f"{arm['label']}: missing model.safetensors or config.json in {ckpt}")
        if not cfg.get("include_partner_chain", False):
            raise ValueError(f"{arm['label']}: expected include_partner_chain=True, got {cfg.get('include_partner_chain')!r}")
        if cfg.get("target_region") != "h3":
            raise ValueError(f"{arm['label']}: expected target_region='h3', got {cfg.get('target_region')!r}")
        if cfg.get("seed") != arm["seed"]:
            raise ValueError(f"{arm['label']}: seed mismatch ({cfg.get('seed')} != {arm['seed']})")
        if abs(float(cfg.get("context_dropout", 0.0)) - arm["dropout"]) > 1e-6:
            raise ValueError(f"{arm['label']}: context_dropout mismatch ({cfg.get('context_dropout')} != {arm['dropout']})")
        if cfg.get("model_type") != "scratch":
            pretrained_input_files(ckpt)
        if arm["method"] == "mlm":
            for fname in (LENGTH_CONFIG_FILENAME, LENGTH_HEAD_FILENAME):
                if not (ckpt / fname).is_file():
                    raise FileNotFoundError(f"{arm['label']}: missing {fname} in {ckpt}")
        declared_hash = (cfg.get("provenance", {}).get("dataset") or {}).get("declared_processed_file_sha256")
        if declared_hash and manifest_hash and declared_hash != manifest_hash:
            raise ValueError(f"{arm['label']}: corpus hash mismatch")
        validated.append({
            **arm,
            "checkpoint_path": str(ckpt),
            "run_config": cfg,
            "run_config_path": str(cfg_path),
            "decoding": resolve_decoding(cfg),
        })
    return validated


def sample_arm_regime(
    arm: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    regime: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    device_str: str,
) -> list[dict[str, Any]]:
    import torch
    from cdrseqgen.methods.registry import checkpoint_method

    ckpt = Path(arm["checkpoint_path"])
    cfg = arm["run_config"]
    method = checkpoint_method(cfg)
    max_len = int(cfg["max_len"])
    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available() else device_str)

    backbone = method.module("backbones").resolve(cfg["model_type"])
    tokenizer = backbone.build_tokenizer(cfg["model_type"], str(ckpt))
    model = backbone.load_pretrained_model(str(ckpt)).to(device)
    model.eval()

    if method.name == "mlm":
        from cdrseqgen.methods.mlm.length import LengthPredictor, resolve_length_policy, sample_length_conditioned_spans
        length_predictor = LengthPredictor.load(ckpt, device=device)
        policy = resolve_length_policy()
    elif method.name == "ar":
        from cdrseqgen.datasets.dataset_ar import build_causal_prompt
        sample_loops = method.module("sampling").sample_loops
        max_new_tokens = int(cfg.get("gen_max_new_tokens", 97))
    elif method.name == "s2s":
        from cdrseqgen.datasets.dataset_s2s import build_encoder_input
        sample_loops = method.module("sampling").sample_loops
        max_new_tokens = int(cfg.get("gen_max_new_tokens", 97))
    else:
        raise ValueError(f"unsupported method {method.name!r}")

    out = []
    use_partner = (regime == "with")
    for rec in records:
        seed = stable_example_seed(base_seed, rec["example_id"])
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        partner = rec["partner_chain"] if use_partner else None
        t0 = time.perf_counter()

        if method.name == "mlm":
            spans, _ = sample_length_conditioned_spans(
                model, tokenizer, device, gen,
                pt1=rec["pt1"], pt2=rec["pt2"], policy=policy,
                length_predictor=length_predictor,
                partner_chain=partner, antigen=None,
                max_len=max_len, decoding=arm["decoding"],
                num_samples=num_samples, temperature=temperature,
                top_k=0, top_p=top_p, length_temperature=1.0,
            )
            raw_loops = [tokenizer.decode(sp.token_ids).upper().strip() for sp in spans]
            valid_loops = [lp for lp in raw_loops if lp and not (set(lp) - AMINO_ACIDS)]
        else:
            fw = dict(pt1=rec["pt1"], pt2=rec["pt2"], target_length=0,
                      max_len=max_len, partner_chain=partner, antigen=None,
                      target_region="h3")
            prompt_ids, _ = (build_causal_prompt if method.name == "ar" else build_encoder_input)(tokenizer, fw)
            draws = sample_loops(
                model, tokenizer, prompt_ids,
                device=device, generator=gen, num_samples=num_samples,
                temperature=temperature, top_k=0, top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            valid_loops = [
                d["loop"] for d in draws
                if d.get("complete", True) and d.get("valid", True) and d.get("loop")
                and not (set(d["loop"]) - AMINO_ACIDS)
            ]

        native = rec["native"]
        n_len = len(native)
        core_idx, _ = core_anchor_indices(n_len, 2)
        matched = [s for s in valid_loops if len(s) == n_len]
        pair_div = [hamming_fraction(a, b) for a, b in combinations(valid_loops, 2)]
        out.append({
            "example_id": rec["example_id"],
            "native": native,
            "native_length": n_len,
            "attempts": num_samples,
            "valid_attempts": len(valid_loops),
            "samples": valid_loops,
            "edit_similarity_mean": statistics.mean(edit_similarity(s, native) for s in valid_loops) if valid_loops else None,
            "length_mae_mean": statistics.mean(abs(len(s) - n_len) for s in valid_loops) if valid_loops else None,
            "length_match_rate": (len(matched) / len(valid_loops)) if valid_loops else None,
            "core_aar_mean": statistics.mean(recovery_at_indices(s, native, core_idx) for s in matched) if (matched and core_idx) else None,
            "diversity_hamming": statistics.mean(pair_div) if pair_div else (0.0 if valid_loops else None),
            "runtime_seconds": time.perf_counter() - t0,
        })

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def score_with_partner_judge(records: list[dict[str, Any]], paired_lookup: dict[str, dict[str, Any]],
                             judge_name: str, max_len: int, device_str: str,
                             batch_size: int = 256) -> None:
    """Attach partner-present IgBert pseudo-perplexity using exact length-grouped batching + cache."""
    import math
    import torch
    from cdrseqgen.datasets.dataset_mlm import assemble_masked_input, inject_context
    from cdrseqgen.methods.mlm.backbones import resolve
    from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood_batch

    device = torch.device("cuda" if device_str == "auto" and torch.cuda.is_available() else device_str)
    backbone = resolve(judge_name)
    tokenizer = backbone.build_tokenizer(judge_name)
    model = backbone.load_pretrained_model(judge_name).to(device)
    model.eval()

    cache: dict[tuple[str, str], float | None] = {}
    for rec in records:
        ex_id = rec["example_id"]
        samples = rec["samples"]
        uncached = []
        seen = set()
        for loop in samples:
            if (ex_id, loop) not in cache and loop not in seen:
                seen.add(loop)
                uncached.append(loop)
        if uncached:
            ref = paired_lookup[ex_id]
            pt1_ids = tokenizer.encode(ref["pt1"])
            pt2_ids = tokenizer.encode(ref["pt2"])
            prepared_by_len: dict[int, tuple[list[int], list[int]]] = {}
            batch_items = []
            for loop in uncached:
                loop_ids = tokenizer.encode(loop)
                n = len(loop_ids)
                if n not in prepared_by_len:
                    j_pt1, j_pt2 = inject_context(
                        tokenizer, list(pt1_ids), list(pt2_ids), n,
                        partner_seq=ref["partner_chain"], antigen_seq=None, max_len=max_len,
                    )
                    input_ids, mask_start = assemble_masked_input(tokenizer, j_pt1, j_pt2, n)
                    prepared_by_len[n] = (input_ids, list(range(mask_start, mask_start + n)))
                input_ids, mask_pos = prepared_by_len[n]
                batch_items.append((input_ids, mask_pos, loop_ids))
            scored = pseudo_log_likelihood_batch(
                model, batch_items, tokenizer.special_ids, tokenizer.mask_id, device,
                batch_size=batch_size,
            )
            for loop, (_, ppl) in zip(uncached, scored):
                cache[(ex_id, loop)] = float(ppl) if (ppl is not None and math.isfinite(ppl)) else None

        ppls = [cache[(ex_id, loop)] for loop in samples if cache[(ex_id, loop)] is not None]
        rec["judge_perplexity_mean"] = statistics.mean(ppls) if ppls else None
        rec["judge_candidates_scored"] = len(ppls)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_key(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    return {
        "frameworks": len(records),
        "attempts": sum(r["attempts"] for r in records),
        "valid_attempts": sum(r["valid_attempts"] for r in records),
        "edit_similarity_mean": mean_key("edit_similarity_mean"),
        "length_mae_mean": mean_key("length_mae_mean"),
        "length_match_rate": mean_key("length_match_rate"),
        "core_aar_mean": mean_key("core_aar_mean"),
        "diversity_hamming": mean_key("diversity_hamming"),
        "judge_perplexity_mean": mean_key("judge_perplexity_mean"),
    }


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_json(out_dir / "summary.json", {"rows": rows}, allow_nan=False)
    lines = [
        "# Held-Out OAS4 Test Context-Dropout Confirmation",
        "",
        "| Arm | Family | Dropout | Seed | Regime | Edit sim. | MAE | Match (%) | Div. | Partner Judge PPL | Valid |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        a = r["aggregate"]
        fmt = lambda v, f=".4f": f"{v:{f}}" if v is not None else "—"
        match = f"{100 * a['length_match_rate']:.2f}" if a["length_match_rate"] is not None else "—"
        lines.append(
            f"| {r['label']} | {r['family']} | {r['dropout']:.1f} | {r['seed']} | {r['regime']} | "
            f"{fmt(a['edit_similarity_mean'])} | {fmt(a['length_mae_mean'], '.3f')} | {match} | "
            f"{fmt(a['diversity_hamming'])} | {fmt(a['judge_perplexity_mean'], '.3f')} | "
            f"{a['valid_attempts']}/{a['attempts']} |"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("plan", "smoke", "full"), required=True)
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value for smoke/full.")
    p.add_argument("--output", type=Path, default=None, help="Output directory (defaults to timestamped results/benchmark/...).")
    p.add_argument("--arms", nargs="*", default=None, help="Optional subset of arm labels or families.")
    p.add_argument("--regimes", nargs="+", choices=("with", "without"), default=["with", "without"])
    p.add_argument("--judge", default="Exscientia/IgBert")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest_path = DEFAULT_SMOKE_MANIFEST if args.mode == "smoke" else DEFAULT_MANIFEST
    num_samples = 5 if args.mode == "smoke" else 20

    arms = DROPOUT_ARMS
    if args.arms:
        wanted = set(args.arms)
        arms = [a for a in DROPOUT_ARMS if a["label"] in wanted or a["family"] in wanted]
        if not arms:
            raise ValueError(f"no dropout arms matched {args.arms!r}")
    elif args.mode == "smoke":
        # Smoke tests one pair per family (seed 42).
        arms = [a for a in DROPOUT_ARMS if a["seed"] == 42]

    manifest, paired = load_paired_records(manifest_path, DEFAULT_CORPUS)
    validated = validate_arms(arms, manifest)

    print(f"Plan OK: mode={args.mode} manifest={manifest_path.name} frameworks={len(paired)} "
          f"arms={len(validated)} regimes={args.regimes} samples={num_samples} "
          f"total_draws={len(paired) * len(validated) * len(args.regimes) * num_samples:,}")
    for a in validated:
        print(f"  - {a['label']:24s} ({a['method']:3s}, d={a['dropout']:.1f}, s={a['seed']}, decoding={a['decoding']})")
    if args.mode == "plan":
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = args.output or (ROOT / f"results/benchmark/{ts}_oas4_h3_test_context_dropout_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paired_lookup = {r["example_id"]: r for r in paired}

    summary_rows = []
    for arm in validated:
        for regime in args.regimes:
            out_file = out_dir / "generations" / f"{arm['label']}__{regime}.json"
            if out_file.is_file():
                payload = json.loads(out_file.read_text())
                print(f"[resume] {arm['label']} ({regime}) already complete", flush=True)
            else:
                print(f"[run] {arm['label']} ({regime}) on {len(paired)} frameworks x {num_samples} draws...", flush=True)
                recs = sample_arm_regime(
                    arm, paired, regime=regime, num_samples=num_samples,
                    temperature=1.0, top_p=0.9, base_seed=args.seed, device_str="auto",
                )
                if not args.skip_judge:
                    score_with_partner_judge(recs, paired_lookup, args.judge, max_len=384, device_str="auto")
                payload = {
                    "label": arm["label"],
                    "family": arm["family"],
                    "dropout": arm["dropout"],
                    "seed": arm["seed"],
                    "method": arm["method"],
                    "regime": regime,
                    "manifest": str(manifest_path.relative_to(ROOT)),
                    "checkpoint": arm["checkpoint"],
                    "judge": None if args.skip_judge else {"name": args.judge, "partner_context": "with", "max_len": 384},
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "aggregate": aggregate(recs),
                    "records": recs,
                }
                atomic_write_json(out_file, payload, allow_nan=False)
            summary_rows.append({
                "label": arm["label"], "family": arm["family"], "dropout": arm["dropout"],
                "seed": arm["seed"], "method": arm["method"], "regime": regime,
                "aggregate": payload["aggregate"],
            })
            write_summary(out_dir, summary_rows)

    print(f"Done! Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
