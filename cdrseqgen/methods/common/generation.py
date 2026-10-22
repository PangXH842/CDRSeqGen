"""Shared generation command lifecycle for all methods.

A method supplies option validation, optional state loading and its per-case
producer. This module checks compatibility, loads cases/model/tokenizer, seeds
sampling, preserves case order and writes the scored output payload.
"""
import json
import logging
import os

import torch

from cdrseqgen.artifacts import load_run_config
from cdrseqgen.methods.registry import resolve

logger = logging.getLogger(__name__)


def get_configs(args):
    """Reads configs from run_config.json; if unavailable, build configs from CLI args.

    Returns a config dict, carrying a "source" key for which of the two it came from. That
    difference matters downstream: a CLI-built config only restates this invocation, so it cannot
    be used to check the invocation against (see check_compatibility).
    """
    # A corrupt run config is deliberately fatal. Falling back to CLI args would set
    # source="cli", which makes check_compatibility skip both context checks — the failure mode
    # that costs 0.98 -> 0.26 recovery — so an unreadable file must stop the run, not soften it.
    try:
        configs, _ = load_run_config(args.model_path)
    except FileNotFoundError:
        configs = {}

    if configs:
        configs["source"] = "run_config"
        # --model_type picks the tokenizer as well as the backbone, so the wrong one mis-encodes
        # every residue. Only warn when a flag was actually passed and disagrees — omitting it just
        # means "use whatever this checkpoint is", which is the resolution on the next line.
        trained_type = configs.get("model_type")
        if args.model_type and trained_type and args.model_type != trained_type:
            logger.warning(f"⚠️  Model type mismatch: checkpoint was trained as '{trained_type}', but "
                           f"--model_type says '{args.model_type}' — the tokenizer comes from that "
                           f"flag, so every residue would be encoded against the wrong vocabulary.")
        configs["model_type"] = args.model_type or trained_type or "scratch"
    else:
        logger.info("ℹ️  No run_config.json - building configs from CLI args instead.")
        configs = {"target_region": args.target_loop,
                   "model_type": args.model_type or "scratch",
                   "include_partner_chain": bool(args.partner_chain),
                   "include_antigen": bool(args.antigen),
                   "data_path": None,
                   "max_len": args.max_len or 256,
                   "source": "cli"}

    # Print configs
    ctx = [n for f, n in [("include_partner_chain", "partner chain"),
                          ("include_antigen", "antigen")] if configs.get(f)]
    lead = "🧬 Checkpoint expects" if configs["source"] == "run_config" else "🧬 Assuming from CLI args"
    print(f"{lead}: CDR '{configs.get('target_region') or 'unknown'}'"
          + (f", with {' + '.join(ctx)} context" if ctx else ", framework context only"))
    if args.target_loop:
        print(f"🎯 This run is designing: CDR '{args.target_loop}'.")

    return configs


def get_test_case(args):
    """Test cases to run: a JSON batch if --test_file, else the single case built from CLI args."""
    if args.test_file:
        with open(args.test_file, "r") as f:
            test_cases = json.load(f)
    else:
        test_cases = [{
            "name": "CLI Test Case",
            "pt1": args.pt1,
            "pt2": args.pt2,
            "target_truth": args.target_truth,
            "target_length": args.target_length,
            "min_length": getattr(args, "min_length", None),
            "max_length": getattr(args, "max_length", None),
        }]
    return test_cases


# Where the loop sits along the chain, as the length of pt1 (everything before the gap). This is
# the ONLY loop signal present in the input: pt1 for a CDR1 target is just fwr1, for CDR2 it is
# fwr1+cdr1+fwr2, and for CDR3 everything up to fwr3. Measured over all 25,090 entries of both
# processed datasets (both chains, so 50,180 frameworks) the three are cleanly separated:
#
#     CDR1   8 .. 26        CDR2  33 .. 57        CDR3  78 .. 106
#
# with empty gaps at 26..33 and 57..78; the cuts below sit in the middle of those. The wide floors
# come from SAbDab2, whose crystal structures are often ragged at the N-terminus (CDR1 pt1 down to
# 8). What this CANNOT do is separate heavy from light: H1 (23-25) and L1 (24-26) overlap outright,
# as do H2 and L2. So the number is inferable and the chain is not, and the check below only ever
# claims the number.
LOOP_PT1_BANDS = [(1, 0, 30), (2, 31, 67), (3, 68, 10**6)]


def infer_loop_number(pt1_len):
    """Which CDR (1/2/3) a gap at this offset looks like; None if pt1 is too short to judge"""
    if pt1_len < 8:      # below every framework observed — a deliberately truncated stub
        return None
    for number, lo, hi in LOOP_PT1_BANDS:
        if lo <= pt1_len <= hi:
            return number
    return None


def check_compatibility(configs, cases, args):
    """Check for arg mismatches, returns problems and warnings for logging / printing."""
    problems, warnings = [], []
    trained_for = (configs or {}).get("target_region")

    # WARNING: CLI arg target_loop mismatch
    if args.target_loop and trained_for and args.target_loop.lower() != trained_for.lower():
        warnings.append(
            f"LOOP: checkpoint was trained to design CDR '{trained_for}', but --target_loop says "
            f"'{args.target_loop}'. The model has no loop identity in its input, so it will happily "
            f"emit something — it just won't be a '{args.target_loop}' model's answer."
        )

    # WARNING: the gap sits at the wrong offset for the expected loop. Needs no declaration, so it
    # is the only loop check that still fires when --target_loop is omitted.
    loop = trained_for or args.target_loop
    if loop and loop[-1].isdigit():
        expected_n = int(loop[-1])
        odd = [(c.get("name", f"case {i}"), len(c.get("pt1", "") or ""), n)
               for i, c in enumerate(cases)
               for n in [infer_loop_number(len(c.get("pt1", "") or ""))]
               if n is not None and n != expected_n]
        if odd:
            shown = "; ".join(f"'{name}' pt1={ln} (looks like CDR{n})" for name, ln, n in odd[:3])
            more = f" (+{len(odd) - 3} more)" if len(odd) > 3 else ""
            warnings.append(
                f"SHAPE: expecting CDR '{loop}', but {len(odd)}/{len(cases)} case(s) have a gap "
                f"positioned like a different CDR — {shown}{more}. Either the wrong loop is being "
                f"designed, or these frameworks are truncated. Only the CDR NUMBER is inferable from "
                f"pt1's length; heavy vs light is not, so an h/l mix-up cannot be caught this way."
            )

    # PROBLEM: missing or unexpected partner chain / antigen context
    for flag, key, name in [("include_partner_chain", "partner_chain", "partner chain"),
                            ("include_antigen", "antigen", "antigen")]:
        # Only meaningful against a real run_config. A CLI-built config sets include_* from what this
        # run supplies, so checking one against the other would be comparing args with themselves —
        # and would misfire outright on a test file whose cases carry their own context keys.
        if configs.get("source") != "run_config":
            break
        supplied_cli = bool(getattr(args, key, None))
        with_ctx = [c.get("name", "case") for c in cases if (c.get(key) or supplied_cli)]

        if configs.get(flag) and len(with_ctx) < len(cases):
            problems.append(
                f"CONTEXT MISSING: checkpoint was trained WITH {name} context, but "
                f"{len(cases) - len(with_ctx)}/{len(cases)} case(s) supply none — it will run off-"
                f"distribution and quietly underperform. Pass --{key}, or a '{key}' field per case."
            )
        elif not configs.get(flag) and with_ctx:
            problems.append(
                f"CONTEXT UNEXPECTED: checkpoint was trained WITHOUT {name} context, but "
                f"{len(with_ctx)}/{len(cases)} case(s) supply it — prepending it shifts every "
                f"learned position and is equally destructive. Drop --{key} and the '{key}' case field."
            )
    return problems, warnings


def _print_case(name, pt1, pt2, target_length, native, args, samples, metrics):
    """Human-readable console report for one case."""
    print("\n" + "=" * 72)
    print(f"🧪 {name}")
    print(f"   context {len(pt1)}aa │ gap {target_length} │ {len(pt2)}aa   "
          f"decoding={args.decoding} T={args.temperature} top_k={args.top_k} "
          f"top_p={args.top_p} n={len(samples)}")
    if native:
        print(f"   🍀 native: {native}")
    print("-" * 72)

    shown = samples[: args.show_samples]
    for i, s in enumerate(shown, 1):
        extra = ""
        if "recovery" in s and s["recovery"] is not None:
            extra += f"  rec={s['recovery'] * 100:5.1f}%"
        if "perplexity" in s:
            extra += f"  ppl={s['perplexity']:6.2f}"
        if "pseudo_perplexity" in s:
            extra += f"  ppl={s['pseudo_perplexity']:6.2f}"
        print(f"   {i:2d}) {s['seq']}{extra}")
    if len(samples) > len(shown):
        print(f"   … (+{len(samples) - len(shown)} more)")
    print("-" * 72)

    line = (f"   diversity: hamming={metrics['diversity_hamming']:.3f}  "
            f"unique={metrics['n_unique']}/{metrics['n_samples']}")
    if metrics.get("per_position_entropy_bits") is not None:
        line += f"  entropy={metrics['per_position_entropy_bits']:.2f}b"
    print(line)
    if "aar_mean" in metrics:
        print(f"   recovery : mean={metrics['aar_mean'] * 100:.1f}%  "
              f"max={metrics['aar_max'] * 100:.1f}%  min={metrics['aar_min'] * 100:.1f}%  "
              f"novelty={metrics['novelty_mean']:.3f}")
    if "pseudo_perplexity_mean" in metrics:
        print(f"   pseudo-ppl: mean={metrics['pseudo_perplexity_mean']:.2f}  "
              f"min={metrics['pseudo_perplexity_min']:.2f}  max={metrics['pseudo_perplexity_max']:.2f}")
    print("=" * 72)


def run_generation_pipeline(method, args, configs, *, prepare_generation,
                            generation_state, generate_case):
    """Validate, load and generate supplied cases; return their scored records.

    Compatibility and method-option checks precede checkpoint loading. A single
    seeded generator and optional method state are shared across cases. Output
    retains the model/context compatibility record and original case order.
    """
    method = resolve(method)
    if getattr(args, 'max_len', None) is None:
        args.max_len = configs.get('max_len', 256)
    test_cases = get_test_case(args)

    problems, warnings = check_compatibility(configs, test_cases, args)
    for w in warnings:
        logger.warning(f"⚠️  {w}")
    for p in problems:
        logger.warning(f"⛔ {p}")
    if problems and not args.force:
        raise SystemExit(
            f"\n❌ Refusing to run: {len(problems)} compatibility problem(s) above.\n"
            f"   Fix them, or pass --force to run anyway (the mismatch is then recorded in the "
            f"output JSON so the result stays self-describing)."
        )
    if problems:
        logger.warning(f"🔓 --force given: continuing despite {len(problems)} problem(s). "
                       f"They are recorded under 'model.compatibility' in the output.")

    # Validate method-specific options before loading any checkpoint weights.
    try:
        policies = prepare_generation(args, test_cases, configs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Load model for inference. model_type comes from configs, not args: --model_type is optional and
    # falls back to whatever the checkpoint says it is.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = configs["model_type"]
    print(f"🧬 Loading '{model_type}' weights onto {device}…")
    backbone = method.module("backbones").resolve(model_type)
    tokenizer = backbone.build_tokenizer(model_type, checkpoint_path=args.model_path)
    model = backbone.load_pretrained_model(args.model_path)
    model.to(device)
    model.eval()

    # Set seed
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    state = generation_state(args, policies, device)
    results = [generate_case(model, tokenizer, device, args, generator, case, state, _print_case)
               for case in test_cases]

    # Persist samples + metrics for downstream analysis (mirrors train.py's per-run CSV habit).
    if args.output != "none":
        out_path = args.output or os.path.join(args.model_path, "generation_results.json")
        # Wrap the per-case records with what produced them. Without this a generation_results.json
        # from a deliberate off-target run is indistinguishable from an on-target one later.
        payload = {
            "model": {
                "model_path": args.model_path,
                "model_type": model_type,
                "expects": configs,
                "designing": args.target_loop,
                "context_supplied": {
                    "partner_chain": bool(args.partner_chain) or any(c.get("partner_chain") for c in test_cases),
                    "antigen": bool(args.antigen) or any(c.get("antigen") for c in test_cases),
                },
                "compatibility": {"problems": problems, "warnings": warnings,
                                "forced": bool(args.force)},
            },
            "cases": results,
        }
        try:
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\n💾 Saved {len(results)} case(s) of samples + metrics to: {out_path}")
        except OSError as e:
            print(f"\n⚠️  Could not write results to {out_path}: {e}")
    return results
