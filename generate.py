"""Candidate-generation showcase for antibody frameworks and optional context.

Method calls:
  generate.py -> methods.registry -> <method>/gen_cli.py
  prepare_generation -> generation_state -> generate_case (per case)

Usage:
  Select a trained model, provide the protein framework before (pt1) and after (pt2) the selected loop,
  and it will generate a sample library of candidate loops, with corresponding eval metrics.

Args:
  * --model_path    -- path to trained model
  * --model_type    -- specify the type of model; defaults to whatever run_config.json says the
                        checkpoint is, so it only needs passing for an external one
  * --test_file     -- read from a json file including test cases
                        (args to include: pt1, pt2, target_truth*, min_length*, max_length*,
                         target_length**, partner_chain***, antigen***)
                        (*truth or equal bounds; **deprecated exact-length alias; ***optional)
  * --pt1, --pt2    -- framework regions before and after the target CDR
  * --target_loop   -- target CDR (h1/h2/h3/l1/l2/l3) this run is designing; declarative only, and
                        warns on mismatch with the loop in run_config.json
                        (a gap sitting at the wrong offset is warned for either way, from pt1's
                         length, since that needs no declaration)
  * --target_truth  -- optional ground truth
  * --min_length, --max_length -- inclusive length bounds; equal values fix the length, unequal or
                                  one-sided bounds constrain the learned length distribution, and
                                  omitting both lets the checkpoint infer length
  * --target_length -- deprecated alias for an exact fixed CDR length
  * --partner_chain -- paired chain context, optional (refused depending on model requirements)
  * --antigen       -- antigen sequence context, optional (refused depending on model requirements)
  * --force         -- run despite the context mismatches above (which block, unlike the loop
                        warnings); recorded in the output json
  * --max_len       -- length cap for the assembled input; match the training config
                        (SAbDab2 antigen runs use 512)
  * --decoding      -- decoding strategy choice: "independent" for individual position sampling;
                        "iterative" for iterative most-to-least confident sampling, uses more time
                        depending on loop length
  * --num_samples   -- number of CDR loop samples to generate (1 if greedy)
  * --show_samples  -- number of CDR loop samples to display
  * --temperature   -- likeliness to sample unlikely AAs, 0 for greedy
  * --top_k         -- limit to top k most possible AAs per position, optional
  * --top_p         -- eliminate tail AAs, optional
  * --seed
  * --skip_perplexity -- skip pseudo-perplexity scoring (the compute-heavy per-candidate part)
  * --output        -- path to store result json

Usage examples (results/runs/latest is the newest *batch* — name the config you want under it):
    # design 20 coherent H3 loops for one framework
    python generate.py --model_path ./results/runs/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... \
        --min_length 17 --max_length 17

    # a curated batch with nucleus sampling
    python generate.py --model_path ./results/runs/latest/h3_OAS_scratch --test_file showcase/h3.json \
        --decoding iterative --num_samples 50 --top_p 0.9
"""
import argparse
import json
import logging
import os
import torch

from cdrseqgen.artifacts import load_run_config
from cdrseqgen.methods.common.logging import configure_logging
configure_logging()

from cdrseqgen.methods.registry import decoding_choices, checkpoint_method

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


def main(args):
    print(f"🧬 Checkpoint: {args.model_path}")
    configs = get_configs(args)
    method = checkpoint_method(configs)
    gen_cli = method.module('gen_cli')
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
        policies = gen_cli.prepare_generation(args, test_cases, configs)
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

    state = gen_cli.generation_state(args, policies, device)
    results = [gen_cli.generate_case(model, tokenizer, device, args, generator, case, state, _print_case)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Design CDR loops for frameworks you provide (qualitative tool; use evaluate.py for benchmarks)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your trained model folder")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Backbone: 'scratch', an ESM-2 id, a protein-BERT id, or any HF masked-LM hub path")

    parser.add_argument("--test_file", type=str, default=None,
                        help="JSON file of test cases (omit to use the single-case CLI args below)")

    # Fallback CLI arguments for a quick single test
    parser.add_argument("--pt1", type=str, default="", help="Sequence BEFORE the missing loop")
    parser.add_argument("--pt2", type=str, default="", help="Sequence AFTER the missing loop")
    parser.add_argument("--min_length", type=int, default=None,
                        help="Inclusive minimum CDR length (alone or with --max_length)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Inclusive maximum CDR length (alone or with --min_length)")
    parser.add_argument("--target_length", type=int, default=None,
                        help="Deprecated exact-length alias; prefer equal --min_length/--max_length")
    parser.add_argument("--use_native_length", action="store_true",
                        help="Explicitly use target_truth length for the fixed track; by default "
                             "target_truth is scoring-only and omitted bounds are inferred")
    parser.add_argument("--target_truth", type=str, default=None, help="Optional ground-truth loop")
    parser.add_argument("--partner_chain", type=str, default=None,
                        help="Paired partner chain (light for an h* target, heavy for l*). "
                             "Required by any checkpoint trained with include_partner_chain.")
    parser.add_argument("--antigen", type=str, default=None,
                        help="Antigen sequence; required by checkpoints trained with include_antigen. "
                             "Injection side follows the backbone's position_type automatically.")
    parser.add_argument("--target_loop", type=str, choices=['h1', 'h2', 'h3', 'l1', 'l2', 'l3'], default=None,
                        help="Which CDR this run is designing (h1/h2/h3/l1/l2/l3). Declarative: it "
                             "cannot change the computation, and a mismatch with the checkpoint's "
                             "trained loop warns rather than refuses (off-target probing is a valid "
                             "experiment). The mismatch is recorded in the output JSON.")
    parser.add_argument("--force", action="store_true",
                        help="Run despite blocking CONTEXT problems (the checkpoint expects partner "
                             "chain / antigen that this run does not supply, or vice versa). Loop "
                             "and shape mismatches only warn and never need this.")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Length cap for the assembled input; match the training config (SAbDab2 antigen runs use 512).")

    # Sampling controls
    parser.add_argument("--decoding", choices=decoding_choices(), default=None,
                        help="independent (per-position, 1 pass) or iterative (Mask-Predict, coherent). "
                             "--temperature 0 makes either deterministic (greedy).")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="AR decode cap, including EOS.")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Candidate loops to sample per framework (collapses to 1 when --temperature 0)")
    parser.add_argument("--show_samples", type=int, default=5, help="How many sampled sequences to print per case")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (>1 flatter, <1 sharper)")
    parser.add_argument("--length_temperature", type=float, default=1.0,
                        help="Temperature for sampled length classes; 0 selects the most likely "
                             "allowed length")
    parser.add_argument("--top_k", type=int, default=0, help="Keep only the top-k logits (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling cumulative-prob cutoff (1.0 = disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling")
    parser.add_argument("--skip_perplexity", action="store_true",
                        help="Skip pseudo-perplexity scoring (the compute-heavy per-candidate part)")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write the results JSON ('none' to skip; default: <model_path>/generation_results.json)")

    main(parser.parse_args())
