"""Load a checkpoint and generate CDR candidates for supplied framework cases.

Functions:
  * run_generation_pipeline -- validate -> load -> generate cases -> save records
  * resolve_config          -- resolve checkpoint/CLI settings and their source
  * load_test_cases         -- read supplied JSON cases or build one from CLI args
  * infer_loop_number       -- estimate the CDR number from a flank-length heuristic
  * check_compatibility     -- report target/context mismatches before model loading
  * _print_case             -- display a method's samples and case measurements

Generation workflow:
  1. generate.py parses inputs -> <method>.gen_cli.main() configures logging.
  2. The CLI connects method functions to run_generation_pipeline():
       prepare_generation -> load model/tokenizer -> generation_state
       -> generate_case for each case -> optional JSON -> return case records.
  3. Each method builds its prompts, samples loops and calculates case measurements.

Notes:
  * Cases are supplied inputs, not a selected validation/test cohort. Native truth
    is optional; methods report diversity and any available reference/self scores.
  * CLI logging, shared loading/output and method decoding have separate owners.
    This workflow neither trains weights nor loads an independent judge.
  * One model, sampling generator and optional method state are reused across cases.
"""
import json
import logging
import os

import torch

from cdrseqgen.artifacts import load_run_config
from cdrseqgen.methods.registry import resolve

logger = logging.getLogger(__name__)


def resolve_config(args, *, run_config=None):
    """Return resolved settings tagged with their checkpoint or CLI source.

    Sources:
      run_config=None -> read args.model_dir/run_config.json
      run_config={}   -> no saved config; use CLI assumptions
      nonempty dict  -> copy saved settings, applying the model_type override

    Missing saved configuration permits CLI fallback; malformed configuration
    propagates its error. source='run_config' enables context compatibility checks;
    source='cli' describes assumptions, not independently verified training inputs.
    Call after CLI logging setup so source and mismatch messages are visible.
    """
    if run_config is None:
        try:
            config, _ = load_run_config(args.model_dir)
        except FileNotFoundError:
            config = {}
    else:
        config = dict(run_config)

    if config:
        config["source"] = "run_config"
        # An explicit model_type override also selects the tokenizer.
        trained_type = config.get("model_type")
        if args.model_type and trained_type and args.model_type != trained_type:
            logger.warning(f"⚠️  Model type mismatch: checkpoint was trained as '{trained_type}', but "
                           f"--model_type says '{args.model_type}' — the tokenizer comes from that "
                           f"flag, so every residue would be encoded against the wrong vocabulary.")
        config["model_type"] = args.model_type or trained_type or "scratch"
    else:
        logger.info("ℹ️  No run_config.json - building the config from CLI args instead.")
        config = {"target_region": args.target_loop,
                  "model_type": args.model_type or "scratch",
                  "include_partner_chain": bool(args.partner_chain),
                  "include_antigen": bool(args.antigen),
                  "data_path": None,
                  "max_len": args.max_len or 256,
                  "source": "cli"}

    # Report checkpoint expectations or the assumptions made from CLI inputs.
    ctx = [n for f, n in [("include_partner_chain", "partner chain"),
                          ("include_antigen", "antigen")] if config.get(f)]
    lead = "🧬 Checkpoint expects" if config["source"] == "run_config" else "🧬 Assuming from CLI args"
    print(f"{lead}: CDR '{config.get('target_region') or 'unknown'}'"
          + (f", with {' + '.join(ctx)} context" if ctx else ", framework context only"))
    if args.target_loop:
        print(f"🎯 This run is designing: CDR '{args.target_loop}'.")

    return config


def load_test_cases(args):
    """Return supplied JSON cases or one framework case assembled from CLI args.

    --test_file is a generation-case input, not a dataset split selector. JSON
    contents pass through unchanged; the CLI case carries flanks, optional native
    truth and length settings for the method's preparation function.
    """
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


# Historical flank-length bands used only for compatibility warnings.
LOOP_PT1_BANDS = [(1, 0, 30), (2, 31, 67), (3, 68, 10**6)]


def infer_loop_number(pt1_len):
    """Estimate the CDR number from residues before the gap; return None if unknown.

    Bands:
      pt1 length: <8   8..30   31..67   68..1,000,000
      returned:   None 1       2        3

    The cutoffs come from historical OAS/SAbDab2 frameworks. This warning heuristic
    does not validate numbering, distinguish heavy/light chains or inspect model
    conditioning. Truncated frameworks can fall into another loop's band.
    """
    if pt1_len < 8:      # below every framework observed — a deliberately truncated stub
        return None
    for number, lo, hi in LOOP_PT1_BANDS:
        if lo <= pt1_len <= hi:
            return number
    return None


def check_compatibility(config, cases, args):
    """Return (problems, warnings) for supplied target/context mismatches.

    Checks:
      target declaration or flank-length mismatch -> warning
      missing/unexpected partner or antigen      -> problem

    Context checks require source='run_config'; CLI assumptions cannot establish
    what training used. Context may come from CLI defaults or individual cases.
    This function collects messages; run_generation_pipeline decides whether
    problems stop the command or are recorded with an explicit --force override.
    """
    problems, warnings = [], []
    trained_for = (config or {}).get("target_region")

    # Warn when the requested target differs from the recorded training target.
    if args.target_loop and trained_for and args.target_loop.lower() != trained_for.lower():
        warnings.append(
            f"LOOP: checkpoint was trained to design CDR '{trained_for}', but --target_loop says "
            f"'{args.target_loop}'. The model has no loop identity in its input, so it will happily "
            f"emit something — it just won't be a '{args.target_loop}' model's answer."
        )

    # The flank-length heuristic also runs when --target_loop is omitted.
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

    # Only saved training settings provide an independent context contract.
    if config.get("source") != "run_config":
        return problems, warnings
    for flag, key, name in [("include_partner_chain", "partner_chain", "partner chain"),
                            ("include_antigen", "antigen", "antigen")]:
        supplied_cli = bool(getattr(args, key, None))
        with_ctx = [c.get("name", "case") for c in cases if (c.get(key) or supplied_cli)]

        if config.get(flag) and len(with_ctx) < len(cases):
            problems.append(
                f"CONTEXT MISSING: checkpoint was trained WITH {name} context, but "
                f"{len(cases) - len(with_ctx)}/{len(cases)} case(s) supply none — it will run off-"
                f"distribution and quietly underperform. Pass --{key}, or a '{key}' field per case."
            )
        elif not config.get(flag) and with_ctx:
            problems.append(
                f"CONTEXT UNEXPECTED: checkpoint was trained WITHOUT {name} context, but "
                f"{len(with_ctx)}/{len(cases)} case(s) supply it — prepending it shifts every "
                f"learned position and is equally destructive. Drop --{key} and the '{key}' case field."
            )
    return problems, warnings


def _print_case(name, pt1, pt2, target_length, native, args, samples, metrics):
    """Display supplied samples and metrics for one case without recalculating them.

    Methods call this callback after scoring. show_samples limits displayed
    candidates; the returned/saved case still contains all retained samples.
    """
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


def run_generation_pipeline(method, args, config, *, prepare_generation,
                            generation_state, generate_case):
    """Run the shared supplied-case workflow; return case records in input order.

    General pipeline:
      1. Resolve config/cases and check compatibility; prepare_generation()
           validates method options and returns policies or None.
      2. Load the checkpoint/tokenizer, seed sampling, then call
           generation_state(args, policies, device) for optional shared state.
      3. generate_case(..., case, state, _print_case) produces each scored record.
      4. Optionally save model/context metadata and cases; return the case list.

    Compatibility and method-option checks precede weight loading. --force only
    overrides collected compatibility problems. output='none' skips saving;
    otherwise the default is model_dir/generation_results.json. Output OSErrors
    are printed and records still return; loading/generation failures propagate.
    """
    # ------ 1.1 Resolve the method, configuration and supplied cases. ------
    method = resolve(method)
    print(f"🧬 Checkpoint: {args.model_dir}")
    if 'source' not in config:
        config = resolve_config(args, run_config=config)
    if getattr(args, 'max_len', None) is None:
        args.max_len = config.get('max_len', 256)
    test_cases = load_test_cases(args)

    # ------ 1.2 Check the saved target/context contract before loading weights. ------
    problems, warnings = check_compatibility(config, test_cases, args)
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

    # ------ 1.3 Validate method options and resolve per-case policies. ------
    try:
        policies = prepare_generation(args, test_cases, config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # ------ 2.1 Load the tokenizer and checkpoint for inference. ------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = config["model_type"]
    print(f"🧬 Loading '{model_type}' weights onto {device}…")
    backbone = method.module("backbones").resolve(model_type)
    tokenizer = backbone.build_tokenizer(model_type, checkpoint_path=args.model_dir)
    model = backbone.load_pretrained_model(args.model_dir)
    model.to(device)
    model.eval()

    # ------ 2.2 Seed sampling and load optional method state once. ------
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    state = generation_state(args, policies, device)
    # ------ 3. Generate, measure and display each case in input order. ------
    results = [generate_case(model, tokenizer, device, args, generator, case, state, _print_case)
               for case in test_cases]

    # ------ 4. Save the outer metadata/case payload when output is enabled. ------
    if args.output != "none":
        out_path = args.output or os.path.join(args.model_dir, "generation_results.json")
        payload = {
            "model": {
                "model_path": args.model_dir,     # key unchanged: saved outputs and their readers use it
                "model_type": model_type,
                "expects": config,
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
