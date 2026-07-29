"""Inference/showcase tool - Design CDR loops with given antibody frameworks and context.

Usage:
  Select a trained model, provide the protein framework before (pt1) and after (pt2) the selected loop,
  and it will generate a sample library of candidate loops, with corresponding eval metrics.

  There is NO --target_loop argument, because nothing in the tensor names a loop or a chain: the
  dataset uses target_region to decide where to CUT, but the model only ever sees
  `[cls] pt1 [mask x L] pt2 [end]` and infers the loop from the framework. Which loop you design is
  fixed by WHICH checkpoint you load (h3_OAS_scratch vs l2_OAS_scratch are different models), so
  --model_path is effectively that argument; run_meta.json is read to report it back to you.

Args:
  * --model_path    -- path to trained model
  * --model_type    -- specify the type of model
  * --test_file     -- read from a json file including test cases
                        (args to include: pt1, pt2, target_seq*, target_length*, partner_chain**,
                         antigen**) (*one of either) (**if the checkpoint was trained with it)
  * --pt1, --pt2    -- framework regions before and after the target CDR
  * --target_seq    -- optional ground truth
  * --target_length -- specified CDR length (?)
  * --partner_chain -- paired chain (light for h*, heavy for l*); REQUIRED by any checkpoint trained
                        with include_partner_chain — omitting it does not error, it silently degrades
                        (measured 0.986 -> 0.260 recovery on an _LC H1 model)
  * --antigen       -- antigen sequence; required by checkpoints trained with include_antigen.
                        Which side it is injected on follows the backbone's position_type
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
  * --output        -- path to store result json

Usage examples (results/latest is the newest *batch* — name the config you want under it):
    # design 20 coherent H3 loops for one framework
    python predict.py --model_path ./results/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... \
        --target_length 17

    # a curated batch with nucleus sampling
    python predict.py --model_path ./results/latest/h3_OAS_scratch --test_file test/test_h3.json \
        --decoding iterative --num_samples 50 --top_p 0.9
"""
import argparse
import json
import logging
import os
import sys
import torch

# Entrypoints configure logging; library modules just getLogger(__name__). Without this, the
# backbone-loading messages in registry.py (logger.info) would be dropped entirely here — the root
# logger defaults to WARNING with no handler, so INFO goes nowhere.
from cdrseqgen.train.workspace import configure_logging
configure_logging()

from cdrseqgen.data.dataset import assemble_masked_input, inject_context
from cdrseqgen.models.registry import resolve
from cdrseqgen.eval.sampling import (
    sample_independent,
    sample_iterative,
    pseudo_log_likelihood,
)
from cdrseqgen.eval.metrics import amino_acid_recovery, calculate

logger = logging.getLogger(__name__)


def read_run_meta(model_path):
    """What this checkpoint was trained on, from run_meta.json (train.py writes it).

    Returns the dict, or {} for hub models / older runs that have no run_meta.json. Used both to
    report the trained loop and to check the context the checkpoint expects — see warn_on_mismatch.
    """
    path = os.path.join(model_path, "run_meta.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def warn_on_mismatch(meta, cases):
    """Warn when the checkpoint expects context this invocation isn't supplying.

    A model trained with partner-chain or antigen context does NOT fail on a bare framework — it
    silently degrades, because the input is merely out-of-distribution rather than malformed.
    Measured on h1_OAS_LC_scratch over 60 held-out frameworks: 0.986 recovery when the light chain
    is supplied as trained, 0.260 without it. That is the whole reason this check exists.
    """
    if not meta:
        logger.info("ℹ️  No run_meta.json — can't verify this checkpoint's expected context.")
        return
    for flag, key, name in [("include_partner_chain", "partner_chain", "partner chain"),
                            ("include_antigen", "antigen", "antigen")]:
        if not meta.get(flag):
            continue
        missing = [c.get("name", "case") for c in cases if not c.get(key)]
        if missing:
            logger.warning(
                f"⚠️  This checkpoint was TRAINED WITH {name} context, but {len(missing)}/{len(cases)} "
                f"case(s) supply none — the model will be run off-distribution and will quietly "
                f"underperform (an _LC H1 model measures 0.986 -> 0.260 recovery this way). "
                f"Pass --{key} (or a '{key}' field per case in --test_file)."
            )


def build_masked_input(tokenizer, pt1, pt2, target_length, partner_chain=None, antigen=None,
                       max_len=256):
    """Assemble the masked input for one framework, minus the label track.

    Returns (input_ids, mask_positions). Both the context injection and the layout come from
    dataset.py — `inject_context` then `assemble_masked_input`, the SAME functions the training
    examples are built with — so inference cannot drift from training. Which side the antigen lands
    on, and which side gets truncated, follow the backbone's position_type automatically.
    """
    pt1_ids, pt2_ids = inject_context(
        tokenizer, tokenizer.encode(pt1), tokenizer.encode(pt2), target_length,
        partner_seq=partner_chain, antigen_seq=antigen, max_len=max_len,
    )
    input_ids, start = assemble_masked_input(tokenizer, pt1_ids, pt2_ids, target_length)
    return input_ids, list(range(start, start + target_length))


def decode_case(model, tokenizer, device, args, generator, case):
    """Generate + score one test case. Returns a JSON-serialisable record."""
    name = case.get("name", "Unnamed Test")
    pt1, pt2 = case["pt1"], case["pt2"]
    native = case.get("target_seq")
    native = native.upper().strip() if native else None
    target_length = len(native) if native else case.get("target_length", 10)

    input_ids, mask_positions = build_masked_input(
        tokenizer, pt1, pt2, target_length,
        partner_chain=case.get("partner_chain") or args.partner_chain,
        antigen=case.get("antigen") or args.antigen,
        max_len=args.max_len,
    )
    banned = tokenizer.special_ids

    # --- decode ---  (greedy == temperature 0; see sampling.py)
    if args.decoding == "independent":
        sampled_ids = sample_independent(
            model, input_ids, mask_positions, banned, device=device,
            num_samples=args.num_samples, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, generator=generator,
        )
    else:  # iterative
        sampled_ids = sample_iterative(
            model, input_ids, mask_positions, banned, tokenizer.mask_id, device=device,
            num_samples=args.num_samples, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, generator=generator,
        )

    # --- decode ids -> strings + per-sample scores ---
    samples = []
    for ids in sampled_ids:
        seq = tokenizer.decode(ids).upper().strip()
        entry = {"seq": seq}
        if native:
            entry["recovery"] = amino_acid_recovery(seq, native)
        if not args.skip_perplexity:
            _, ppl = pseudo_log_likelihood(
                model, input_ids, mask_positions, ids, banned, tokenizer.mask_id, device
            )
            entry["pseudo_perplexity"] = ppl
        samples.append(entry)

    # --- aggregate generative metrics over the sample set ---
    seqs = [s["seq"] for s in samples]
    metrics = calculate(seqs, native)
    if not args.skip_perplexity and samples:
        ppls = [s["pseudo_perplexity"] for s in samples]
        metrics["pseudo_perplexity_mean"] = sum(ppls) / len(ppls)
        metrics["pseudo_perplexity_min"] = min(ppls)
        metrics["pseudo_perplexity_max"] = max(ppls)

    _print_case(name, pt1, pt2, target_length, native, args, samples, metrics)

    return {
        "name": name,
        "pt1": pt1,
        "pt2": pt2,
        "target_length": target_length,
        "native": native,
        "decoding": {
            "mode": args.decoding,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "num_samples": len(samples),
            "seed": args.seed,
        },
        "samples": samples,
        "metrics": metrics,
    }


def _print_case(name, pt1, pt2, target_length, native, args, samples, metrics):
    """Human-readable console report for one case (a compact superset of the old alignment print)."""
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
    if "recovery_mean" in metrics:
        print(f"   recovery : mean={metrics['recovery_mean'] * 100:.1f}%  "
              f"max={metrics['recovery_max'] * 100:.1f}%  min={metrics['recovery_min'] * 100:.1f}%  "
              f"novelty={metrics['novelty_mean']:.3f}")
    if "pseudo_perplexity_mean" in metrics:
        print(f"   pseudo-ppl: mean={metrics['pseudo_perplexity_mean']:.2f}  "
              f"min={metrics['pseudo_perplexity_min']:.2f}  max={metrics['pseudo_perplexity_max']:.2f}")
    print("=" * 72)


def main(args):
    print(f"🧬 Loading model from: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = resolve(args.model_type)
    tokenizer = backbone.build_tokenizer(args.model_type)
    model = backbone.load_pretrained_model(args.model_path)
    model.to(device)
    model.eval()

    # The loop being designed is fixed by WHICH checkpoint is loaded — nothing in the tensor names a
    # loop or a chain, so there is no runtime argument for it. Report what this one was trained for.
    meta = read_run_meta(args.model_path)
    if meta.get("target_region"):
        ctx = [n for f, n in [("include_partner_chain", "partner chain"),
                              ("include_antigen", "antigen")] if meta.get(f)]
        print(f"🧬 This model was trained to design CDR '{meta['target_region']}' loops"
              + (f", with {' + '.join(ctx)} context." if ctx else ", framework context only."))

    # Reproducible sampling: one seeded generator on the model's device drives every draw.
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # Gather test cases (JSON batch, else a single case from the CLI args).
    if args.test_file:
        with open(args.test_file, "r") as f:
            test_cases = json.load(f)
    else:
        test_cases = [{
            "name": "CLI Test Case",
            "pt1": args.pt1,
            "pt2": args.pt2,
            "target_seq": args.target_seq,
            "target_length": args.target_length,
        }]

    warn_on_mismatch(meta, test_cases)

    results = [decode_case(model, tokenizer, device, args, generator, case) for case in test_cases]

    # Persist samples + metrics for downstream analysis (mirrors train.py's per-run CSV habit).
    if args.output != "none":
        out_path = args.output or os.path.join(args.model_path, "generation_results.json")
        try:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Saved {len(results)} case(s) of samples + metrics to: {out_path}")
        except OSError as e:
            print(f"\n⚠️  Could not write results to {out_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Design CDR loops for frameworks you provide (qualitative tool; use gen_eval.py for benchmarks)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your trained model folder")
    parser.add_argument("--model_type", type=str, default="scratch",
                        help="Backbone: 'scratch', an ESM-2 id, a protein-BERT id, or any HF masked-LM hub path")

    parser.add_argument("--test_file", type=str, default=None,
                        help="JSON file of test cases (omit to use the single-case CLI args below)")

    # Fallback CLI arguments for a quick single test
    parser.add_argument("--pt1", type=str, default="", help="Sequence BEFORE the missing loop")
    parser.add_argument("--pt2", type=str, default="", help="Sequence AFTER the missing loop")
    parser.add_argument("--target_length", type=int, default=10, help="Gap length when no ground truth is given")
    parser.add_argument("--target_seq", type=str, default=None, help="Optional ground-truth loop")
    # Context the checkpoint may have been TRAINED with. A model trained with these and run without
    # them does not fail — it silently degrades (measured 0.986 -> 0.260 on an _LC H1 model), so
    # warn_on_mismatch() checks run_meta.json and says so. Per-case keys in --test_file win over these.
    parser.add_argument("--partner_chain", type=str, default=None,
                        help="Paired partner chain (light for an h* target, heavy for l*). "
                             "Required by any checkpoint trained with include_partner_chain.")
    parser.add_argument("--antigen", type=str, default=None,
                        help="Antigen sequence; required by checkpoints trained with include_antigen. "
                             "Injection side follows the backbone's position_type automatically.")
    parser.add_argument("--max_len", type=int, default=256,
                        help="Length cap for the assembled input; match the training config (SAbDab2 antigen runs use 512).")

    # Sampling controls
    parser.add_argument("--decoding", choices=["independent", "iterative"], default="iterative",
                        help="independent (per-position, 1 pass) or iterative (Mask-Predict, coherent). "
                             "--temperature 0 makes either deterministic (greedy).")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Candidate loops to sample per framework (collapses to 1 when --temperature 0)")
    parser.add_argument("--show_samples", type=int, default=5, help="How many sampled sequences to print per case")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (>1 flatter, <1 sharper)")
    parser.add_argument("--top_k", type=int, default=0, help="Keep only the top-k logits (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling cumulative-prob cutoff (1.0 = disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling")
    parser.add_argument("--skip_perplexity", action="store_true",
                        help="Skip pseudo-perplexity scoring (the compute-heavy per-candidate part)")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write the results JSON ('none' to skip; default: <model_path>/generation_results.json)")

    main(parser.parse_args())
