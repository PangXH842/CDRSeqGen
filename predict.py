"""Inference/showcase tool - Design CDR loops for antibody frameworks supplied.

Give predict.py a framework with its CDR loop cut out — pt1 (sequence before the loop), pt2 (after),
and a gap length — and it fills the gap: samples a library of candidate loops and prints them, each
with recovery-vs-native (when you provide the native) and pseudo-perplexity, plus quick per-framework
diversity/novelty stats. The frameworks come from YOU: one case on the CLI (--pt1/--pt2/
--target_length) or a JSON of curated cases (--test_file).

Decoding (--decoding): independent (per-position, one forward pass) or iterative (Mask-Predict, each
residue conditioned on those already committed — coherent, dominates on CDR-H3). --temperature 0
makes either deterministic (greedy). Warp with --temperature / --top_k / --top_p. Everything runs
through the backbone registry, so the same path drives the scratch transformer, ESM-2, or any
protein-BERT with no model-family branching.

Usage examples (results/latest is the newest *batch* — name the config you want under it):
    # design 20 coherent H3 loops for one framework
    python predict.py --model_path ./results/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... --target_length 17

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
# logger defaults to WARNING with no handler, so INFO goes nowhere. train.py and gen_eval.py get
# this from train.py's own basicConfig; predict.py imports neither, so it needs its own.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from cdrseqgen.models.registry import resolve
from cdrseqgen.eval.sampling import (
    sample_independent,
    sample_iterative,
    pseudo_log_likelihood,
)
from cdrseqgen.eval.metrics import amino_acid_recovery, calculate


def model_trained_for(model_path):
    """The CDR loop this checkpoint was trained to design, from run_meta.json (train.py writes it).

    predict.py takes pt1/pt2 for *any* loop, so there is no config to cross-check against — but
    surfacing what the model was trained for still flags the "designing H3 with an L2 model" mistake.
    Returns None for hub models or older runs that have no run_meta.json.
    """
    path = os.path.join(model_path, "run_meta.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f).get("target_region")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def build_masked_input(tokenizer, pt1, pt2, target_length):
    """Assemble the masked input exactly as AntibodyInfillingDataset does (minus the label track).

    Returns (input_ids, mask_positions): the token-id list fed to the model, and the indices of the
    masked target positions within it. The [cls] … [end] wrapping and the leading offset both come
    from the TokenizerAdapter, so this is identical across backbones.
    """
    pt1_ids = tokenizer.encode(pt1)
    pt2_ids = tokenizer.encode(pt2)
    mask_ids = [tokenizer.mask_id] * target_length

    if tokenizer.uses_special_tokens:
        input_ids = [tokenizer.cls_id] + pt1_ids + mask_ids + pt2_ids + [tokenizer.end_id]
        start = 1 + len(pt1_ids)  # 1 for the leading cls token
    else:
        input_ids = pt1_ids + mask_ids + pt2_ids
        start = len(pt1_ids)

    mask_positions = list(range(start, start + target_length))
    return input_ids, mask_positions


def decode_case(model, tokenizer, device, args, generator, case):
    """Generate + score one test case. Returns a JSON-serialisable record."""
    name = case.get("name", "Unnamed Test")
    pt1, pt2 = case["pt1"], case["pt2"]
    native = case.get("target_seq")
    native = native.upper().strip() if native else None
    target_length = len(native) if native else case.get("target_length", 10)

    input_ids, mask_positions = build_masked_input(tokenizer, pt1, pt2, target_length)
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

    shown = samples[: args.show]
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

    trained_for = model_trained_for(args.model_path)
    if trained_for:
        print(f"🧬 This model was trained to design CDR '{trained_for}' loops.")

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

    # Sampling controls
    parser.add_argument("--decoding", choices=["independent", "iterative"], default="iterative",
                        help="independent (per-position, 1 pass) or iterative (Mask-Predict, coherent). "
                             "--temperature 0 makes either deterministic (greedy).")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Candidate loops to sample per framework (collapses to 1 when --temperature 0)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (>1 flatter, <1 sharper)")
    parser.add_argument("--top_k", type=int, default=0, help="Keep only the top-k logits (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling cumulative-prob cutoff (1.0 = disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling")
    parser.add_argument("--skip_perplexity", action="store_true",
                        help="Skip pseudo-perplexity scoring (the compute-heavy per-candidate part)")
    parser.add_argument("--show", type=int, default=5, help="How many sampled sequences to print per case")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write the results JSON ('none' to skip; default: <model_path>/generation_results.json)")

    main(parser.parse_args())
