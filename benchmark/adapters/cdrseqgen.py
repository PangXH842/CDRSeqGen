#!/usr/bin/env python3
"""Generate CDRSeqGen outputs for a frozen benchmark manifest and checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Direct execution puts benchmark/adapters first on sys.path, where this file would shadow the
# `cdrseqgen` package. Replace that entry with the repository root before lazy package imports.
ADAPTER_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == ADAPTER_DIR:
    sys.path.pop(0)
    sys.path.insert(0, str(ADAPTER_DIR.parents[1]))

from cdrseqgen.artifacts import (
    LENGTH_CONFIG_FILENAME,
    LENGTH_HEAD_FILENAME,
    RUN_CONFIG_FILENAMES,
    load_run_config as shared_load_run_config,
)
from benchmark._common import (
    ADAPTER_SCHEMA_VERSION as SCHEMA_VERSION,
    AMINO_ACIDS,
    assert_resume_compatible,
    atomic_write_json,
    execution_source_identity,
    git_state,
    sha256_file,
    stable_example_seed,
    validate_manifest_record,
)


def load_run_config(checkpoint: Path) -> tuple[dict[str, Any], Path]:
    """Adapter-side alias so the benchmark reads checkpoints exactly as the entrypoints do."""
    return shared_load_run_config(checkpoint)


TOKENIZER_ASSETS = ('tokenizer_config.json', 'tokenizer.json', 'vocab.txt', 'vocab.json', 'merges.txt',
                    'special_tokens_map.json', 'added_tokens.json', 'tokenizer.model', 'spiece.model',
                    'sentencepiece.bpe.model', 'span_layout.json', 'generation_config.json')
TOKENIZER_VOCABULARIES = ('tokenizer.json', 'vocab.txt', 'vocab.json', 'tokenizer.model', 'spiece.model',
                        'sentencepiece.bpe.model')


def pretrained_input_files(checkpoint: Path) -> list[Path]:
    """Identify a self-contained saved tokenizer and generation settings without loading models.

    Requiring tokenizer_config.json prevents the shared loader from falling back
    to the original model source. Tokenizer assets and generation_config.json
    affect inference and must be frozen alongside checkpoint weights.
    """
    checkpoint = Path(checkpoint)
    vocabularies = [checkpoint / name for name in TOKENIZER_VOCABULARIES]
    vocabularies.extend(checkpoint.glob('*.tiktoken'))
    if not (checkpoint / 'tokenizer_config.json').is_file() or not any(p.is_file() for p in vocabularies):
        raise ValueError(f'{checkpoint}: pretrained benchmark arms require a saved tokenizer_config.json '
                         'and vocabulary beside their weights; original-source fallback is unsupported')
    paths = [checkpoint / name for name in TOKENIZER_ASSETS if (checkpoint / name).is_file()]
    paths.extend(path for path in sorted(checkpoint.glob('*.tiktoken')) if path not in paths)
    return paths


def validate_checkpoint_contract(run_config: dict[str, Any], manifest: dict[str, Any]) -> None:
    task = manifest.get("task") or {}
    expected_target = task.get("target_region")
    if run_config.get("target_region") != expected_target:
        raise ValueError(
            f"checkpoint targets {run_config.get('target_region')!r}, manifest targets "
            f"{expected_target!r}"
        )
    if task.get("track") not in {"fixed_native_length", "length_free"}:
        raise ValueError("manifest track must be fixed_native_length or length_free")
    from cdrseqgen.methods.registry import checkpoint_method
    method = checkpoint_method(run_config)
    if method.name != 'mlm' and task['track'] != 'length_free':
        raise ValueError(f'{method.name} chooses length by EOS; use the length_free track')
    for config_key, task_key in (
        ("include_partner_chain", "partner_chain_context"),
        ("include_antigen", "antigen_context"),
    ):
        if bool(run_config.get(config_key, False)) != bool(task.get(task_key, False)):
            raise ValueError(
                f"checkpoint {config_key}={run_config.get(config_key)!r} does not match "
                f"manifest {task_key}={task.get(task_key)!r}"
            )
    if task.get('partner_chain_context') or task.get('antigen_context'):
        raise ValueError('This benchmark adapter supplies heavy-chain flanks only; optional context is unsupported')
    processed_name = (manifest.get("source") or {}).get("processed_file")
    if processed_name and Path(run_config.get("data_path", "")).name != processed_name:
        raise ValueError(
            f"checkpoint data file {run_config.get('data_path')!r} does not match manifest "
            f"source {processed_name!r}"
        )
    declared_hash = (run_config.get('provenance', {}).get('dataset') or {}).get(
        'declared_processed_file_sha256')
    manifest_hash = (manifest.get('source') or {}).get('processed_file_sha256')
    if declared_hash and manifest_hash and declared_hash != manifest_hash:
        raise ValueError('checkpoint processed corpus hash differs from manifest')


def resolve_decoding(run_config, requested=None):
    """Resolve a method decoder before loading weights.

    MLM uses an explicit request or a compatible saved gen_decoding, falling back
    to independent. A recorded iterative choice requires mixed-mask training;
    explicit iterative requests remain allowed for full_span/legacy checkpoints.
    AR/S2S retain their registered decoder defaults.
    """
    from cdrseqgen.methods.registry import checkpoint_method
    method = checkpoint_method(run_config)
    if requested is None:
        if method.name == 'mlm':
            requested = 'independent'
            recorded = run_config.get('gen_decoding')
            policy = (run_config.get('masking_policy') or 'full_span').strip().lower()
            if policy == 'mixed' and recorded in {'independent', 'iterative'}:
                requested = recorded
        else:
            requested = method.decoders[0]
    if requested not in method.decoders:
        raise ValueError(f'{method.name} decoding must be one of {method.decoders}, got {requested!r}')
    return requested


class CDRSeqGenSampler:
    def __init__(self, checkpoint: Path, run_config: dict[str, Any], device: str,
                 track: str = "fixed_native_length"):
        import torch

        from cdrseqgen.methods.mlm.backbones import resolve
        from cdrseqgen.methods.mlm.length import (
            LengthPredictor,
            prepare_fixed_length_prompt,
            sample_length_conditioned_spans,
            fixed_policy,
            native_mask_length,
            resolve_length_policy,
        )

        self.torch = torch
        self.prepare_fixed_length_prompt = prepare_fixed_length_prompt
        self.sample_length_conditioned_spans = sample_length_conditioned_spans
        self.resolve_length_policy = resolve_length_policy
        self.fixed_policy = fixed_policy
        self.track = track
        self.native_mask_length = native_mask_length
        self.run_config = run_config
        self.max_len = int(run_config["max_len"])

        self.backbone = resolve(run_config["model_type"])
        self.tokenizer = self.backbone.build_tokenizer(run_config["model_type"])
        self.model = self.backbone.load_pretrained_model(str(checkpoint))
        selected_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.device = selected_device
        self.model.to(self.device)
        self.model.eval()
        self.length_predictor = (
            LengthPredictor.load(checkpoint, device=self.device)
            if track == "length_free" else None
        )

    def prepare(self, record: dict[str, Any]) -> None:
        # Clear per-example state before anything can raise: run_example reads length_metadata and
        # prompt_info even when this call fails, and the previous example's values would then be
        # recorded against this one.
        self.record = record
        self.length_metadata = None
        self.length_policy = None
        self.length_decision = None
        self.prompt_info = None
        self.input_ids = None
        self.mask_positions = None
        if self.track == "length_free":
            self.length_policy = self.resolve_length_policy()
            self.length_metadata = self.length_policy.as_dict()
            return
        # The native mask count is a length that is known rather than predicted, so it becomes an
        # ordinary fixed policy and takes the same generation path as the length-free track.
        self.length_decision = self.native_mask_length(record["native_length"])
        self.length_policy = self.fixed_policy(
            self.length_decision.length, source=self.length_decision.source
        )
        # Built once here purely to enforce the checkpoint's max_len before any GPU work; the
        # prompt used for sampling is rebuilt inside sample_length_conditioned_spans.
        prompt = self.prepare_fixed_length_prompt(
            self.tokenizer,
            record["pt1"],
            record["pt2"],
            self.length_decision.length,
            max_len=self.max_len,
        )
        self.input_ids = prompt.input_ids
        self.mask_positions = prompt.mask_positions
        self.prompt_info = prompt.metadata()
        if len(self.input_ids) > self.max_len:
            raise ValueError(
                f"{record['example_id']} assembled length {len(self.input_ids)} exceeds "
                f"checkpoint max_len {self.max_len}"
            )

    def draw_many(
        self,
        *,
        num_samples: int,
        decoding: str,
        temperature: float,
        top_k: int,
        top_p: float,
        length_temperature: float,
        seed: int,
    ) -> list[str]:
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(seed)

        generated, self.length_metadata = self.sample_length_conditioned_spans(
            self.model,
            self.tokenizer,
            self.device,
            generator,
            pt1=self.record["pt1"],
            pt2=self.record["pt2"],
            policy=self.length_policy,
            length_predictor=self.length_predictor,
            max_len=self.max_len,
            decoding=decoding,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            length_temperature=length_temperature,
        )
        self.prompt_info = self.length_metadata.get("prompts_by_length")
        return [
            self.tokenizer.decode(span.token_ids).upper().strip()
            for span in generated
        ]


class EOSSampler:
    """Use the method's existing prompt/sampler and retain incomplete/empty raw draws."""

    def __init__(self, checkpoint, run_config, device, max_new_tokens=None):
        import torch
        from cdrseqgen.methods.registry import checkpoint_method
        self.torch = torch
        self.method = checkpoint_method(run_config)
        self.run_config = run_config
        self.track = 'length_free'
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else run_config.get('gen_max_new_tokens', 97)
        if self.method.name == 'ar':
            from cdrseqgen.datasets.dataset_ar import build_causal_prompt, validate_causal_decode_steps
            self.build_prompt = build_causal_prompt
            validate_causal_decode_steps(self.max_new_tokens)
        elif self.method.name == 's2s':
            from cdrseqgen.datasets.dataset_s2s import build_encoder_input, validate_s2s_decode_steps
            self.build_prompt = build_encoder_input
            validate_s2s_decode_steps(self.max_new_tokens)
        else:
            raise ValueError(f'EOSSampler supports AR/S2S, got {self.method.name!r}')
        self.backbone = self.method.module('backbones').resolve(run_config['model_type'])
        self.tokenizer = self.backbone.build_tokenizer(run_config['model_type'], str(checkpoint))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.model = self.backbone.load_pretrained_model(str(checkpoint)).to(self.device)
        self.model.eval()
        self.sample_loops = self.method.module('sampling').sample_loops

    def prepare(self, record):
        self.prompt_info = self.length_metadata = None
        # The native loop and its length never enter prompt assembly or decoding.
        framework = dict(pt1=record['pt1'], pt2=record['pt2'], target_length=0,
                         max_len=int(self.run_config['max_len']), partner_chain=None, antigen=None)
        # Declared conditioning only, for a layout that reads it (IgLM's chain and species tokens): the
        # manifest's tokens and the checkpoint's region. Every other layout ignores these keys.
        framework.update(target_region=self.run_config.get('target_region'),
                         **{key: record[key] for key in ('chain_token', 'species_token') if record.get(key)})
        self.prompt_ids, meta = self.build_prompt(self.tokenizer, framework)
        self.prompt_info = {'input_tokens': len(self.prompt_ids), 'max_len': framework['max_len']}
        if 'conditioning' in meta:      # which chain/species tokens this record's prompt actually carried
            self.prompt_info['conditioning'] = meta['conditioning']
        self.length_metadata = {'mode': 'inferred', 'source': 'eos_token',
                                'max_new_tokens': self.max_new_tokens}

    def draw_many(self, *, num_samples, decoding, temperature, top_k, top_p, length_temperature, seed):
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        return self.sample_loops(self.model, self.tokenizer, self.prompt_ids,
                                 device=self.device, generator=generator, num_samples=num_samples,
                                 temperature=temperature, top_k=top_k, top_p=top_p,
                                 max_new_tokens=self.max_new_tokens)


def build_sampler(checkpoint, run_config, device, track, max_new_tokens=None):
    from cdrseqgen.methods.registry import checkpoint_method
    if checkpoint_method(run_config).name == 'mlm':
        if max_new_tokens is not None:
            raise ValueError('MLM uses its length policy; --max-new-tokens is for EOS methods')
        return CDRSeqGenSampler(checkpoint, run_config, device, track=track)
    if track != 'length_free':
        raise ValueError('EOS methods require the length_free track')
    return EOSSampler(checkpoint, run_config, device, max_new_tokens)


def run_example(
    record: dict[str, Any],
    sampler,
    *,
    num_samples: int,
    decoding: str,
    temperature: float,
    top_k: int,
    top_p: float,
    base_seed: int,
    length_temperature: float = 1.0,
) -> dict[str, Any]:
    validate_manifest_record(record)
    seed = stable_example_seed(base_seed, record["example_id"])
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    generation_failed = False
    try:
        # prepare() is inside the guard so an over-long prompt is recorded as one failed example
        # rather than aborting the whole benchmark run.
        sampler.prepare(record)
        samples = sampler.draw_many(
            num_samples=num_samples,
            decoding=decoding,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            length_temperature=length_temperature,
            seed=seed,
        )
    except Exception as exc:
        samples = []
        generation_failed = True
        failures.append({
            "code": "generation_exception",
            "detail": f"{type(exc).__name__}: {exc}",
        })

    valid_samples = []
    raw_generations = []
    for attempt, draw in enumerate(samples, start=1):
        loop = draw['loop'] if isinstance(draw, dict) else draw
        complete = draw.get('complete', True) if isinstance(draw, dict) else True
        decoder_valid = draw.get('valid', True) if isinstance(draw, dict) else True
        invalid = sorted(set(loop) - AMINO_ACIDS)
        valid_generation = bool(complete and decoder_valid and loop and not invalid)
        length_match = len(loop) == record["native_length"]
        raw_generations.append({
            "attempt": attempt,
            "full_sequence": record["pt1"] + loop + record["pt2"],
            "loop": loop,
            "length": len(loop),
            "valid_generation": valid_generation,
            "complete": bool(complete),
            "length_match": length_match,
        })
        if isinstance(draw, dict):
            raw_generations[-1].update({key: draw[key] for key in
                                       ('valid', 'terminated', 'failure', 'raw_token_ids') if key in draw})
        decoder_failure = draw.get('failure') if isinstance(draw, dict) else None
        if decoder_failure or not decoder_valid:
            failures.append({'attempt': attempt, 'code': decoder_failure or 'invalid_generation',
                             'detail': 'decoder rejected the raw completion'})
        elif not complete:
            failures.append({'attempt': attempt, 'code': 'incomplete_generation',
                             'detail': 'EOS was not emitted within the decode cap'})
        elif not loop:
            failures.append({'attempt': attempt, 'code': 'empty_generation',
                             'detail': 'generated loop is empty'})
        elif invalid:
            failures.append({
                "attempt": attempt,
                "code": "invalid_amino_acid",
                "detail": f"generated loop contains unsupported residues: {invalid}",
            })
        elif not length_match and getattr(sampler, "track", "fixed_native_length") == "fixed_native_length":
            failures.append({
                "attempt": attempt,
                "code": "length_mismatch",
                "detail": f"generated {len(loop)} residues, expected {record['native_length']}",
            })
        retain = length_match or getattr(sampler, "track", "fixed_native_length") == "length_free"
        if valid_generation and retain:
            valid_samples.append(loop)

    attempts = len(samples)
    valid_attempts = sum(item["valid_generation"] for item in raw_generations)
    length_matches = sum(
        item["valid_generation"] and item["length_match"] for item in raw_generations
    )
    length_policy = getattr(sampler, "length_metadata", None)
    if generation_failed:
        length_policy = {"mode": "unavailable", "source": "generation_failed"}
    elif length_policy is None:
        length_policy = {
            "mode": "fixed",
            "source": "native_mask_count",
            "minimum": record["native_length"],
            "maximum": record["native_length"],
            "selected_length": record["native_length"],
        }
    return {
        "example_id": record["example_id"],
        "native": record["native"],
        "native_length": record["native_length"],
        "length_policy": length_policy,
        "generation_failed": generation_failed,
        "samples": valid_samples,
        "attempts": attempts,
        "valid_attempts": valid_attempts,
        "invalid_attempts": attempts - valid_attempts,
        "length_matches": length_matches,
        "length_mismatches": valid_attempts - length_matches,
        "fixed_length_acceptance_rate": (length_matches / attempts) if attempts else 0.0,
        "complete": len(valid_samples) == num_samples,
        "raw_generations": raw_generations,
        "failures": failures,
        "prompt": getattr(sampler, "prompt_info", None),
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
    }


def build_payload(
    args,
    manifest: dict[str, Any],
    manifest_hash: str,
    run_config: dict[str, Any],
    run_config_path: Path,
    sampler,
) -> dict[str, Any]:
    import torch
    import transformers

    weights_path = args.checkpoint / "model.safetensors"
    config_path = args.checkpoint / "config.json"
    if not weights_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"checkpoint lacks model.safetensors/config.json: {args.checkpoint}")
    track = (manifest.get("task") or {}).get("track", "fixed_native_length")
    length_artifacts = None
    from cdrseqgen.methods.registry import checkpoint_method
    method = checkpoint_method(run_config).name
    extra_model = {}
    if run_config.get('model_type') != 'scratch':
        extra_model['inference_files_sha256'] = {
            path.name: sha256_file(path) for path in pretrained_input_files(args.checkpoint)}
    sampling_policy = getattr(getattr(sampler, 'tokenizer', None), 'causal_sampling_metadata', None)
    if sampling_policy is not None:
        extra_model['causal_sampling'] = sampling_policy
    if track == "length_free" and method == 'mlm':
        length_config_path = args.checkpoint / LENGTH_CONFIG_FILENAME
        length_weights_path = args.checkpoint / LENGTH_HEAD_FILENAME
        length_artifacts = {
            "config_sha256": sha256_file(length_config_path),
            "weights_sha256": sha256_file(length_weights_path),
        }
    repo = Path(__file__).resolve().parents[2]
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter": "cdrseqgen",
        "benchmark_id": manifest["benchmark_id"],
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_hash,
            "schema_version": manifest.get("schema_version"),
        },
        "model": {
            "method": method,
            "name": run_config["model_type"],
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(weights_path),
            "config_sha256": sha256_file(config_path),
            "run_config_sha256": sha256_file(run_config_path),
            "run_config": run_config,
            "backbone": sampler.backbone.id,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "device": str(sampler.device),
            "length_predictor": length_artifacts,
            **extra_model,
            "execution_source": execution_source_identity(),
            **git_state(repo),
        },
        "sampling": {
            "track": track,
            "length_policy": ("native_mask_count" if track == "fixed_native_length"
                              else "learned" if method == 'mlm' else 'eos'),
            "max_new_tokens": getattr(sampler, 'max_new_tokens', None),
            "num_samples": args.num_samples,
            "decoding": args.decoding,
            "temperature": args.temperature,
            "length_temperature": args.length_temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "base_seed": args.seed,
            "record_limit": args.limit,
            "duplicates_preserved": True,
            "raw_generations_preserved": True,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "records": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    from cdrseqgen.methods.registry import decoding_choices
    parser.add_argument("--decoding", choices=decoding_choices(), default=None)
    parser.add_argument('--max-new-tokens', type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--length-temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.temperature == 0 and args.num_samples != 1:
        raise ValueError("greedy decoding (--temperature 0) supports exactly one sample")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    manifest_hash = sha256_file(args.manifest)
    with args.manifest.open() as handle:
        manifest = json.load(handle)
    records = manifest.get("records") or []
    if args.length_temperature < 0:
        raise ValueError("--length-temperature must be non-negative")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[:args.limit]
    if not records:
        raise ValueError("manifest contains no selected records")

    run_config, run_config_path = load_run_config(args.checkpoint)
    validate_checkpoint_contract(run_config, manifest)
    args.decoding = resolve_decoding(run_config, args.decoding)
    track = (manifest.get("task") or {}).get("track", "fixed_native_length")
    sampler = build_sampler(args.checkpoint, run_config, args.device, track, args.max_new_tokens)
    expected = build_payload(
        args, manifest, manifest_hash, run_config, run_config_path, sampler
    )
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"output already exists; pass --resume to continue: {args.output}")
        with args.output.open() as handle:
            payload = json.load(handle)
        assert_resume_compatible(payload, expected)
    else:
        payload = expected

    completed = {record["example_id"] for record in payload["records"]}
    total = len(records)
    for position, record in enumerate(records, start=1):
        if record["example_id"] in completed:
            print(f"[{position}/{total}] skip {record['example_id']} (already complete)")
            continue
        print(f"[{position}/{total}] sample {record['example_id']}", flush=True)
        result = run_example(
            record,
            sampler,
            num_samples=args.num_samples,
            decoding=args.decoding,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            length_temperature=args.length_temperature,
            base_seed=args.seed,
        )
        payload["records"].append(result)
        atomic_write_json(args.output, payload)
        print(
            f"  retained={len(result['samples'])}/{args.num_samples} "
            f"failures={len(result['failures'])} runtime={result['runtime_seconds']:.2f}s",
            flush=True,
        )

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    failed_records = [r for r in payload["records"] if r.get("generation_failed")]
    payload["summary"] = {
        "records_requested": total,
        "records_completed": len(payload["records"]),
        "records_with_full_sample_count": sum(r["complete"] for r in payload["records"]),
        "records_with_generation_failure": len(failed_records),
        "failed_example_ids": [r["example_id"] for r in failed_records],
        "total_attempts": sum(r["attempts"] for r in payload["records"]),
        "total_valid_attempts": sum(r["valid_attempts"] for r in payload["records"]),
        "total_invalid_attempts": sum(r["invalid_attempts"] for r in payload["records"]),
        "total_length_matches": sum(r["length_matches"] for r in payload["records"]),
        "total_length_mismatches": sum(r["length_mismatches"] for r in payload["records"]),
        "total_runtime_seconds": sum(r["runtime_seconds"] for r in payload["records"]),
    }
    atomic_write_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output}")
    if failed_records:
        print(
            f"WARNING: {len(failed_records)}/{len(payload['records'])} examples produced no "
            f"candidates: {[r['example_id'] for r in failed_records][:5]}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed records remain saved and can be resumed.", file=sys.stderr)
        raise SystemExit(130)
