#!/usr/bin/env python3
"""Generate attempt-bounded IgCraft outputs for a length-free CDRSeqGen benchmark manifest.

IgCraft inpaints IMGT regions of a paired antibody, so each framework is given as its heavy-chain flank regions
(benchmark/export_regions.py) with CDR3 and the whole light chain hidden. Hidden regions are masked, pad
positions included (`reveal_pads=False`, `fix_length=False`), so neither the native loop nor its length reaches
the model: the loop length is however many of its 30 slots the model fills with residues rather than pads.
Hidden content is a fixed placeholder that the masks ignore; benchmark/blind_check.py tests that claim.

The step count, particle count and temperature (0.1) are IgCraft's documented inpainting defaults and are model
controls, not the matched neural budget: the matched temperature/top-p are recorded but not applied, as for the
empirical sampler. Run it in the `igcraft` environment; nothing here needs the training stack.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Direct execution puts benchmark/adapters first on sys.path, where this file would shadow the installed
# `igcraft` package. Replace that entry with the repository root before any lazy IgCraft import.
ADAPTER_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == ADAPTER_DIR:
    sys.path.pop(0)
    sys.path.insert(0, str(ADAPTER_DIR.parents[1]))

from benchmark._common import (
    ADAPTER_SCHEMA_VERSION as SCHEMA_VERSION,
    AMINO_ACIDS,
    assert_resume_compatible,
    atomic_write_json,
    execution_source_identity,
    repository_commit,
    sha256_file,
    stable_example_seed,
    validate_manifest_record,
)
from benchmark.export_regions import FLANK_REGIONS, flank_regions

REGIONS = ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4')
# Masked, so the model never reads them. Any residues that fit the slots would do; a fixed kappa-like chain
# keeps every input well formed and identical across frameworks.
HIDDEN_CDR3 = 'ARDYFDY'
HIDDEN_LIGHT = dict(fwr1='DIQMTQSPSSLSASVGDRVTITC', cdr1='QSISSY', fwr2='LNWYQQKPGKAPKLLIY', cdr2='AAS',
                    fwr3='SLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYC', cdr3='QQSYSTP', fwr4='FGQGTKVEIK')
HIDDEN_REGIONS = ['H-cdr3', *(f'L-{region}' for region in REGIONS)]


class InvalidGeneration(ValueError):
    """A sampled slot could not be read as an infilled loop."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def config_capacities(config_path: Path) -> dict[str, dict[str, int]]:
    """Per-region slot sizes ('vh' and 'vl') from IgCraft's model config."""
    import yaml

    tokenizers = yaml.safe_load(Path(config_path).read_text())['model']['datamodule']
    return {'vh': dict(tokenizers['vh_tokenizer']['region_lengths']),
            'vl': dict(tokenizers['vl_tokenizer']['region_lengths'])}


def capacity_problems(regions: dict[str, str], capacities: dict[str, dict[str, int]]) -> list[str]:
    """Regions that overflow their slot. IgCraft's encoder would pad a negative amount and misalign the chain."""
    problems = [f"{name} has {len(regions[name])} residues for {capacities['vh'][name]} slots"
                for name in FLANK_REGIONS if len(regions[name]) > capacities['vh'][name]]
    if len(HIDDEN_CDR3) > capacities['vh']['cdr3']:
        problems.append('hidden CDR3 placeholder overflows its slot')
    problems += [f'hidden light {name} placeholder overflows its slot'
                 for name, residues in HIDDEN_LIGHT.items() if len(residues) > capacities['vl'][name]]
    return problems


def extract_loop(slot: str) -> str:
    """The inpainted loop: the CDR3 slot with pad characters removed, as IgCraft's own decoder does."""
    loop = slot.replace('-', '')
    if not loop:
        raise InvalidGeneration('empty_generation', 'every CDR3 slot position is a pad')
    invalid = sorted(set(loop) - AMINO_ACIDS)
    if invalid:
        raise InvalidGeneration('invalid_amino_acid', f'CDR3 slot contains unsupported residues: {invalid}')
    return loop


class IgCraftSampler:
    """Draws CDR3 loops for one framework at a time, following scripts/inpaint.py with the length switches off."""

    def __init__(self, checkpoint: Path, config: Path, device: str, num_steps: int, num_particles: int,
                 temperature: float, compile_network: bool):
        import torch
        from omegaconf import OmegaConf

        from igcraft.bfn.bfn import DiscreteBFNSDESolver
        from igcraft.model import load_model
        from igcraft.model.datamodule import seq_batch_to_aa1
        from igcraft.model.model import PairedSequenceBFNSampler

        self.torch = torch
        self.seq_batch_to_aa1 = seq_batch_to_aa1
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else
                                   'cpu' if device == 'auto' else device)
        settings = OmegaConf.create({'checkpoint': {'config_path': str(config), 'checkpoint_path': str(checkpoint),
                                                    'reset_optimizer': False}})
        # IgCraft's loader calls torch.load without a map_location, and this checkpoint was saved from a CUDA
        # device, so it would only load where that device exists. Start the tensors on the CPU and place the
        # model afterwards; the values are unchanged.
        torch_load = torch.load
        torch.load = lambda *args, **kwargs: torch_load(*args, **{'map_location': 'cpu', **kwargs})
        try:
            self.datamodule, self.model, _ = load_model(settings, compile=compile_network)
        finally:
            torch.load = torch_load
        self.missing_buffers, self.unexpected_keys = self._check_weights_loaded(checkpoint)
        self.model.to(self.device).eval()
        vh_regions = self.datamodule.vh_tokenizer.region_lengths
        self.sampler = PairedSequenceBFNSampler(
            vh_len=sum(vh_regions.values()), num_steps=num_steps, solver=DiscreteBFNSDESolver(),
            temperature=temperature, num_particles=num_particles)
        self.cdr3_slots = self.datamodule.vh_region_indices['cdr3']
        self.row = None

    def _check_weights_loaded(self, checkpoint: Path) -> tuple[list[str], list[str]]:
        """IgCraft loads with strict=False, so a renamed or missing tensor would leave random weights silently."""
        try:
            saved = self.torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')['state_dict']
        except RuntimeError:  # an archive that cannot be memory-mapped
            saved = self.torch.load(checkpoint, weights_only=True, map_location='cpu')['state_dict']
        names = {name for name, _ in self.model.named_parameters()}
        missing = sorted(names - set(saved))
        if missing:
            raise RuntimeError(f'{len(missing)} model parameters are absent from the checkpoint, e.g. {missing[:3]}')
        buffers = sorted(set(self.model.state_dict()) - set(saved) - names)
        return buffers, sorted(set(saved) - set(self.model.state_dict()))

    @property
    def capacities(self) -> dict[str, dict[str, int]]:
        return {'vh': dict(self.datamodule.vh_tokenizer.region_lengths),
                'vl': dict(self.datamodule.vl_tokenizer.region_lengths)}

    def provenance(self) -> dict[str, Any]:
        count = lambda module: sum(p.numel() for p in module.parameters())
        return {'parameters_network': count(self.model.network),
                'parameters_by_module': {name: count(child) for name, child in self.model.network.named_children()},
                'parameters_ema_network': count(self.model.ema_network),
                'parameters_registered': count(self.model), 'buffers_absent_from_checkpoint': self.missing_buffers,
                'checkpoint_keys_not_in_model': self.unexpected_keys, 'device': str(self.device)}

    def set_seed(self, seed: int) -> None:
        from pytorch_lightning import seed_everything

        seed_everything(seed, workers=False, verbose=False)

    def prepare(self, regions: dict[str, str]) -> None:
        """Encode the framework once: flanks as given, CDR3 and the light chain as masked placeholders."""
        problems = capacity_problems(regions, self.capacities)
        if problems:
            raise ValueError('; '.join(problems))
        heavy = {**regions, 'cdr3': HIDDEN_CDR3}
        vh = self.datamodule.vh_tokenizer.encode([heavy[name] for name in REGIONS])
        vl = self.datamodule.vl_tokenizer.encode([HIDDEN_LIGHT[name] for name in REGIONS])
        self.row = (vh, vl)

    def draw(self, count: int) -> list[str]:
        """`count` independent CDR3 slots, pads included, from one batched call."""
        vh, vl = (row.unsqueeze(0).repeat(count, 1) for row in self.row)
        batch = self.model.transfer_batch_to_device(((vh, vl), {}), self.device, 0)
        mask = self.datamodule.get_imgt_inpaint_mask(batch[0], HIDDEN_REGIONS, count, reveal_pads=False)
        with self.torch.no_grad():
            heavy, _ = self.sampler(self.model, cond_x=batch, cond_mask=mask, fix_length=False, progress_bar=False)
        return self.seq_batch_to_aa1(heavy[:, self.cdr3_slots], remove_gaps=False)


def load_flank_table(path: Path) -> dict[str, Any]:
    table = json.loads(Path(path).read_text())
    if table.get('schema_version') != 'cdrseqgen.flank_regions.v1':
        raise ValueError(f'unsupported flank-region table schema {table.get("schema_version")!r}')
    return table


def run_example(record: dict[str, Any], regions: dict[str, str], sampler, *, num_samples: int, batch_size: int,
                base_seed: int, max_errors: int, maximum_length: int | None = None) -> dict[str, Any]:
    """Draw exactly `num_samples` attempts for one framework, keeping every valid loop and every failure."""
    validate_manifest_record(record)
    seed = stable_example_seed(base_seed, record['example_id'])
    sampler.set_seed(seed)
    sampler.prepare(regions)

    started = time.perf_counter()
    samples: list[str] = []
    raw_generations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempts = 0
    consecutive_errors = 0

    while attempts < num_samples and consecutive_errors < max_errors:
        count = min(batch_size, num_samples - attempts)
        try:
            slots = sampler.draw(count)
        except Exception as exc:  # preserve a runtime failure in the artifact, charged to every attempt it cost
            for _ in range(count):
                attempts += 1
                failures.append({'attempt': attempts, 'code': 'generation_exception',
                                 'detail': f'{type(exc).__name__}: {exc}'})
            consecutive_errors += count
            continue
        for slot in slots:
            attempts += 1
            try:
                loop = extract_loop(slot)
            except InvalidGeneration as exc:
                failures.append({'attempt': attempts, 'code': exc.code, 'detail': exc.detail, 'slot': slot})
                consecutive_errors += 1
                continue
            consecutive_errors = 0
            raw_generations.append({
                'attempt': attempts, 'slot': slot, 'loop': loop, 'length': len(loop), 'valid_generation': True,
                'length_match': len(loop) == record['native_length'],
                'interior_gap': '-' in slot.rstrip('-'),
                'full_sequence': record['pt1'] + loop + record['pt2'],
            })
            samples.append(loop)  # length-free: duplicates and every length are kept; diversity is scored later
    if consecutive_errors >= max_errors:
        failures.append({'attempt': attempts, 'code': 'max_consecutive_errors',
                         'detail': f'stopped after {max_errors} consecutive failed attempts'})

    valid_attempts = len(raw_generations)
    length_matches = sum(item['length_match'] for item in raw_generations)
    return {
        'example_id': record['example_id'],
        'native': record['native'],
        'native_length': record['native_length'],
        'length_policy': {'mode': 'inferred', 'source': 'igcraft_pad_tokens', 'minimum': None,
                          'maximum': maximum_length},
        'samples': samples,
        'attempts': attempts,
        'valid_attempts': valid_attempts,
        'invalid_attempts': attempts - valid_attempts,
        'length_matches': length_matches,
        'length_mismatches': valid_attempts - length_matches,
        'fixed_length_acceptance_rate': (length_matches / attempts) if attempts else 0.0,
        'complete': len(samples) == num_samples,
        'raw_generations': raw_generations,
        'failures': failures,
        'seed': seed,
        'runtime_seconds': time.perf_counter() - started,
    }


def build_payload(args, manifest: dict[str, Any], manifest_hash: str, sampler, file_hashes: dict[str, Any]):
    track = (manifest.get('task') or {}).get('track')
    packages = {name: importlib.metadata.version(name) for name in ('igcraft', 'torch', 'pytorch-lightning', 'numpy', 'anarci')}
    return {
        'schema_version': SCHEMA_VERSION,
        'adapter': 'igcraft',
        'benchmark_id': manifest['benchmark_id'],
        'manifest': {'path': str(args.manifest.resolve()), 'sha256': manifest_hash,
                     'schema_version': manifest.get('schema_version')},
        'model': {
            'name': 'IgCraft', 'repository_commit': repository_commit(args.igcraft_repo),
            'hf_revision': args.hf_revision, 'packages': packages,
            'checkpoint_file': str(args.checkpoint.resolve()), 'checkpoint_sha256': file_hashes['checkpoint'],
            'config_file': str(args.config.resolve()), 'config_sha256': file_hashes['config'],
            **sampler.provenance(), 'execution_source': execution_source_identity(),
        },
        'sampling': {
            'track': track, 'num_samples': args.num_samples,
            # The matched neural budget, recorded for assert_comparable; IgCraft has no such controls.
            'temperature': args.temperature, 'top_p': args.top_p, 'base_seed': args.seed,
            'sampling_controls_applied': False,
            'note': 'Temperature/top-p describe the matched neural budget; IgCraft samples with its own controls below',
            'model_controls': {'num_steps': args.num_steps, 'num_particles': args.num_particles,
                               'temperature': args.igcraft_temperature, 'solver': 'DiscreteBFNSDESolver',
                               'fix_length': False, 'reveal_pads': False, 'batch_size': args.batch_size,
                               'compiled': not args.no_compile},
            'conditioning': {'heavy': 'IMGT FWR1, CDR1, FWR2, CDR2, FWR3 and FWR4 from the flank-region table',
                             'hidden': HIDDEN_REGIONS, 'hidden_content': 'fixed placeholders under a mask'},
            'flank_regions': {'path': str(args.flank_regions.resolve()), 'sha256': file_hashes['flank_regions']},
            'record_limit': args.limit, 'duplicates_preserved': True, 'raw_generations_preserved': True,
        },
        'started_at': datetime.now(timezone.utc).isoformat(),
        'finished_at': None,
        'records': [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--flank-regions', type=Path, required=True, help='Table from benchmark/export_regions.py')
    parser.add_argument('--checkpoint-sha256', help='Refuse to run unless the checkpoint has this digest')
    parser.add_argument('--hf-revision', help='Hugging Face revision the weights came from; provenance only')
    parser.add_argument('--num-samples', type=int, default=20, help='Attempts per framework; failures are not replaced')
    parser.add_argument('--temperature', type=float, default=1.0, help='Matched budget; recorded, not applied')
    parser.add_argument('--top-p', type=float, default=0.9, help='Matched budget; recorded, not applied')
    parser.add_argument('--num-steps', type=int, default=20)
    parser.add_argument('--num-particles', type=int, default=32)
    parser.add_argument('--igcraft-temperature', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=20, help='Attempts drawn per model call')
    parser.add_argument('--no-compile', action='store_true', help='Skip torch.compile of the network')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-errors', type=int, default=5)
    parser.add_argument('--limit', type=int, default=None, help='Run only the first N manifest records')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--igcraft-repo', type=Path, default=ADAPTER_DIR.parents[2] / 'IgCraft')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.num_samples, args.batch_size, args.max_errors, args.num_steps, args.num_particles) <= 0:
        raise ValueError('sample, batch, error, step and particle counts must be positive')
    if args.igcraft_temperature <= 0:
        raise ValueError('--igcraft-temperature must be positive')

    manifest_hash = sha256_file(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if (manifest.get('task') or {}).get('track') != 'length_free':
        raise ValueError('IgCraft runs only on length_free manifests: its fixed-length mode reads the native pads')
    records = manifest.get('records') or []
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit must be positive')
        records = records[:args.limit]
    if not records:
        raise ValueError('manifest contains no selected records')

    # Everything that can be wrong with the inputs is checked before the model is loaded.
    table = load_flank_table(args.flank_regions)
    capacities = config_capacities(args.config)
    regions_by_id = {}
    for record in records:
        validate_manifest_record(record)
        if record['example_id'] not in table['records']:
            raise ValueError(f"{record['example_id']} is absent from the flank-region table")
        regions = flank_regions(record, table['records'][record['example_id']])
        problems = capacity_problems(regions, capacities)
        if problems:
            raise ValueError(f"{record['example_id']} does not fit IgCraft: {'; '.join(problems)}")
        regions_by_id[record['example_id']] = regions
    file_hashes = {'checkpoint': sha256_file(args.checkpoint), 'config': sha256_file(args.config),
                   'flank_regions': sha256_file(args.flank_regions)}
    if args.checkpoint_sha256 and file_hashes['checkpoint'] != args.checkpoint_sha256:
        raise ValueError(f"checkpoint digest {file_hashes['checkpoint']} differs from the pinned {args.checkpoint_sha256}")

    sampler = IgCraftSampler(args.checkpoint, args.config, args.device, args.num_steps, args.num_particles,
                             args.igcraft_temperature, not args.no_compile)
    expected = build_payload(args, manifest, manifest_hash, sampler, file_hashes)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f'output already exists; pass --resume to continue: {args.output}')
        payload = json.loads(args.output.read_text())
        assert_resume_compatible(payload, expected)
    else:
        payload = expected

    completed = {record['example_id'] for record in payload['records']}
    total = len(records)
    for position, record in enumerate(records, start=1):
        if record['example_id'] in completed:
            print(f"[{position}/{total}] skip {record['example_id']} (already complete)")
            continue
        print(f"[{position}/{total}] sample {record['example_id']}", flush=True)
        result = run_example(record, regions_by_id[record['example_id']], sampler, num_samples=args.num_samples,
                             batch_size=args.batch_size, base_seed=args.seed, max_errors=args.max_errors,
                             maximum_length=capacities['vh']['cdr3'])
        payload['records'].append(result)
        atomic_write_json(args.output, payload)
        lengths = [len(sample) for sample in result['samples']]
        span = f'{min(lengths)}-{max(lengths)}' if lengths else 'none'
        print(f"  retained={len(result['samples'])}/{args.num_samples} lengths={span} "
              f"matches={result['length_matches']} failures={len(result['failures'])} "
              f"runtime={result['runtime_seconds']:.2f}s", flush=True)

    payload['finished_at'] = datetime.now(timezone.utc).isoformat()
    payload['summary'] = {
        'records_requested': total, 'records_completed': len(payload['records']),
        'records_with_full_sample_count': sum(r['complete'] for r in payload['records']),
        'total_attempts': sum(r['attempts'] for r in payload['records']),
        'total_valid_attempts': sum(r['valid_attempts'] for r in payload['records']),
        'total_invalid_attempts': sum(r['invalid_attempts'] for r in payload['records']),
        'total_length_matches': sum(r['length_matches'] for r in payload['records']),
        'total_length_mismatches': sum(r['length_mismatches'] for r in payload['records']),
        'total_runtime_seconds': sum(r['runtime_seconds'] for r in payload['records']),
    }
    atomic_write_json(args.output, payload)
    print(f"Saved {len(payload['records'])} records to {args.output}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; completed records remain saved and can be resumed.', file=sys.stderr)
        raise SystemExit(130)
