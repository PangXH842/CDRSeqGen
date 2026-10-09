#!/usr/bin/env python3
"""Write the experiment specs for the weekend queue, derived from the completed no-partner control specs.

Every trial is a copy of a completed one (scratch MLM / AR / S2S, no partner chain, same optimiser, selection and
generation settings) with only the seed or the model size changed, so the new checkpoints are comparable with the
finished ones. Three kinds of batch, one spec per method so the queue can run the methods side by side:

  seeds_<method>.json         seeds 44-48
  repeat_<method>_r<k>.json   seed 42 again with an identical config: run-to-run variation at a fixed seed.
                              One spec each, because the runner rejects duplicate resolved configs within a spec.
  capacity_<method>.json      medium / large / extra-large at seeds 42 and 43, everything else unchanged

Parameter counts (millions; MLM / AR / S2S): M 4.9 / 4.8 / 5.5, L 10.9 / 10.7 / 12.5 (IgLM has 12.9), XL 25.5 / 25.3 / 29.5.
The learning rate is deliberately not retuned, so the scan isolates size; larger models may deserve their own.

  python scripts/queue_specs.py            # (re)writes configs/experiments/weekend_20261009/
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs/experiments/weekend_20261009'
METHODS = ('mlm', 'ar', 's2s')
SEEDS = (44, 45, 46, 47, 48)
CAPACITY_SEEDS = (42, 43)
SIZES = {'M': dict(hidden_size=256, num_layers=6, depth=3), 'L': dict(hidden_size=384, num_layers=6, depth=3),
         'XL': dict(hidden_size=512, num_layers=8, depth=4)}


def read(name):
    return json.loads((ROOT / 'configs/experiments' / name).read_text())


def template(method):
    """(spec skeleton without trials, the finished seed-42 trial to copy)."""
    if method == 'mlm':
        spec = read('20261007_mlm_no_partner_controls.json')
    else:
        spec = read('20261006_no_partner_controls.json')
    trial = next(t for t in spec['trials'] if t['name'] == f'{method}_scratch_no_partner_s42')
    skeleton = {k: v for k, v in spec.items() if k not in ('name', 'trials')}
    return skeleton, trial


def trial(method, name, seed, size=None):
    _, source = template(method)
    new = copy.deepcopy(source)
    new['name'] = name
    new['overrides'].update(seed=seed, data_seed=seed)
    if size:
        dims = SIZES[size]
        new['overrides'].update(hidden_size=dims['hidden_size'], num_layers=dims['num_layers'])
        if method == 's2s':
            new['overrides'].update(num_encoder_layers=dims['depth'], num_decoder_layers=dims['depth'],
                                    num_layers=2 * dims['depth'])
    return new


def spec(method, name, trials):
    skeleton, _ = template(method)
    return {'name': name, **copy.deepcopy(skeleton), 'trials': trials}


def specs():
    out = {}
    for m in METHODS:
        out[f'seeds_{m}'] = spec(m, f'{m}_seed_expansion',
                                 [trial(m, f'{m}_scratch_no_partner_s{s}', s) for s in SEEDS])
        for k in (1, 2):
            out[f'repeat_{m}_r{k}'] = spec(m, f'{m}_repeat_r{k}', [trial(m, f'{m}_scratch_no_partner_s42_r{k}', 42)])
        out[f'capacity_{m}'] = spec(m, f'{m}_capacity', [
            trial(m, f'{m}_scratch_no_partner_{size}_s{s}', s, size) for size in SIZES for s in CAPACITY_SEEDS])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, content in specs().items():
        (OUT / f'{stem}.json').write_text(json.dumps(content, indent=2) + '\n')
    print(f'Wrote {len(specs())} specs to {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
