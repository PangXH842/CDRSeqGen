# OAS4 length-free validation: IgCraft as an added external generator

**Prepared and run 9 October 2026; audited complete (see Completion).** The GPU stages were user-run; preparation
was CPU-only: environment, weights, flank-region table, adapter, plan validation, and a plumbing check of the
adapter and of `blind_check.py` on the real model's code path with 3 sampler steps.

IgCraft is a 300M-parameter Bayesian-flow-network model of paired human antibodies with public weights
([repository](https://github.com/mgreenig/IgCraft), commit `4a053de`; weights revision `18eba7ff`, MIT). This
batch adds it to the finished eight-arm comparison on the same 150 validation frameworks, 20 attempts each, the
same IgBert heavy-chain judge, and the same shared scorer. The finished batches are not modified.

## What IgCraft sees

| Item | Policy |
|---|---|
| Heavy chain | IMGT FWR1, CDR1, FWR2, CDR2, FWR3 and FWR4, from `oas4_h3_validation_length_free_v1_regions.json` (flanks only, built from the corpus and re-verified against the manifest's `pt1`/`pt2`) |
| Hidden | Heavy CDR3 **and the whole light chain**, masked with pads included (`fix_length=false`, `reveal_pads=false`), so neither the native loop nor its length is visible |
| Hidden content | Fixed placeholders under the mask; they are ignored by the sampler and `blind_check.py` tests that |
| Loop length | However many of the 30 CDR3 slots the model fills with residues rather than pads |
| Sampling | IgCraft's own inpainting defaults: 20 steps, 32 particles, temperature 0.1. The matched temperature 1.0 / top-p 0.9 are recorded but **not applied** (`sampling_controls_applied: false`) |
| Attempts | Exactly 20 per framework in batches of 20; invalid attempts (an all-pad slot) are kept as failures and not replaced |

## Run it

All commands run from `/mnt/hdd/pang/CDRSeqGen`. Use GPU 0 (GPU 1 is shared). `IGC` is the environment built on the
data disk (root is nearly full); `CDR` is the main environment.

```bash
cd /mnt/hdd/pang/CDRSeqGen
export IGC=/mnt/hdd/pang/envs/igcraft/bin/python CDR=/home/pang/miniconda3/envs/cdrprj/bin/python
RECIPE=benchmark/recipes/oas4_h3_igcraft.json
```

**1. Plan (CPU).** Validates all 150 frameworks against IgCraft's slot sizes, the pinned digests of the checkpoint,
config and region table, and both environments.

```bash
./scripts/run.sh benchmark plan --recipe $RECIPE --stage full --external-python igcraft=$IGC
```

**2. Information-budget test (GPU).** Runs the real adapter on the manifest, again as a determinism control, and on a
copy with a different hidden H3 (other length and residues) and swapped identifiers. Expect `blind check pass`.

```bash
CUDA_VISIBLE_DEVICES=0 $CDR benchmark/blind_check.py \
  --manifest benchmark/manifests/oas4_h3_validation_length_free_smoke_v1.json --limit 4 \
  --workdir results/benchmark/igcraft_blind_check -- \
  $IGC benchmark/adapters/igcraft.py --manifest '{manifest}' --output '{output}' \
  --checkpoint ../IgCraft/igcraft-model/checkpoints/final.ckpt --config ../IgCraft/igcraft-model/.hydra/config.yaml \
  --flank-regions benchmark/manifests/oas4_h3_validation_length_free_v1_regions.json --num-samples 5 --seed 42 --device cuda
```

- `pass`: nothing hidden reached the candidates. Keep `results/benchmark/igcraft_blind_check/blind_check.json`.
- `inconclusive`: the repeat run differed, so GPU nondeterminism blocks the comparison. Retry once with `--no-compile`
  appended to the adapter command; if it persists, report it rather than proceeding.
- `leak`: stop. Hidden content changed the output.

**3. Smoke batch (5 frameworks x 5 attempts).** Checks the compile path, memory, validity and runtime on the GPU.

```bash
./scripts/run.sh benchmark smoke --recipe $RECIPE --external-python igcraft=$IGC --gpu 0 \
  --output results/benchmark/oas4_h3_igcraft_smoke_20261009
```

Look for: `retained=5/5` and `failures=0` per framework; lengths spread over a plausible range (not all 1-3
residues); the per-framework runtime. Multiply it by 150 for the full run (my back-of-envelope estimate is one to
a few hours; the smoke measures it). If the first framework spends minutes compiling, that is `torch.compile` and
happens once. If compilation fails, copy the recipe to `oas4_h3_igcraft_nocompile.json` with `"compile": false`
and use that. If you see `generation_exception ... out of memory`, lower `batch_size` the same way.

**4. Full batch (150 x 20 = 3,000 attempts).**

```bash
./scripts/run.sh benchmark run --recipe $RECIPE --external-python igcraft=$IGC --gpu 0 \
  --output results/benchmark/oas4_h3_igcraft_20261009
./scripts/run.sh benchmark status results/benchmark/oas4_h3_igcraft_20261009
```

Interrupted? `./scripts/run.sh benchmark resume results/benchmark/oas4_h3_igcraft_20261009 --gpu 0` continues from the
saved frameworks and refuses if the model, sampling, source or environment changed. The batch scores its candidates
with the shared judge at the end, in the main environment, offline.

**5. Merge into the eight-arm comparison and run the wider paired analysis (CPU).**

```bash
./scripts/run.sh benchmark extend plan --added results/benchmark/oas4_h3_igcraft_20261009
./scripts/run.sh benchmark extend run  --added results/benchmark/oas4_h3_igcraft_20261009
```

This verifies the follow-up batch's sealed hashes, merges only if the manifest, judge revision and scoring policy
match, and writes a new directory with the merged scores, the capacity table (with IgCraft's parameter breakdown)
and the paired intervals. If IgCraft returned invalid attempts the merge refuses; add `--allow-incomplete` to
accept that explicitly, and the shortfall stays in the accounting.

## Send back

The `blind_check` verdict line, the smoke log (every `retained=` line), and the tail of the full-batch `status` output.
I will audit the saved artifacts and update the manuscript tables from the extension directory.

## Reading the result

- The family widens: the base 13 contrasts become 20 (each local arm against IgLM and IgCraft, and all eight non-empirical
  arms against the empirical pool; IgLM versus IgCraft is not a planned contrast). Bonferroni intervals for the old pairs
  therefore get wider in the extension. The base analysis is untouched and stays the published 13.
- IgCraft has about 300M parameters (of which 17.3M is a frozen structure encoder that sequence-only inpainting does
  not use) against 0.8-0.9M for the scratch arms and 12.9M for IgLM. Scale and tuning differ, so this is a framework
  demonstration, not an architecture ranking.
- Its pretraining exposure to these OAS donors is unverified (see `benchmark/pretraining_overlap.md`).
- Its default temperature (0.1) is far sharper than the matched 1.0, so expect lower diversity; judge it on the
  supporting columns, not as a like-for-like sampling comparison.
- `fix_length=false` is the length-free setting but is not IgCraft's default, so its length behaviour here is untested
  by its authors. 13 of 15,273 validation and 11 of 15,273 test H3 loops exceed its 30-residue slot; the 150 benchmark
  frameworks top out at 26.

## Completion (audited 9 October 2026)

User-run: `blind_check` (4 frameworks, GPU, official settings), smoke, the full batch (16:30-18:04) and the
extension (18:30). **No rerun is needed.** Outputs: `results/benchmark/igcraft_blind_check`,
`oas4_h3_igcraft_smoke_20261009`, `oas4_h3_igcraft_20261009`, and
`20261009_183032_oas4_h3_igcraft_20261009_extension`.

- **Information budget:** `blind_check` passed; the repeat run and the hidden-field run were identical.
- **Full batch:** 150 frameworks, 3,000/3,000 valid attempts, 0 failures, 3,000/3,000 judged. Generation took 1.41 h
  (median 33.8 s per framework; the first included compilation). Checkpoint, config and region-table digests match
  the recipe pins; the edit similarity recomputed with an independent Levenshtein implementation equals the scorer's.
- **Extension:** nine runs and 20 contrasts. The first eight runs equal the base comparison, the base directory is
  unchanged, and every old marginal interval and difference is identical; only their family intervals widened (13 -> 20).
- **Population caution:** IgCraft's conditional core recovery rests on 51 frameworks, against 102-117 for the other arms.

| Arm | Edit | Match % | MAE | Core (n) | Judge | Hamming diversity |
|---|---:|---:|---:|---:|---:|---:|
| IgCraft | 0.3769 | 7.90 | 4.540 | 0.1854 (51) | 5.562 | 0.635 |
| IgLM | 0.3880 | 8.87 | 3.707 | 0.1944 (104) | 4.047 | 0.785 |
| Scratch MLM/AR/S2S (six) | 0.3870-0.3919 | 6.83-9.70 | 3.855-4.351 | 0.2052-0.2292 (102-117) | 3.150-7.676 | 0.797-0.805 |
| Empirical | 0.2817 | 6.97 | 4.508 | 0.1208 (113) | 5.009 | 0.859 |

- **Edit similarity:** each scratch arm exceeds IgCraft by 0.010-0.015 with marginal intervals excluding zero. Under the 20-contrast
  Bonferroni family only the two MLM contrasts still exclude zero ([+0.0001, +0.0284] and [+0.0011, +0.0291]); AR and S2S span zero.
  IgCraft exceeds the empirical pool by +0.095 (family [+0.079, +0.113]). IgLM versus IgCraft was not a planned contrast
  (descriptive difference -0.011).
- **Length:** IgCraft's H3 loops average 12.4 residues (sd 1.4, max 18) against natives at 16.3 (sd 4.0, max 26); every other
  arm averages 15.2-16.4. 1.0% of its slots had interior gaps (removed, as IgCraft's own decoder does) and 0.4% were exact
  training H3 copies, as for the other generative arms (0.1-0.5%). No draw copied a native loop.
- **Not established:** whether the compressed lengths are a property of the model or of these sampler settings
  (`fix_length=false`, temperature 0.1, 32 particles). A temperature-1.0 sensitivity batch has not been run.
