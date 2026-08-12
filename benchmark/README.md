# CDR-H3 benchmark manifests

The manifest is the frozen, model-independent input shared by CDRSeqGen and external baselines.
Model adapters may tokenize these strings differently, but must not reconstruct or reselect the
examples themselves.

## Included manifests

`manifests/oas_h3_test_sample_v1.json` contains a deterministic 25-record sample of the 2,111-record
clustered OAS test split. Keep it frozen until model and sampling settings are selected. The
`oas_h3_validation_smoke_v1.json` manifest is the corresponding 25-record development set for
adapter debugging and policy selection. Each companion `.sha256` file pins the exact artifact.

Each record contains:

- `example_id`: stable join key for model outputs and paired statistics
- `pt1`, `native`, `pt2`: framework-before, native CDR-H3, and framework-after strings
- `parent_sequence`: validated as `pt1 + native + pt2`
- `native_length`: fixed-length benchmark target
- `infill_range`: zero-indexed, end-exclusive coordinates for `IgLM.infill`
- IgLM chain/species conditioning tokens

The sample excludes light-chain and antigen context because pretrained IgLM's infill API consumes a
single antibody chain. Species and source-run provenance come from the processed records, and the
exporter verifies the processed file against its preprocessing metadata sidecar before writing a
manifest. It fails instead of guessing when species or stable source provenance is missing.

## Regenerate

From the repository root:

```bash
python benchmark/export_manifest.py \
  --input dataset/data/processed/processed_oas_tuning.json \
  --output benchmark/manifests/oas_h3_test_sample_v1.json \
  --sample-size 25 \
  --benchmark-id oas_h3_test_sample_v1
```

Omit `--sample-size 25` and change the benchmark/output names to export the full test split after the
sampling and scoring adapters pass the smoke experiment. To regenerate the development manifest,
add `--split validation` and use the validation names above.

## IgLM adapter

Run IgLM in its separate environment. Fixed-length redesign defaults to constrained decoding: the
adapter uses the pretrained checkpoint's underlying generation API to prevent end-of-span before the
native length and force it at that length. It does not change model weights. Raw draws and duplicates
are preserved so the shared scorer, rather than the sampler, measures diversity.

```bash
conda run -n iglm python benchmark/adapters/iglm.py \
  --manifest benchmark/manifests/oas_h3_validation_smoke_v1.json \
  --output benchmark/results/iglm_validation_smoke.json \
  --num-samples 20 \
  --max-attempts 25 \
  --length-policy constrain \
  --temperature 1.0 \
  --top-p 0.9 \
  --seed 42
```

Use `--length-policy filter` to sample IgLM's natural variable-length distribution and retain only
native-length candidates in `samples`. Every valid draw remains in `raw_generations`, including its
loop, length, and match status; the attempt cap prevents low-acceptance cases from running forever.
Completed examples are written atomically after each framework, and the same command with `--resume`
continues a compatible interrupted run.

The verified 25-framework constrained smoke run produced 125/125 valid, length-matched draws with no
adapter failures. Results live under ignored `benchmark/results/`; they are local run artifacts, not
frozen benchmark inputs.

## CDRSeqGen adapter

Run CDRSeqGen in its own environment and point the adapter at a self-describing checkpoint directory:

```bash
conda run -n cdrprj python benchmark/adapters/cdrseqgen.py \
  --manifest benchmark/manifests/oas_h3_validation_smoke_v1.json \
  --checkpoint results/sweeps/h3_20260811_181946/h3_OAS_scratch_lr1e-3_wu0.06_wd0.01_s42 \
  --output benchmark/results/cdrseqgen_validation_smoke.json \
  --num-samples 20 \
  --decoding iterative \
  --temperature 1.0 \
  --top-p 0.9 \
  --seed 42
```

The adapter validates the target loop, source filename, and context regime against `run_config.json`;
recreates training-time tokenization, truncation, and native mask count; and hashes the weights,
model config, and run config. It supports both `independent` and `iterative` decoders. The decoder and
sampling point used above are development settings, not yet a frozen final comparison policy.

The verified 25-framework smoke run produced 125/125 valid, native-length draws with zero failures.
Four frameworks exceeded this checkpoint's 128-token limit; training-compatible assembly trimmed
2–3 far-C-terminal `pt2` tokens, and each record reports that explicitly in `prompt`.

## Adapter contract

Both adapters should emit the same model-independent record shape:

```json
{
  "example_id": "oas:...",
  "native": "...",
  "samples": ["...", "..."],
  "attempts": 20,
  "raw_generations": [
    {
      "attempt": 1,
      "loop": "...",
      "length": 15,
      "valid_generation": true,
      "length_match": true
    }
  ],
  "failures": [],
  "complete": true
}
```

For the fixed-length track, retain only samples whose length equals `native_length`, while preserving
the total attempt count and length-mismatch failures. Do not alter the manifest to store generated
outputs.

## Shared scorer

Score both adapter outputs in one invocation so the scorer can verify that they use the same
manifest records, requested sample count, temperature, `top_p`, and base seed:

```bash
conda run -n cdrprj python benchmark/scorer.py \
  --manifest benchmark/manifests/oas_h3_validation_smoke_v1.json \
  --input cdrseqgen=benchmark/results/cdrseqgen_validation_smoke.json \
  --input iglm=benchmark/results/iglm_validation_smoke.json \
  --output benchmark/results/validation_smoke_scores.json \
  --judge Exscientia/IgBert \
  --judge-max-len 128
```

The scorer validates every record against the manifest before calculating whole-span, two-residue
anchor-trimmed core, and anchor-only AAR; duplicate-preserving Hamming diversity; unique fraction;
per-position entropy; distance from the native (`novelty_to_native_mean`); and optional independent
judge pseudo-perplexity. It also retains position-resolved recovery for later plots and reports
short/medium/long H3 buckets.

Aggregate metric means are macro-averaged over frameworks, making the framework the statistical
unit. Attempt accounting is also reported separately: `valid_generation_rate` uses all attempts,
`length_match_rate` uses syntactically valid draws, and `fixed_length_acceptance_rate` uses all
attempts. Runtime per retained candidate covers generation only, not judge scoring. Partial adapter
outputs are rejected by default; `--allow-partial` exists only for development diagnostics, and two
inputs must still contain the same framework IDs.

The verified five-draw, 25-framework validation smoke comparison produced the following development
numbers with the same IgBert revision for both runs:

| adapter | whole AAR | core AAR | anchor AAR | diversity | entropy (bits) | IgBert ppl |
|---|---:|---:|---:|---:|---:|---:|
| CDRSeqGen | 0.332 | 0.149 | 0.900 | 0.603 | 1.286 | 10.085 |
| IgLM | 0.216 | 0.108 | 0.552 | 0.762 | 1.620 | 7.001 |

These are scorer smoke results, not benchmark claims: the sample is a small validation slice, the
decoder policy is not frozen, and IgLM's OAS pretraining creates an unresolved overlap caveat. The
final test manifest remains unused.
