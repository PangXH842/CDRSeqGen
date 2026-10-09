# CDR-H3 benchmark manifests

The manifest is the frozen, model-independent input shared by CDRSeqGen and external baselines.
Model adapters may tokenize these strings differently, but must not reconstruct or reselect the
examples themselves.

## Adding a generator

External generators run in their own environments and meet the scorer only through adapter-output JSON.
[`adapters/README.md`](adapters/README.md) specifies that contract and its information-budget rule;
[`externals.py`](externals.py) registers an adapter so `run.py` can plan and launch it, and
[`blind_check.py`](blind_check.py) tests that an adapter's candidates do not depend on the hidden native
loop. Recipes name externals under `"externals"`; the completed batch's frozen recipe still uses the
legacy `"iglm"` key, which the runner reads as one external. A batch may list externals without `arms`, and
[`extend.py`](extend.py) merges such a batch into a finished comparison. The IgCraft protocol
([recipe](recipes/oas4_h3_igcraft.json), [commands and checks](recipes/oas4_h3_igcraft.md)) is the worked example;
it has completed and been audited (see its Completion section).

## Completed OAS4 three-method validation comparison

The [current recipe](recipes/oas4_h3_length_free.json) and [protocol](recipes/oas4_h3_length_free.md)
reuse six scratch MLM/AR/S2S no-partner checkpoints and fitted MLM length heads, with one frozen
IgLM baseline. Both the four-arm smoke and seven-arm full batch passed their artifact audits.
The full batch uses the same 150 validation frameworks and 20 attempts per arm: **21,000/21,000
valid candidates with complete judge coverage**. [Full audit](../results/reviews/20261008-benchmark-full.md).

```bash
./scripts/run.sh benchmark status results/benchmark/oas4_h3_length_free_20261007
python results/reviews/20261008-benchmark-full-audit.py
```

All generators receive heavy-chain flanks without partner/antigen or native length. The shared
IgBert judge receives heavy-chain context only; older partner-present judge values remain separate.
MLM uses learned length classes; AR/S2S/IgLM terminate naturally within 97 new tokens. IgLM gets
the same attempt budget, with no replenishment of invalid draws or forced EOS at the cap.

Mean edit similarity spans 0.3870–0.3919; IgLM has the lowest observed length MAE (3.707 residues).
Conditional core recovery uses 102–117 frameworks and 205–291 matched candidates per arm.
These descriptive validation values do not establish architecture superiority or significance.
The [completed follow-up](recipes/oas4_h3_followup.md) passed its [artifact audit](../results/reviews/20261008-benchmark-followup.md):
3,000 new training-only empirical H3 draws, all split labels verified, 13 primary paired contrasts
and saved capacity/actual exposure. The combined eight-arm benchmark has 24,000 valid candidates
with full recorded judge coverage. Original seven rows are unchanged. Empirical edit is 0.2817;
all learned-system-versus-empirical family intervals are positive, while local-checkpoint-versus-
IgLM intervals span zero. The 150 frameworks occupy 150 distinct recovered groups, so resampling
reduces to paired frameworks. Fixed-pool exploratory intervals exclude training/generation variance.
No completion rerun is needed; leakage decisions precede final testing.

```bash
./scripts/run.sh benchmark followup status results/benchmark/oas4_h3_followup_20261008
python results/reviews/20261008-benchmark-followup-audit.py
```

Reproduction has CPU group reconstruction/analysis and GPU judging of empirical draws only;
existing neural rows are reused. See the protocol for fresh-output commands and reporting limits.

## Completed local split-overlap audit

The [audit recipe](recipes/oas4_h3_leakage.md) completed and passed its [artifact review](../results/reviews/20261008-leakage-audit.md). It searches
every same-length reference H3 for all validation/test queries: validation→train,
test→train and test→validation. It reuses verified groups, reports the current
150-framework benchmark separately, and saves exact sequence/context overlaps and
candidate test eligibility. Splits and earlier results remain unchanged.

```bash
./scripts/run.sh benchmark leakage status results/benchmark/oas4_h3_leakage_20261008
python results/reviews/20261008-leakage-audit.py
```

This batch is CPU-only; no models are loaded. Resume reuses sealed 256-query chunks.
No completion rerun is needed. The existing benchmark has 3/150 affected frameworks;
the stated test rule retains 14,818/15,273 records (97.02%), preserving all donors and
represented lengths. Review those counts before selecting the final cohort. The
[external pretraining note](pretraining_overlap.md) records known OAS provenance
and unverified membership; local separation cannot establish external independence.
See [baseline coverage](baseline_scope.md) for the current references and one targeted
external-generator feasibility check before freezing a broader final comparison.

Plans validate inputs/environments without loading weights. `smoke`, `run` and `resume` execute
models and are user-run; `status` reads saved outputs. Reproduction commands are in the recipe;
choose a new output directory for a new batch. Source/input/environment fingerprints stay fixed
within a running batch. Saved checkpoints retain their actual training provenance; the completed
inference outputs record frozen source hashes and a dirty checkout.

The one-donor commands below are historical examples with their original manifests and source
versions. Their numbers are not pooled with the OAS4 comparison.

## Included manifests

`manifests/oas_h3_test_sample_v1.json` contains a deterministic 25-record sample of the 2,111-record
clustered OAS test split. Keep it frozen until model and sampling settings are selected. The
`oas_h3_validation_smoke_v1.json` manifest is the corresponding 25-record development set for
adapter debugging and policy selection. Each companion `.sha256` file pins the exact artifact.

`oas_h3_validation_length_free_v1.json` selects the same validation frameworks but declares the
separate `length_free` contract; it must never be mixed with fixed-track adapter outputs in one
score artifact.

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
  --input data/processed/processed_oas_tuning.json \
  --output benchmark/manifests/oas_h3_test_sample_v1.json \
  --sample-size 25 \
  --benchmark-id oas_h3_test_sample_v1
```

Pass `--track length_free` and a distinct benchmark/output name to export the same deterministic
membership for the length-free track; the native remains in the manifest for scoring only.

Omit `--sample-size 25` and change the benchmark/output names to export the full test split after the
sampling and scoring adapters pass the smoke experiment. To regenerate the development manifest,
add `--split validation` and use the validation names above.

## IgLM adapter

Run IgLM in its separate environment. Fixed-length redesign defaults to constrained decoding: the
adapter uses the pretrained checkpoint's underlying generation API to prevent end-of-span before the
native length and force it at that length. It does not change model weights. Raw draws and duplicates
are preserved so the shared scorer, rather than the sampler, measures diversity.

> **⚠️ `constrain` corrupts IgLM's loop endings — prefer `filter` for any reported comparison.**
> Preventing end-of-span before the native length means IgLM cannot stop where it would naturally
> stop, so it keeps generating into FR4 and those framework residues are counted as loop residues.
> Measured on the 25-framework smoke run: only **2%** of constrained loops end in the J-gene motif
> (`…DV`) against **20%** under natural EOS and 20% in the natives, and only **14%** reproduce the
> native's final two residues against **76%** naturally. This drops IgLM's anchor AAR to 0.552 from
> its natural 0.826 and is why it scores *worse* given the native length (0.2155 whole-span) than
> choosing freely (0.3532). `constrain` remains useful as a mechanism check; `filter` is the fair
> protocol for a fixed-length comparison.

```bash
conda run -n iglm python benchmark/adapters/iglm.py \
  --manifest benchmark/manifests/oas_h3_validation_smoke_v1.json \
  --output results/benchmark/iglm_validation_smoke.json \
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
adapter failures — "length-matched" here means the constraint was applied successfully, not that the
draws were naturally that length; see the warning above. Results live under ignored `results/benchmark/`; they are local run artifacts, not
frozen benchmark inputs.

### Matched-population scoring

`filter` retains only naturally length-matched draws, so a framework the baseline never matches
contributes no samples and therefore enters no macro mean — while the other run still scores it.
Two runs can then appear matched (same manifest, same IDs) while their averages cover different
frameworks. `scorer.py` warns when this happens; `assert_comparable` compares the *scored*
populations, not just the ID sets.

For a reportable comparison, restrict both runs to the frameworks every run scored and re-aggregate.
`results/benchmark/oas_h3_validation_smoke_v1/scores/filter_matched.json` is that artifact for the
2026-08-21 fixed-length re-run (20 of 25 frameworks; the 5 dropped are recorded under
`scoring.population.dropped`).

## CDRSeqGen adapter

Run CDRSeqGen in its own environment and point the adapter at a self-describing checkpoint directory:

```bash
conda run -n cdrprj python benchmark/adapters/cdrseqgen.py \
  --manifest benchmark/manifests/oas_h3_validation_smoke_v1.json \
  --checkpoint results/runs/20260811_181946_sweep_h3/h3_OAS_scratch_lr1e-3_wu0.06_wd0.01_s42 \
  --output results/benchmark/cdrseqgen_validation_smoke.json \
  --num-samples 20 \
  --decoding iterative \
  --temperature 1.0 \
  --top-p 0.9 \
  --seed 42
```

The adapter validates the target loop, source filename, and context regime against `run_config.json`;
recreates training-time tokenization, truncation, and native mask count through the same shared
generation core as `generate.py` and held-out evaluation; and hashes the weights, model config, and
run config. It supports both `independent` and `iterative` decoders. The decoder and sampling point
used above are development settings, not yet a frozen final comparison policy.

Each CDRSeqGen record also carries a structured `length_policy`. For this benchmark track it is
always fixed, selected from the manifest's native mask count; this makes the length information
given to CDRSeqGen explicit rather than leaving it implicit in the number of mask tokens.

The verified 25-framework smoke run produced 125/125 valid, native-length draws with zero failures.
Four frameworks exceeded this checkpoint's 128-token limit; training-compatible assembly trimmed
2–3 far-C-terminal `pt2` tokens, and each record reports that explicitly in `prompt`.


## Length-free track

Use the dedicated validation manifest so both adapters are scored under the same no-native-length
contract. A new CDRSeqGen checkpoint already contains the required length artifacts; migrate an older
checkpoint once before sampling:

```bash
conda run -n cdrprj python fit_mlm_length.py \
  --model_dir results/path/to/checkpoint
```

The command discovers the checkpoint's saved `run_config.json`, so the length head uses the same
processed dataset and split as the sequence run. Pass `--config` only for a legacy checkpoint that
does not contain a saved run configuration.

Before benchmarking, inspect `length_config.json`. Compare `eval_accuracy` and `eval_length_mae`
against the recorded `eval_mode_baseline_accuracy` and `eval_mode_baseline_length_mae`; this prevents
a mode-only length guesser from being mistaken for a useful learned predictor.

The first tuned-scratch fit (2026-08-17) did **not** pass that check: 9.24% exact accuracy / 3.64 MAE
versus 10.52% / 3.28 for the mode baseline. Its CDRSeqGen adapter output is nevertheless retained as
an honest development result: 25/25 frameworks complete, 500/500 valid candidates, zero failures,
and 30/500 (6.0%) native-length matches. The matched IgLM output and joint score are still pending.

Generate CDRSeqGen candidates from its learned length distribution:

```bash
conda run -n cdrprj python benchmark/adapters/cdrseqgen.py \
  --manifest benchmark/manifests/oas_h3_validation_length_free_v1.json \
  --checkpoint results/path/to/checkpoint \
  --output results/benchmark/oas_h3_validation_length_free_v1/generations/cdrseqgen.json \
  --num-samples 20 --decoding iterative \
  --temperature 1.0 --length-temperature 1.0 --top-p 0.9 --seed 42
```

Generate IgLM candidates with its natural EOS stopping rule:

```bash
conda run -n iglm python benchmark/adapters/iglm.py \
  --manifest benchmark/manifests/oas_h3_validation_length_free_v1.json \
  --output results/benchmark/oas_h3_validation_length_free_v1/generations/iglm.json \
  --num-samples 20 --max-attempts 25 --length-policy natural \
  --temperature 1.0 --top-p 0.9 --seed 42
```

Then score both outputs together:

```bash
conda run -n cdrprj python benchmark/scorer.py \
  --manifest benchmark/manifests/oas_h3_validation_length_free_v1.json \
  --input cdrseqgen=results/benchmark/oas_h3_validation_length_free_v1/generations/cdrseqgen.json \
  --input iglm=results/benchmark/oas_h3_validation_length_free_v1/generations/iglm.json \
  --output results/benchmark/oas_h3_validation_length_free_v1/scores/length_free.json \
  --judge Exscientia/IgBert --judge-max-len 128
```

Render the track-aware validation dashboard (length match/error and edit similarity replace the
fixed-length-only panels automatically):

```bash
conda run -n cdrprj python benchmark/plot_comparison.py \
  --scores results/benchmark/oas_h3_validation_length_free_v1/scores/length_free.json \
  --output results/benchmark/validation_length_free_comparison_v1.png
```

No length-free smoke result is claimed until these commands have been run. Keep its score artifact
separate from the fixed-native comparison: the recovery denominators and scientific question differ.

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

For the length-free track, retain every syntactically valid sample and preserve its generated length.
The native sequence and length remain scoring labels; neither CDRSeqGen's length head nor IgLM's
natural EOS generation receives the native mask count.

## Shared scorer

Score both adapter outputs in one invocation so the scorer can verify that they use the same
manifest records, requested sample count, temperature, `top_p`, and base seed:

```bash
conda run -n cdrprj python benchmark/scorer.py \
  --manifest benchmark/manifests/oas_h3_validation_smoke_v1.json \
  --input cdrseqgen=results/benchmark/cdrseqgen_validation_smoke.json \
  --input iglm=results/benchmark/iglm_validation_smoke.json \
  --output results/benchmark/validation_smoke_scores.json \
  --judge Exscientia/IgBert \
  --judge-max-len 128
```

The scorer validates every record against the manifest. Whole-span, two-residue anchor-trimmed core,
anchor-only AAR, position recovery, and native-distance novelty use only candidates whose generated
length matches the native. Every valid length contributes to length-match rate, length MAE,
alignment-aware edit similarity, duplicate-preserving Hamming diversity, unique fraction, and the
optional independent judge pseudo-perplexity. Per-position entropy is reported only when the retained
set has one common length. The scorer also reports short/medium/long native-H3 buckets.

Aggregate metric means are macro-averaged over frameworks, making the framework the statistical
unit. Attempt accounting is also reported separately: `valid_generation_rate` uses all attempts,
`length_match_rate` uses syntactically valid draws, and `fixed_length_acceptance_rate` uses all
attempts. Runtime per retained candidate covers generation only, not judge scoring. Partial adapter
outputs are rejected by default; `--allow-partial` exists only for development diagnostics, and two
inputs must still contain the same framework IDs.

**A crashed example is refused, not averaged around.** If generation throws, the CDRSeqGen adapter
records the example with `generation_failed: true` and `attempts: 0` rather than aborting the run.
Such a record is internally consistent (`0 == 0 + 0`), so it would otherwise pass validation, drop
out of every macro-average, and still be counted in `micro_accounting.examples` — a run where half
the examples died would score and plot as if it were complete. The scorer therefore refuses any
input where more than 5% of examples have zero attempts; `--allow-dead-records` overrides it for
diagnostics, and `micro_accounting.examples_without_attempts` always reports the count. The adapter
prints the same total in its `summary.records_with_generation_failure` and on stderr.

The verified five-draw, 25-framework validation smoke comparison produced the following development
numbers with the same IgBert revision for both runs:

| adapter | whole AAR | core AAR | anchor AAR | diversity | entropy (bits) | IgBert ppl |
|---|---:|---:|---:|---:|---:|---:|
| CDRSeqGen | 0.332 | 0.149 | 0.900 | 0.603 | 1.286 | 10.085 |
| IgLM | 0.216 | 0.108 | 0.552 | 0.762 | 1.620 | 7.001 |

These are scorer smoke results, not benchmark claims: the sample is a small validation slice, the
decoder policy is not frozen, and IgLM's OAS pretraining creates an unresolved overlap caveat. The
final test manifest remains unused.

## Comparison figure

Render the matched score artifact as a four-panel recovery, length, position, and
recovery--diversity dashboard:

```bash
conda run -n cdrprj python benchmark/plot_comparison.py \
  --scores results/benchmark/validation_smoke_scores_igbert_v1.json \
  --output results/benchmark/validation_smoke_comparison_v1.png
```

The plotter consumes only shared-scorer output; it never scores adapter records independently.
Validation or smoke manifests are automatically labeled as development figures rather than final
benchmark claims. The normalized-position curve averages residues within each framework before
averaging across frameworks, so longer H3 loops do not silently receive more statistical weight.
