# CDR-H3 Benchmark Harness

Cross-model evaluation layer for CDRSeqGen:

```text
frozen manifest (manifest.py)
  → adapters (run.py → adapters/{cdrseqgen, iglm, igcraft, retrieval}.py)
  → shared scoring (scorer.py)
  → post-hoc analysis & plots (analysis/)
```

All adapters consume the same frozen, model-independent manifest of antibody strings and emit the same candidate-record JSON schema. Scoring, judging, uncertainty intervals, and plotting operate only on that shared schema, keeping model code and environments isolated.

---

## Layout

| Layer | Path | Responsibility |
|---|---|---|
| **Core** | [`_common.py`](_common.py) | Provenance hashing, resume-safe seeding, atomic JSON writes, manifest/resume checks |
| | [`manifest.py`](manifest.py) | Export frozen CDR-H3 manifests (`export`) and IMGT flank-region sidecars (`regions`) |
| | [`run.py`](run.py) | `plan`, `smoke`, `run`, `resume`, `status`, and `recipe` (build arm recipes from finished training batches) |
| | [`scorer.py`](scorer.py) | Shared validation, metric computation, attempt accounting, and IgBert judge scoring |
| | [`blind_check.py`](blind_check.py) | Information-budget test verifying length-free adapters ignore hidden native fields |
| **Adapters** | [`adapters/README.md`](adapters/README.md) | Adapter contract, information budget, and registration guide |
| | [`adapters/registry.py`](adapters/registry.py) | External generator registry (`iglm`, `igcraft`, `retrieval`) |
| | [`adapters/cdrseqgen.py`](adapters/cdrseqgen.py), [`iglm.py`](adapters/iglm.py), [`igcraft.py`](adapters/igcraft.py), [`retrieval.py`](adapters/retrieval.py) | Model- and reference-specific candidate producers |
| **Analysis** | [`analysis/stats.py`](analysis/stats.py) | Paired H3-group and crossed-seed bootstrap CIs, parameter/exposure extraction, and 2-arm bootstrap CLI |
| | [`analysis/empirical.py`](analysis/empirical.py) | Training-only H3 repertoire sampler and verified representative H3 clustering |
| | [`analysis/followup.py`](analysis/followup.py) | Empirical baseline, verified-group paired contrasts, and capacity export |
| | [`analysis/extend.py`](analysis/extend.py) | Merge newly scored external batches into a finished comparison |
| | [`analysis/variability.py`](analysis/variability.py) | Seven-seed spread, same-seed determinism, capacity scan, sensitivity, and core-endpoint analysis |
| | [`analysis/leakage.py`](analysis/leakage.py) | Local same-length split-overlap and holdout-eligibility audit (CPU-only) |
| | [`analysis/plot_comparison.py`](analysis/plot_comparison.py) | Four-panel matched recovery, length, position, and diversity dashboard |
| **Inputs & notes** | `manifests/` | Frozen manifests + `.sha256` sidecars: the active OAS4 ones and the earlier one-donor pilot manifests (`oas_h3_*_v1`; the paper's evidence exporter reads the validation one by this path) |
| | `recipes/` | Matched benchmark recipes, protocols, and `generated/` arm recipes |
| | [`notes/baseline_scope.md`](notes/baseline_scope.md), [`notes/pretraining_overlap.md`](notes/pretraining_overlap.md) | External baseline coverage and pretraining-overlap claim boundaries |

---

## Common commands

All batch commands are exposed via `./scripts/run.sh benchmark ...` (which forwards through `python -m experiments benchmark`):

```bash
# Plan / run / inspect the main matched benchmark (benchmark/run.py)
./scripts/run.sh benchmark plan --recipe benchmark/recipes/oas4_h3_length_free.json --stage full
./scripts/run.sh benchmark smoke --gpu 0
./scripts/run.sh benchmark run --gpu 0
./scripts/run.sh benchmark status results/benchmark/oas4_h3_length_free_20261007
./scripts/run.sh benchmark resume results/benchmark/BATCH --gpu 0

# Build a recipe from completed training batches
python benchmark/run.py recipe --name oas4_h3_seeds --batch results/runs/BATCH \
    --output benchmark/recipes/generated/oas4_h3_seeds.json

# Export a manifest or its IMGT flank-region sidecar
python benchmark/manifest.py export --input data/processed/processed_oas_paired.json \
    --output benchmark/manifests/NEW.json --split validation --track length_free --sample-size 150
python benchmark/manifest.py regions --manifest benchmark/manifests/NEW.json \
    --corpus data/processed/processed_oas_paired.json --output benchmark/manifests/NEW_regions.json

# Post-hoc analysis tools (benchmark/analysis/)
./scripts/run.sh benchmark followup status results/benchmark/oas4_h3_followup_20261008
./scripts/run.sh benchmark extend plan --base results/benchmark/oas4_h3_followup_20261008 --added results/benchmark/BATCH
./scripts/run.sh benchmark variability plan
./scripts/run.sh benchmark leakage status results/benchmark/oas4_h3_leakage_20261008

# Standalone scoring, plotting, and adapter blind check
python benchmark/scorer.py --manifest M.json --input arm=out.json --output scores.json --judge Exscientia/IgBert --judge-max-len 128
python benchmark/analysis/plot_comparison.py --scores scores.json --output comparison.png
python benchmark/blind_check.py --manifest M.json --plan results/benchmark/BATCH/plan.json --label mlm_s42
```

Plans validate inputs, hashes, and environments without loading models. `smoke`, `run`, and `resume` execute models and remain user-run.

---

## Completed OAS4 validation evidence

- **Core matched benchmark** ([recipe](recipes/oas4_h3_length_free.md), [audit](../results/reviews/20261008-benchmark-full.md)): 150 validation frameworks × 20 attempts across 6 scratch MLM/AR/S2S checkpoints (seeds 42/43) and frozen IgLM — 21,000/21,000 valid, heavy-only IgBert-judged candidates.
- **Follow-up (empirical baseline + paired CIs + capacity)** ([recipe](recipes/oas4_h3_followup.md), [audit](../results/reviews/20261008-benchmark-followup.md)): adds 3,000 training-only empirical H3 draws (24,000 total), verified H3 split clusters, 13 primary paired contrasts, and actual capacity/exposure.
- **External extensions:** IgCraft ([recipe](recipes/oas4_h3_igcraft.md)) and flank-keyed retrieval references ([recipe](recipes/oas4_h3_retrieval.md)) are complete and audited.
- **Variability & capacity:** [`analysis/variability.py`](analysis/variability.py) ([protocol](recipes/oas4_h3_variability.md)) aggregates the seven-seed runs, same-seed repeats, capacity scan, and IgCraft temperature sensitivity.
- **Local split-overlap audit** ([recipe](recipes/oas4_h3_leakage.md), [audit](../results/reviews/20261008-leakage-audit.md)): CPU-only exhaustive same-length H3 search flags 3/150 validation benchmark frameworks and retains 14,818/15,273 test records (97.02%). External pretraining exposure remains unverified ([`notes/pretraining_overlap.md`](notes/pretraining_overlap.md)).

---

## Scoring & adapter contract notes

- **Information budget:** On the primary `length_free` track, adapters receive `pt1`, `pt2`, and declared chain/species conditioning only — never `native`, `native_length`, or corpus lookups. [`blind_check.py`](blind_check.py) verifies this empirically.
- **Attempt accounting:** Every requested attempt is accounted for (`attempts == valid_attempts + invalid_attempts`, no replenishment of invalid draws). Runs where more than 5% of frameworks crash (`attempts == 0`) are refused unless `--allow-dead-records` is passed.
- **Metric populations:** Edit similarity, length MAE/match, diversity, unique fraction, and IgBert judge pseudo-perplexity cover all valid retained candidates; positional AAR (whole, 2-residue anchor-trimmed core, anchor) covers length-matched candidates only. Macro means weight each contributing framework equally.
- **Historical fixed-length note:** Earlier one-donor smoke manifests (`oas_h3_*_v1.json`) stay in `manifests/`. On that fixed-length track, IgLM `--length-policy constrain` corrupted loop endings by forcing generation into FR4; `--length-policy filter` exposed matched-population bias instead. Length-free generation is the primary track going forward.
