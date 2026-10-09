# OAS4 empirical baseline and paired analysis — audited complete 2026-10-08

Completed at **13:37 Singapore time**, with [artifact audit passed](../../results/reviews/20261008-benchmark-followup.md).
The original seven rows are unchanged. All 3,000 new empirical draws trace to training records;
the combined eight-arm comparison contains **24,000 valid/judge-scored candidates**. Empirical edit
similarity is **0.2817**. All seven learned-system-minus-empirical family intervals are positive;
all six local-checkpoint-minus-IgLM intervals span zero. This is exploratory fixed-pool evidence,
not equivalence or an isolated architecture/conditioning effect.

Recovered **145,172** groups reproduce all **152,727** split labels. The 150 evaluated frameworks
occupy 150 distinct groups, so group resampling reduces to paired framework resampling here.
Capacity/actual training-exposure export is complete. No completion rerun is needed; commands
below describe reproduction and require a fresh output directory. Final leakage decisions and
held-out evaluation remain pending.

This follows the completed seven-arm length-free validation benchmark. All existing neural
candidates, judge scores and trained checkpoints are read-only. The full follow-up adds one
empirical reference: **150 frameworks × 20 attempts = 3,000 new candidates**, then judges those
candidates with the same cached IgBert revision. No neural training or neural generation is repeated.

## What runs

1. **CPU: reconstruct and verify H3 groups.** Recover the original ordered, same-length, 85%-identity
   representative clustering. The optimized implementation preserves first-matching representative
   behavior. Reproduce every train/validation/test split label using the recorded fractions/seed;
   stop if any label differs. This reads the corpus to audit the original partition, including its
   reserved test records; no test generations or performance estimates are made. Representative
   groups do not guarantee that all biological relatives are grouped or separated.
2. **CPU: sample an empirical repertoire.** Uniform sampling with replacement over training paired
   records' H3 loops; duplicated H3s retain their observed frequency. The sampler takes neither
   native sequence/length nor flanks. Source indices are recorded for every draw. This unconditional
   baseline measures performance relative to simple repertoire sampling; it is not nearest-neighbour
   retrieval or a framework-conditioning ablation. It has no logits, temperature or top-p operation.
3. **GPU: judge only the new draws.** Heavy-only IgBert, cap 384, the original cached revision and
   tokenizer. Cached revision/ref/files are frozen by the plan. Append the new scored row only after
   checking common manifest, policy, budgets, native populations and judge identity. Original seven
   score rows are copied unchanged; the added row does not create new training replicates.
4. **CPU: paired uncertainty.** Six local checkpoint-versus-IgLM contrasts and seven systems-versus-
   empirical contrasts: 13 primary edit comparisons. Resample whole verified H3 groups with their
   frameworks, retaining equal framework weighting. Report 20,000-resample 95% marginal intervals
   and Bonferroni family intervals for primary contrasts; supporting metrics have exploratory
   marginal intervals. Conditional metrics use only the shared eligible population and report its
   size. Candidate pools/trained models stay fixed: this does not estimate training or generation
   variability. The comparisons were selected after seeing validation means and remain exploratory.
5. **CPU: capacity/compute.** Count learned tensors from scratch checkpoint headers and separate MLM
   length heads. Count released GPT-2 parameters from configuration, counting tied embeddings once.
   Recover actual stopping/selected epochs, optimizer steps and recorded Trainer runtime. Do not
   use planned-budget throughput to infer actual exposure. Unrecorded historical GPU hardware and
   external pretraining compute remain unknown; sequence training time excludes MLM length fitting
   and all generative evaluation.

Outputs: `plan.json`, `progress.json`, `clusters.json`, `generations/empirical.json`,
`scores/{empirical,comparison}.json`, `capacity.json`, `analysis.{json,md,png}`, `complete.json`.
The combined table has eight arms and 24,000 candidates when every added draw is scored.
Cluster reconstruction can take longer than the other CPU stages; progress prints by H3 length.

## Commands for the user

Run inside your existing `screen` session:

```bash
cd /mnt/hdd/pang/CDRSeqGen
conda activate cdrprj
./scripts/run.sh benchmark followup plan
./scripts/run.sh benchmark followup run --gpu 0 \
  --output results/benchmark/oas4_h3_followup_20261008
```

The plan reads hashes and metadata without running clustering, sampling, bootstrap or models.
The run executes the stages above. Completion checks and paper imports follow artifact review.

```bash
./scripts/run.sh benchmark followup status results/benchmark/oas4_h3_followup_20261008
# If interrupted, preserve the folder and use:
./scripts/run.sh benchmark followup resume results/benchmark/oas4_h3_followup_20261008 --gpu 0
```

Source, environment, inputs and completed-stage hashes are checked on resume. Keep Python sources
unchanged during the batch. Existing outputs require `resume`; a fresh run refuses to overwrite.
New follow-up code has a new source fingerprint; the completed original benchmark is preserved
with its archived source and should not be resumed under this new source version.

## Optional integration smoke

The smoke adds five attempts on the five-framework original smoke subset: 25 new candidates,
then uses the four existing smoke arms. It still reconstructs/verifies the full-corpus groups.
Its intervals are diagnostic and cannot support a performance claim.

```bash
./scripts/run.sh benchmark followup plan --stage smoke
./scripts/run.sh benchmark followup smoke --gpu 0 \
  --output results/benchmark/oas4_h3_followup_smoke_20261008
```

After a completed smoke, avoid repeating cluster recovery by supplying its verified group file
to a new full run (verify the plan with the same `--clusters` argument):

```bash
./scripts/run.sh benchmark followup run --gpu 0 \
  --clusters results/benchmark/oas4_h3_followup_smoke_20261008/clusters.json \
  --output results/benchmark/oas4_h3_followup_20261008
```

## Paper placement

Methods defines empirical sampling, group verification, paired uncertainty and budget accounting.
Results §3.1 includes the empirical row and primary difference intervals; detailed paired populations
and compute rows are supporting tables. The source user-run intervals are imported without rerunning
bootstrap. Final held-out evaluation and pretraining-overlap decisions remain separate work.
