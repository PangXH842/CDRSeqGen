# OAS4 length-free validation: flank-keyed retrieval references

**Prepared 12 October 2026; the judge-scoring stage is user-run and pending.** The generation step is CPU-only and has
been rehearsed (see Verified before running).

The unconditional empirical sampler ignores the framework, so "learned systems beat empirical sampling" bundles two
things: what the flanks already tell you about the loop, and what a model adds. These reference arms separate them. Each
framework gets 20 H3 loops drawn, with replacement, from the **training** records whose own flanks share a key with it.
They have no parameters, no sampling controls and no length input, and they read only the flank strings `pt1` and `pt2`.

| Label | Key | Role |
|---|---|---|
| `retrieval` | last 12 residues of FR3 + first 6 of FR4 | **Primary reference.** The V/J proxy the README documents for its length lookup, fixed before any key was scored |
| `retrieval_j` | the whole of FR4 | Sensitivity: a J-gene proxy |
| `retrieval_fine` | last 20 residues of FR3 + the whole of FR4 | Sensitivity: a finer V/J proxy |

A framework whose key never occurs in training uses the unconditional training pool (3, 1 and 12 of 150 frameworks
for the three keys), and each record states which pool it used. A fourth key looked at during a feasibility check (the
FR3 tail alone) scored 0.293, barely above unconditional, and is not registered.

The primary joins the contrast family as a **reference pool** (compared with every system, and itself compared with
the empirical pool). The two sensitivity arms are merged and scored but kept out of the family, so they widen no
Bonferroni correction. With the IgCraft extension as the base, the family grows from 20 to 29 contrasts.

## Run it

From `/mnt/hdd/pang/CDRSeqGen`. Generation takes seconds; the judge scoring of 3 x 3,000 candidates takes about 25 minutes
on GPU 0 and is the only GPU work.

```bash
cd /mnt/hdd/pang/CDRSeqGen
RECIPE=benchmark/recipes/oas4_h3_retrieval.json
BATCH=results/benchmark/oas4_h3_retrieval_20261012
BASE=results/benchmark/20261009_183032_oas4_h3_igcraft_20261009_extension

./scripts/run.sh benchmark plan --recipe $RECIPE --stage full
./scripts/run.sh benchmark run  --recipe $RECIPE --gpu 0 --output $BATCH 2>&1 | tee $BATCH.log
./scripts/run.sh benchmark status $BATCH
```

Expect `retrieval: 150/150 frameworks; finished=True; returned=3000; invalid=0` for each label. Then the information-budget
test (CPU, about a minute per arm) and the merge:

```bash
for label in retrieval retrieval_j retrieval_fine; do
  python benchmark/blind_check.py --manifest benchmark/manifests/oas4_h3_validation_length_free_smoke_v1.json --limit 4 \
    --workdir results/benchmark/blind_checks/$label --plan $BATCH/plan.json --label $label --num-samples 5
done

./scripts/run.sh benchmark extend plan --base $BASE --added $BATCH --reference retrieval --exclude retrieval_j --exclude retrieval_fine
./scripts/run.sh benchmark extend run  --base $BASE --added $BATCH --reference retrieval --exclude retrieval_j --exclude retrieval_fine
```

The extension prints the 29-row table and writes a new directory; the IgCraft extension it builds on is not modified.

## Verified before running

- The registered adapter reproduces the feasibility check's definitions; its edit similarities are **0.3589 / 0.3469 /
  0.3653** (primary / J / fine), against 0.3552 / 0.3494 / 0.3663 in the check. The difference is only the random stream:
  with 20 draws per framework the Monte-Carlo noise of a reference mean is about +-0.002 to 0.004, which the fixed-pool
  intervals do not include.
- Every draw comes from the training split; the output passes the scorer's validators and `blind_check` (all three keys).
- Rehearsal of the merge on the real files with stand-in judge scores (edit similarity does not depend on the judge):
  the chain resolved the earlier IgCraft extension, whose plan predates roles, back to the original H3 groups, and gave
  29 contrasts. Edit-similarity contrasts, to be reproduced by the official run:

| Contrast | Difference | 95% marginal | Family (29) |
|---|---:|---|---|
| retrieval - empirical | +0.0772 | [+0.0654, +0.0889] | [+0.0578, +0.0964] |
| scratch arms (six) - retrieval | +0.028 to +0.033 | all exclude 0 | all exclude 0 |
| IgLM - retrieval | +0.0291 | [+0.0222, +0.0366] | [+0.0188, +0.0423] |
| IgCraft - retrieval | +0.0180 | [+0.0070, +0.0293] | [+0.0010, +0.0358] |

## Reading the result

- About 70-80% of a learned system's edit-similarity gain over unconditional sampling is already available from the flanks
  (retrieval - empirical = +0.077 of the +0.105 to +0.110 the learned arms show). The learned margin over retrieval is about 0.03.
- The J-gene tail does most of this: FR4 alone reaches 0.347. On the core (two residues trimmed from each end) the learned
  margin is smaller still, and IgCraft sits at retrieval level; that comparison is outside this protocol's planned family.
- These are validation results with fixed candidate pools and checkpoints. They exclude training and generation
  variability, and say nothing about biological quality.
