# OAS4 local split-overlap audit

**Completed 8 October 2026 at 14:58 Singapore time; artifact audit passed.**
All 180 full query chunks are sealed. The benchmark has 3/150 frameworks with a
training H3 neighbour at the cutoff; candidate test eligibility is 14,818/15,273
(97.02%), excluding 455. Every donor and represented test length retains records.
[Completion review](../../results/reviews/20261008-leakage-audit.md).
Existing eight-arm results are unchanged and need no completion rerun.

## Inspect completed results

```bash
./scripts/run.sh benchmark leakage status results/benchmark/oas4_h3_leakage_20261008
python results/reviews/20261008-leakage-audit.py
```

## Reproduce in a fresh output directory

The commands below reproduce the completed batch; substitute **new directory
names**, since these existing outputs are preserved and fresh runs refuse reuse.

```bash
cd /mnt/hdd/pang/CDRSeqGen
conda activate cdrprj
./scripts/run.sh benchmark leakage plan
./scripts/run.sh benchmark leakage smoke \
  --output results/benchmark/oas4_h3_leakage_smoke_20261008
./scripts/run.sh benchmark leakage run \
  --output results/benchmark/oas4_h3_leakage_20261008
```

Run the full command after the smoke succeeds. Both are **CPU-only** and need no
GPU, model downloads or retraining. The smoke audits five validation and five test
queries against the **complete reference partitions**. It checks integration;
its counts cannot support a split decision. Full runtime is unmeasured. Tiled
NumPy comparisons bound temporary search memory, but reading the 607 MB JSON and
building exact-sequence indices requires several GB of RAM. Only one audit should
write a given output directory. New runs refuse an existing output directory.

If interrupted, use the existing directory:

```bash
./scripts/run.sh benchmark leakage status results/benchmark/oas4_h3_leakage_20261008
./scripts/run.sh benchmark leakage resume results/benchmark/oas4_h3_leakage_20261008
```

Chunks contain 256 query records. Sealed chunks are verified and reused; an
interrupted unsealed chunk is recomputed. Partial resume requires the original
source, environment and input hashes. Status remains readable after later source
edits. Keep the prepared source unchanged while this batch is running.

## What gets checked

| Query partition | Complete reference partition | Full query count | Why |
|---|---|---:|---|
| Validation | Train | 15,273 | Development-set sequence separation; includes the existing 150-framework benchmark |
| Test | Train | 15,273 | Direct exposure to local training examples |
| Test | Validation | 15,273 | Exposure through checkpoint selection and tuning |

For every query, compare **every same-length reference H3**, then save its exact
closest positional identity, the closest source index, the number of neighbours
at **≥85%**, and the number of exact H3 matches. Ties choose the smallest source
index. Closest matches below 85% are still measured; this is an exhaustive search,
not a representative-only or approximate-neighbour check. With no same-length
reference, closest identity is undefined and the neighbour count is zero.

```text
length = 20, cutoff = 85% -> at least 17 identical positions

representative 1: AAAAAAAAAAAAAAAAAAAA
its member:      CCCAAAAAAAAAAAAAAAAA
representative 2:CCCCAAAAAAAAAAAAAAAA

rep 1 vs rep 2: 16/20 -> different representative groups
member vs rep 2:19/20 -> a close pair can cross those groups
```

The audit reuses the saved **145,172 groups** and verifies their allocation against
all **152,727 records**; it never reclusters or changes the corpus. It checks the
benchmark manifest's stable IDs, native H3s, flanks and global corpus indices.
The old README records a September finding of 3/150 near-neighbour H3s; its linked
report is missing locally. The completed audit independently verifies the same
3/150 count against the current frozen manifest, without estimating its effect
on the existing performance means.

Exact-overlap indices additionally cover all six target loops, reconstructed
heavy variable sequences, reconstructed heavy/light pairs, and the **two heavy
flanks with H3 removed**. Flanks include H1/H2, as supplied to the generators.
These definitions use the seven processed IMGT regions, not raw alignment gaps.
Empty framework regions are preserved. H3s must be canonical amino acids; any
ambiguous residues in other regions remain literal exact-string diagnostics and
are counted by region in `verification.json`, rather than silently normalized.
Shared germline context can be common and is not automatically H3 target leakage.
Non-H3 results are diagnostics; there is no non-H3 homology separation guarantee.

## Outputs and decision

The output directory contains frozen `plan.json`, sealed per-query files under
`comparisons/`, `verification.json`, donor/length summaries in `summary.json`, a
readable `summary.md`, and a hashed `complete.json` marker. Reference counts are
record counts, so repeated H3s retain their frequency.

`candidate_test` lists original test source indices with **no ≥85% same-length H3
neighbour in either train or validation**. This prespecified rule uses sequence
relationships, with no model-outcome filtering. The list is a candidate cohort,
not a final manifest, and smoke eligibility is incomplete. Before final inference:

1. Audit the full outputs and report overlap in the existing validation cohort.
   Keep its original results and denominators; any restricted reanalysis is
   explicitly a sensitivity analysis.
2. Inspect how much of the original test partition passes the stated rule and
   whether exclusions change donor/length coverage. Choose a common cohort and
   record its original/retained counts before evaluating any model on it.
3. If that cohort is inadequate, revise the split and retrain the affected local
   models. Existing checkpoints cannot be relabelled as trained on a new split.
4. Resolve the external-corpus claim using the separate
   [pretraining provenance note](../pretraining_overlap.md). Local H3 separation
   cannot certify IgLM/IgBert pretraining independence.

Different lengths, gapped alignments, full-variable-domain homology and biological
clonal families are outside this check. Report the operational criterion exactly;
do not label passing records universally homology-free.

For the paper, measured local overlap belongs in **Methods: data/split protocol**
and an **appendix split-audit table**. A chosen final cohort and its denominator
belong beside the final comparison; external exposure remains a Methods/Discussion
limitation until supported by membership evidence. No final test scores exist yet.
