# Pretraining overlap: provenance and claim boundary

**Reviewed 8 October 2026. Membership checks remain pending.** This note records
primary-source evidence; no external training corpus has been downloaded or
searched. It covers the current matched scratch MLM/AR/S2S + IgLM benchmark and
its IgBert judge. Historical pretrained encoder experiments need separate checks.

| Component | Role in the current benchmark | Training provenance | OAS4 membership status |
|---|---|---|---|
| Scratch MLM/AR/S2S, seeds 42/43 | Six local generators | Saved local no-partner training runs; no external pretrained weights | Allocation and declared same-length H3 overlap audit complete; final cohort selection pending |
| Empirical sampler | Unconditioned reference | Draws only from local training H3 records, preserving frequency | Training-only source indices recorded |
| Frozen IgLM | External generator | Antibody pretraining on OAS | Exact/near sequence exposure unverified |
| Exscientia/IgBert | Shared supporting judge | Protein initialization, unpaired OAS pretraining, paired OAS fine-tuning | Exact/near sequence exposure unverified |

IgLM's paper describes 558 million training sequences, both chain types and six
species, with 95% sequence clustering. Its official repository links training/test
data at `https://data.graylab.jhu.edu/OAS95.tar.gz`. Neither the paper nor the link
establishes whether our four-donor cohort belongs to the checkpoint's training
partition. Sources: [IgLM paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11018345/),
[official repository](https://github.com/Graylab/IgLM).

The IgBert model card identifies paired OAS fine-tuning from IgBert-unpaired. The
paper's data preparation describes 95% unpaired clustering and 95% clustering of
concatenated heavy/light sequences for paired data, with separate authors'
validation/test clusters. It reports 1,992,786 unique paired training sequences.
Our local H3 groups do not reproduce those authors' training partitions. Sources:
[official model card](https://huggingface.co/Exscientia/IgBert),
[paper, Sections 4 and 6](https://arxiv.org/pdf/2403.17889).

The frozen judge revision is
`81a4abb8a4dac5cdcbefc28e5b331a82224c485d`; the completed generation/score plans
record local IgLM weight hashes and implementation provenance. Preserve those
identities when obtaining membership evidence. A publication year alone does not
prove sequence absence, and sharing the OAS repository alone does not prove that
every local sequence was used in an external model's training partition.

## Evidence needed to close this question

Record the exact external training release, checksum, checkpoint relationship,
training/validation/test membership and sequence preprocessing. For the selected
local cohort, check full heavy sequences against external training records for
exact membership first. IgBert also needs paired/unpaired and general-protein
initialization exposure considered. Near-homology checks require a declared
alignment, coverage and identity criterion; absence of exact matches only settles
exact membership. A training archive containing only cluster representatives may
not expose all sequences seen by a model.

If membership data remain unavailable before the deadline, keep the comparison
as a **matched evaluation of available systems on a locally held-out cohort**, with
external pretraining exposure explicitly unknown. Native edit similarity remains
the primary endpoint; IgBert perplexity is a supporting model score. Unknown IgLM
exposure still limits generalization claims on that primary endpoint, and an OAS
judge score cannot establish independent biological quality. An independent cohort
or matched retraining is needed for a stronger external generalization claim.

The [local split audit](recipes/oas4_h3_leakage.md) completed and passed its review:
3/150 existing benchmark frameworks have close training H3s; the stated test rule
retains 14,818/15,273 records. It closes only the stated local overlap check. The final
protocol should record this external status separately rather than waiting for or
pretending to have exhaustive public pretraining membership.
