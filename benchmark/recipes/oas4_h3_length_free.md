# OAS4 length-free validation benchmark — prepared 2026-10-07

**Next experiment: inference and shared scoring only.** Reuse six completed scratch sequence
checkpoints (MLM/AR/S2S × training seeds 42/43) and both existing MLM length heads. IgLM is a frozen
external baseline, not another local training seed. ESM-2 arms and new training sweeps are extensions.

The full [manifest](../manifests/oas4_h3_validation_length_free_v1.json) reuses the exact 150 validation
frameworks from the audited context study, sorted by stable example ID. Its five-framework
[smoke subset](../manifests/oas4_h3_validation_length_free_smoke_v1.json) spans the observed native
lengths (6, 13, 16, 19, 26); those lengths choose the diagnostic subset and support scoring only.
The processed corpus SHA is `bd854ae32de886bc9b4466edf5d1aa3ade6c09847b622ceaa60cba395bbd5ad8`.
The reserved test split is not used.

| Setting | Policy |
|---|---|
| Generator input | Heavy-chain flanks; no partner or antigen; native length unavailable for generation |
| MLM | Independent residues, checkpoint-specific inferred length head, length temperature 1 |
| AR/S2S | Existing prompt builders/samplers; natural EOS within 97 new tokens |
| IgLM | Human/heavy conditioning, natural EOS, 97 new tokens; absolute total cap 256 |
| Draws | Temperature 1, top-p 0.9, top-k 0 where supported; stable per-example generation seed from 42 |
| Attempt budget | Exactly 20 requested attempts per framework/arm; failed draws are not replenished |
| Judge | IgBert, heavy-chain context only, cap 384; supporting endpoint |
| Primary endpoint | Native edit similarity: candidate mean within framework, then framework mean |
| Supporting | Length error/match, validity, diversity, uniqueness, conditional core/anchor recovery and populations |
| Uncertainty | Descriptive validation results; no significance/equivalence or cluster-aware interval inferred |

Native sequence is retained for scoring and for locating the gap in IgLM's parent string; the masked
IgLM prompt removes it completely. MLM's supported length classes come from training. Different
methods' intrinsic length mechanisms remain visible. Every selected framework fits its generation
prompt without AR/S2S flank truncation; IgLM's longest prompt plus decode budget is 211 tokens.

Saved training source/runtime versions and dirty-checkout provenance are retained. New adapter
outputs hash inference source, weights/configs and sampling settings. The batch freezes recipe,
manifest, checkpoint/head hashes and both environments; resume rejects drift. This is a system
demonstration, not an isolated architecture or pretraining effect. OAS overlap with IgLM/IgBert
pretraining remains unresolved. Older partner-present judge scores are not reused for this new
heavy-only judge policy.

## Run the smoke first

Preparation passed 52 CPU contract tests and real scratch-tokenizer/prompt checks on all 150
frameworks. Both plans validate without loading weights. A live GPU smoke has not been run by
the assistant.

```bash
cd /mnt/hdd/pang/CDRSeqGen
conda activate cdrprj
./scripts/run.sh benchmark plan --stage smoke
./scripts/run.sh benchmark smoke --gpu 0 \
  --output results/benchmark/oas4_h3_length_free_smoke_20261007
./scripts/run.sh benchmark status results/benchmark/oas4_h3_length_free_smoke_20261007
```

This samples MLM/AR/S2S seed 42 plus IgLM: five frameworks × five attempts × four arms = 100
requested attempts. It then runs the shared judge/scorer and writes a comparison image. Outputs:
`plan.json`, `generations/{mlm_s42,ar_s42,s2s_s42,iglm}.json`, `scores/comparison.json`, `comparison.png`.

Check all four arms cover the same five IDs, sample/attempt accounting is consistent, and no
generation exceptions or unexplained missing judge scores occur. Retain and report genuinely
invalid draws; their existence is not a reason to replenish a model's candidate pool. Inspect
conditional-recovery populations separately. The tiny subset is an integration check, not a
performance estimate.

## Full validation batch after the smoke checks

```bash
./scripts/run.sh benchmark plan --stage full
./scripts/run.sh benchmark run --gpu 0 \
  --output results/benchmark/oas4_h3_length_free_20261007
./scripts/run.sh benchmark status results/benchmark/oas4_h3_length_free_20261007
```

Seven arms, 150 frameworks, 20 attempts = 21,000 requested attempts. Keep per-training-seed rows;
the baseline is one fixed model/candidate pool, rather than two independent training replicates.
After completion, audit framework/metric coverage and failure accounting before adding numerical
results to the manuscript. Prepare cluster-aware paired analysis separately; the historical
two-run framework bootstrap is not automatically a cluster-aware seven-arm analysis.

If interrupted, preserve the directory and code/environment, then use:

```bash
./scripts/run.sh benchmark resume results/benchmark/oas4_h3_length_free_smoke_20261007 --gpu 0
# Or resume the full batch using its own directory.
```

Existing files require `resume`; a fresh launch refuses to overwrite a batch. Review Python code
without editing it during the batch, because source changes invalidate resume. No checkpoint
weights or original training outputs are modified.
