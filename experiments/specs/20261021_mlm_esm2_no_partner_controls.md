# MLM ESM-2 no-partner controls — 21 October 2026

**Completed and audited on 22 October 2026.** Both trials in `results/runs/20261021_155531_796405_mlm_esm2_no_partner_controls` passed the [completion audit](../../results/reviews/20261022-mlm-esm2-no-partner-controls.md), adding the two MLM ESM-2 rows to the no-partner control study (Table 8, `tab:controls`). No completion rerun is needed. This targeted framework demonstration does not block abstract drafting or replace final held-out testing. Training/generation remain user-run.

## Scientific purpose

Show that the same MLM workflow can fine-tune a pretrained masked protein language model and use its own fitted length head for length-free H3 generation. This complements the existing scratch MLM and pretrained-encoder S2S rows. It does not establish that MLM adapts to every PLM or that pretraining causes an improvement: the pretrained/random architectures, capacities and tuning budgets are not controlled by this two-run addition.

The MLM backbone already loads `EsmForMaskedLM`. ESM uses masked language modeling ([official model documentation](https://huggingface.co/docs/transformers/en/model_doc/esm)), so direct encoder/MLM-head reuse is natural. The S2S path reuses an encoder with a separate decoder; the AR path needs a suitable causal model. These are compatibility distinctions rather than a ranking of all protein language models.

## Recipe and fixed protocol

- Recipe: [20261021_mlm_esm2_no_partner_controls.json](20261021_mlm_esm2_no_partner_controls.json).
- Flat base: [bases/h3_mlm_esm2.json](bases/h3_mlm_esm2.json). Existing scratch/S2S bases and completed artifacts are unchanged.
- Model: `facebook/esm2_t12_35M_UR50D`, all sequence-model weights fine-tuned. Its pretrained dimensions are retained; no scratch width/layer overrides are carried over.
- Corpus: OAS4 `data/processed/processed_oas_multi.json`, validation selection; expected SHA-256 `bd854ae32de886bc9b4466edf5d1aa3ade6c09847b622ceaa60cba395bbd5ad8`.
- Training/data seeds 42 and 43; H3 full-span masking, cross-entropy, no partner or antigen, dropout 0. Heavy-chain flanks remain available.
- Maximum assembled length 384; train microbatch 4, accumulation 8 (effective batch 32 on one GPU); evaluation batch 4. Matches the effective batch and PLM optimizer recipe of Table 8's S2S ESM-2 controls.
- LR `3e-5`, 6% warmup, weight decay 0.01, existing linear scheduler; up to 20 epochs, patience 3, select best masked core accuracy. This is a declared initial setting, not a tuned optimum or proof of equal tuning effort.
- Fit a checkpoint-specific gap-sentinel length head on training only: 20 epochs, LR 0.01, batch 16. Native length is withheld during candidate generation.
- Same intended evaluation selection: 150 uniformly sampled validation frameworks, generation seed 42, 20 returned candidates per framework, inferred length and independent residue decoding; residue/length temperatures 1, top-p 0.9, top-k 0.
- **Table 8 judge policy:** IgBert receives partner context even though training/generation do not. Verify exact example membership and that policy against the saved October controls before combining rows. Do not copy these judge values into the heavy-only external benchmark table.
- Record parameter counts, chosen epochs, actual steps/runtime, model/tokenizer/length-head artifacts and source/environment provenance. Retain the distinction between MLM returned-candidate coverage and missing attempt-validity accounting.

The older OAS4 MLM ESM-2 seed-42 run at `results/runs/20260903_192305_mlm_heavy_ar_heavy/h3_OAS4_esm2_t12_35M_UR50D` used a 128-token limit, batch 16, LR `5e-5`, 10-epoch maximum and fixed-native-length evaluation. Preserve it as historical evidence; its scores cannot fill the new length-free control rows.

## Completed outputs and inspection

The runs return 6,000 nonempty canonical candidates with full judge coverage. Edit similarity is 0.3891/0.3883; length MAE is 4.035/4.236; both select epoch 5 and stop at epoch 8. MLM attempt validity remains unrecorded. Table 8 and the length figure are regenerated from these audited outputs.

```bash
./scripts/run.sh status results/runs/20261021_155531_796405_mlm_esm2_no_partner_controls
```

## Reproduction commands (user-run, new batch only)

The batch above is complete. Use the following only for an intentional new source-versioned reproduction, from the repository root inside your screen session:

```bash
conda activate cdrprj
./scripts/run.sh plan mlm_esm2_no_partner_controls --gpus 0
./scripts/run.sh run mlm_esm2_no_partner_controls --gpus 0
```

GPU 0 runs the two trials sequentially. Use `--gpus 0,1` for one trial per GPU only when both are available. Planning creates no training batch or model outputs. The run prints the new timestamped directory; inspect it afterwards with:

```bash
./scripts/run.sh status results/runs/<printed-batch-directory>
```

Fresh runs capture current source/runtime identity. Do not resume the historical controls or change shared bases mid-batch. If memory requires a smaller microbatch, adjust accumulation to preserve effective batch 32, record it in a fresh recipe/plan and keep that setting consistent across seeds.

## Completion and manuscript update

1. Require two successful trials, selected sequence-model weights, exact saved tokenizer, fitted head/config and one generation output per trial.
2. Verify example membership, length mode, generator/judge context, returned counts, judge coverage and per-framework metrics against Table 8's protocol. Audit actual checkpoint/head compatibility and provenance before quoting results.
3. Extend `iAIM/scripts/export_evidence.py` to discover these runs and regenerate the no-partner table from saved artifacts. Do not hand-enter numerical rows or mark the trials complete in advance. Rename the current October 6–7 caption/column notes to cover the new source/runtime and candidate counts.
4. If we add these checkpoints to the main matched external benchmark later, generate/score through its frozen manifest and heavy-only judge policy separately. That is not completed by Table 8 evaluation.

No extra dropout arms are included. Extending the robustness claim specifically to MLM ESM-2 would additionally require partner-trained dropout-0/0.5 models evaluated in both regimes; decide that only if it becomes a paper claim. A same-architecture randomly initialized ESM-2 control is separate again, needed for attributing an effect to pretraining.
