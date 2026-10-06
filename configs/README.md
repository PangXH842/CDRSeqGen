# Choosing a configuration

Base configs are grouped by infilling method and, for MLM, target/context:

| Folder | Contents |
| --- | --- |
| `mlm/heavy/` | H1/H2/H3, scratch or pretrained, optionally with light-chain context; includes the H3 weighted-loss example |
| `mlm/light/` | L1/L2/L3, scratch or ESM-2, optionally with heavy-chain context |
| `mlm/antigen/` | SAbDab2 heavy loops with antigen and/or partner-chain context |
| `ar/` | Autoregressive H1/H2/H3 scratch models |
| `s2s/` | Encoder–decoder H3, scratch or ESM-2/IgBert/ProtBert encoders |
| `experiments/` | Dated multi-arm specifications that reference base configs and override selected settings |

`h1`–`h3` and `l1`–`l3` identify the target loop. `_lc` or `_hc` adds the
partner chain. `_ag` uses antigen plus partner context; `_ag_lconly` is the
partner-only control on the same SAbDab2 corpus. Check the JSON for exact settings.

Run commands from the repository root:

```bash
python train.py configs/mlm/heavy/h3_scratch.json
python train.py configs/ar/h3_ar_scratch.json
python train.py configs/s2s/h3_s2s_esm2.json
DRY_RUN=1 ./scripts/run.sh 0 compare
DRY_RUN=1 ./scripts/run.sh 0 sweep lr configs/s2s/h3_s2s_scratch.json
DRY_RUN=1 ./scripts/run.sh 0 spec configs/experiments/20260915_context_dropout.json
```

Suite names are unchanged. Config and data paths inside experiment specs are relative
to the repository root, not the enclosing config folder. Historical run configs and
plans under `results/` retain their original paths; use a fresh plan for new runs.

The next H3 batches are prepared in
[`20261006_no_partner_controls.json`](experiments/20261006_no_partner_controls.json)
(six trials), followed by
[`20261006_esm2_dropout_confirmation.json`](experiments/20261006_esm2_dropout_confirmation.json)
(four trials). See the [fixed protocol and launch commands](experiments/20261006_h3_followup.md).
The controls batch was launched on 2026-10-06; its initial TensorBoard startup
failure is corrected, with retry instructions in the recipe. Confirmation remains pending.
