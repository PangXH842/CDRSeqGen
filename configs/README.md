# Choosing a configuration

Base configs are grouped by infilling method and, for MLM, target/context:

| Folder | Contents |
| --- | --- |
| `mlm/heavy/` | H1/H2/H3, scratch or pretrained, optionally with light-chain context; includes the H3 weighted-loss example |
| `mlm/light/` | L1/L2/L3, scratch or ESM-2, optionally with heavy-chain context |
| `mlm/antigen/` | SAbDab2 heavy loops with antigen and/or partner-chain context |
| `ar/` | Autoregressive H1/H2/H3 scratch models |
| `s2s/` | Encoder–decoder H3, scratch or ESM-2/IgBert/ProtBert encoders |
| `experiments/` | Named multi-arm recipes; [guide](experiments/README.md), shared settings under `bases/` |

`h1`–`h3` and `l1`–`l3` identify the target loop. `_lc` or `_hc` adds the
partner chain. `_ag` uses antigen plus partner context; `_ag_lconly` is the
partner-only control on the same SAbDab2 corpus. Check the JSON for exact settings.

Run commands from the repository root:

```bash
python train.py configs/mlm/heavy/h3_scratch.json
python train.py configs/ar/h3_ar_scratch.json
python train.py configs/s2s/h3_s2s_esm2.json
./scripts/run.sh list
./scripts/run.sh plan compare
./scripts/run.sh plan sweep:lr --config configs/s2s/h3_s2s_scratch.json
./scripts/run.sh plan mlm_no_partner_controls
```

Suite names are unchanged. Config and data paths inside experiment specs are relative
to the repository root, not the enclosing config folder. Historical run configs and
plans under `results/` retain their original paths; use a fresh plan for new runs.

The October no-partner controls and context/dropout batches are complete;
completed recipes are examples, not a launch queue. The IgLM pilot now includes
the full replay, both zero-shot runs, four IgLM-S fine-tuning trials and all four
random-init/size-matched control trials. Larger-model fine-tuning and the adapted
checkpoint GPU benchmark smoke remain unrun. See the
[IgLM protocol](experiments/20261102_iglm_adaptation.md),
[experiment guide](experiments/README.md) and [results index](../docs/RESULTS.md)
for commands, saved evidence and current status.
