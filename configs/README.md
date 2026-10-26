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

All six October AR/S2S no-partner controls and all four S2S ESM-2 confirmation
trials are complete. The next prepared stage is `mlm_no_partner_controls` (two
runs), followed by `mlm_dropout_screen` (four runs) if cross-method robustness is
in scope. Both share [the MLM control base](experiments/bases/h3_mlm_scratch.json).
See the [experiment guide](experiments/README.md) for commands and
[RESULTS.md](../docs/RESULTS.md) for saved evidence and status.
