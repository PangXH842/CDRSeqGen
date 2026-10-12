# OAS4 length-free validation: seeds, determinism, capacity and sampling sensitivity

**Prepared 12 October 2026; CPU only, from the saved batches of the 9-12 October queue.** Run it with
`./scripts/run.sh benchmark variability run`, adding `--references results/benchmark/<retrieval batch>` once the retrieval
reference arms exist. It reads sealed inputs, writes a new directory and modifies nothing.

The queue trained seeds 44-48 for each method, retrained seed 42 twice with an identical config, scanned model size, and
reran IgCraft at temperature 1.0. [`benchmark/variability.py`](../variability.py) turns those batches into the numbers the
manuscript quotes, each recomputed from saved candidates and scores, with every input hash-checked and frozen.

| Block | Question | Definition |
|---|---|---|
| Seeds | How much does training seed matter, and are the methods distinguishable once it is counted? | Mean and spread over seven seeds per method (42, 43 from the original batch; 44-48 trained later). Contrasts resample H3 groups **and** each arm's seeds (`analysis.crossed_interval`); `groups only` keeps seed means fixed |
| Determinism | Is retraining with an identical config reproducible? | Seed 42 three times: scores, identical 20-candidate sets, largest weight difference, epochs run and best metric |
| Capacity | Does model size matter? | M, L and XL scratch models (about 5M, 11-12M, 25-29M parameters) at seeds 42 and 43 against the small models of the same seeds; the change is also expressed in units of the small models' seed sd |
| Sensitivity | Is IgCraft's loop-length compression the sampler's doing? | Default temperature 0.1 against 1.0, with loop-length statistics and one exploratory paired contrast |
| Endpoints | Does anything change when the conserved anchors are removed? | Every arm again on the **core**: two residues trimmed from each end of candidate and native; a candidate with no core scores 0. A supporting endpoint, defined after the primary one |
| Ladder | How much of the learned gain does flank retrieval capture? | (retrieval - unconditional) / (learned - unconditional), per endpoint. Needs `--references` |

The primary endpoint stays native edit similarity; the core endpoint is supporting. The contrast family is ten
contrasts (three method pairs, three methods against IgLM, four against IgCraft), or sixteen with retrieval; family
intervals are Bonferroni over the family of that endpoint. Everything is validation-split, exploratory, and chosen after
the data were seen.

## What the 12 October preview shows (scratch run; the sealed run supersedes it)

- **Seeds:** edit similarity MLM 0.3918 +- 0.0017, S2S 0.3882 +- 0.0013, AR 0.3868 +- 0.0014; IgLM 0.3880. With seed and framework
  uncertainty both counted, only MLM - AR excludes zero (+0.0049 [+0.0011, +0.0089]); the three methods all tie with IgLM.
  On the core, MLM also separates from IgLM (+0.0072 [+0.0026, +0.0119]), and every system exceeds IgCraft with a family interval
  that excludes zero.
- **Determinism:** AR gives identical candidates for all 150 frameworks in all three runs (weights differ by at most 4e-6); S2S
  varies by about 0.001 (weights up to 5e-2); MLM by about 0.002 (weights up to 0.4), and one repeat trained 20 epochs where the
  others stopped at 11. The seven-seed spread is the same size as the fixed-seed repeat spread. The 7 October refactor left no trace.
- **Capacity:** from 0.8M to 25-29M parameters AR and S2S stay within 1.3 seed sd of their small models; MLM gains about 0.003
  (2 seed sd) by 5M and then plateaus.
- **Sensitivity:** temperature 1.0 lowers IgCraft's edit similarity by 0.026 [-0.033, -0.020] and lengthens its loops (13.8 vs 12.4
  residues, natives 16.3), so the compression is partly the sampler's and partly the model's.

## Limits

Training seeds 42 and 43 predate the 7 October command-interface refactor and 44-48 follow it; the determinism block shows the
refactor changed nothing measurable. A percentile bootstrap over seven seeds is slightly too narrow, the 20 candidates per
framework are a single draw whose Monte-Carlo noise is not in any interval, and the scan changes only size (the learning rate was
not retuned for the larger models). Figures: `capacity.png/.pdf` (size against edit similarity, with IgLM and IgCraft) and
`seed_spread.png/.pdf` (the ladder, and every seed for the learned systems).
