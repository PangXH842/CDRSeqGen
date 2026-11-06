# Research findings and limitations

[Back to README](../README.md) · [Results index](RESULTS.md) · [Current work](PROJECT_STATUS.md)

Detailed notes moved from the README on **26 October 2026**. Start with
[§17: expanded validation evidence](#17-expanded-validation-evidence-seeds-retrieval-and-capacity)
for the seven-seed, retrieval and capacity analyses. Earlier sections preserve their
original datasets, protocols and dates; their headings and interpretations are historical
claims, not findings established across every later experiment. The October 23 handoff
and §17 take precedence for completion status. Final held-out evaluation remains pending.

## 📊 Findings

Phase 1 reports sequence-infilling experiments under the split and selection limitations in
[Open problems](#open-problems). Historical findings below identify their source batches; the
[current S2S run status](WORKFLOW.md#s2s-pretrained-encoders-results-and-next-runs) is recorded separately.

*Numbers below are from the complete 36-run batch `results/runs/20260818_192606` (6 loops ×
{scratch, ESM-2} × context × {OAS, SAbDab2}), read at **best epoch** under the coverage-aware
recovery policy below. Every generated table states the selected metric and minimum core coverage.
Budgets are per family — 20 epochs for scratch, 10 for pretrained, both with
`early_stopping_patience: 3`; generative eval 150 frameworks × 20 candidates, independent decoding,
IgBert judge.*

> **✅ These are the first validation-selected numbers in this README.** The 08-18 batch is the
> re-run of the OAS matrix on `processed_oas_tuning.json` with `eval_split: validation`, so early
> stopping, `load_best_model_at_end` and every `eval_*` figure below read the **validation**
> partition and the test split stays untouched. Every earlier batch — including 07-29 and the 08-17
> post-refactor run — selected on "all non-train", which on the two-way file *is* the final test
> set. Protocol is otherwise identical to 07-29, so the two are directly comparable; the differences
> below are the cost of that contamination, and they are small (OAS H3 scratch 0.3185 → 0.3180).
>
> **The SAbDab2 rows are not covered by this.** SAbDab2 ships no validation partition, so its 12
> configs declare `eval_split: test` — honest, but still selection on the holdout. Every SD2 number
> below remains exploratory until a three-way SAbDab2 split exists (see Open problems).

> **⚠️ These numbers supersede the 07-24 batch, which trained every family for exactly 10 epochs.**
> That budget under-trained the scratch model, and it inflated the magnitude of **four** findings at
> once — §1, §2, §4 and §7 — all in the same direction, against the smaller model. See §9.

> **Every table here is generated, not hand-written** — regenerate them all after any new batch with:
> ```bash
> python -m cdrseqgen.visualization.batch_tables --batch results/runs/20260818_192606
> python -m cdrseqgen.visualization.batch_tables --only regimes,loops   # or a subset
> ```
> It reads only what a finished run already writes (`epoch_metrics.csv`, `generation_eval.json`,
> `run_config.json`) plus the processed dataset JSONs, and prints markdown ready to paste. The one
> exception is the decoding table (Step 4), which needs two `evaluate.py` runs differing only in
> `--decoding`, passed via `--decoding-json` (the pair used above is kept alongside that run as
> `generation_eval_decoding_{iterative,independent}.json`).

#### 1. "Model choice doesn't matter" is only true in two of three regimes

On OAS H3 a 128-dim from-scratch transformer matches ESM-2 (0.3180 vs 0.3214 core). But that generalises badly — whether pretraining pays depends entirely on how much of the test set is *memorisable* from train (see §6):

| case | unseen in train | metric | min core coverage | scratch | ESM-2 | Δ |
|---|---|---|---:|---:|---:|---:|
| OAS H1 | 1.5% | core | 100.0% | 0.9936 | 0.9935 | **−0.000** |
| SD2 H1 | 57.5% | core | 100.0% | 0.5445 | 0.6654 | **+0.121** |
| SD2 H2 | 79.6% | core | 100.0% | 0.3685 | 0.6096 | **+0.241** |
| OAS H3 | 100.0% | core | 100.0% | 0.3180 | 0.3214 | **+0.003** |

Pretraining buys ~nothing when the answer can be looked up, **+0.12–0.24** when the task needs genuine generalisation, and ~nothing again at the H3 ceiling. The original claim was drawn from regimes 1 and 3 only.

#### 2. CDR-H3 is the only hard loop — but the other five are easier than they look

Best-epoch recovery on OAS under the recorded core-coverage policy:

| CDR | reported metric | min core coverage | scratch | +ctx | ESM-2 | +ctx |
|---|---|---:|---:|---:|---:|---:|
| H1 | core | 100.0% | 0.9936 | 0.9905 | 0.9935 | 0.9939 |
| H2 | core | 100.0% | 0.9913 | 0.9793 | 0.9909 | 0.9909 |
| **H3** | core | 100.0% | **0.3180** | 0.3141 | **0.3214** | 0.3206 |
| L1 | core | 100.0% | 0.9937 | 0.9857 | 0.9940 | 0.9938 |
| L2 | whole | 1.1% | 0.9913 | 0.9846 | 0.9932 | 0.9928 |
| L3 | core | 99.9% | 0.9290 | 0.8927 | 0.9304 | 0.9298 |

The biology is right — heavy CDR3 gets D-gene + junctional diversity, the rest is germline-templated — but the ~0.99s are **not** evidence the model generalises; under the current split they are largely retrieval (§6). Only H3 is measured on entirely unseen loops.

#### 3. The standard metric misleads (anchor inflation)

Whole-span `eval_accuracy` (~0.46 on H3) is dominated by the 2-3 V/J-gene-determined anchor residues at each loop end (~0.90 recovery, any model); the hypervariable middle actually being "designed" is ~0.32. The gap is **H3-specific** (H1/H2/L core ≈ whole-span). Measured via `core_accuracy`/`anchor_accuracy` ([`compute_metrics`](../cdrseqgen/methods/mlm/recovery.py), passed to the Trainer by `train.py`); the position-resolved breakdown below is from the standalone re-scorer `eval_recovery.py` (archived in the gitignored `old/`, so it is not in a fresh clone). Illustrative (`h3_esm2`):

| Position from start | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 100% | 89% | 35% | 21% | 20% | 14% |

#### 4. Sequence context — partner chain and antigen — gives no measurable benefit

The valid same-data antigen ablation (`_ag` vs `_ag_lconly`, SAbDab2, all 3 CDRs) shows no signal, and it *survives the core metric*:

| CDR | metric | min core coverage | scratch lc-only → +ag | ESM-2 lc-only → +ag |
|---|---|---:|---|---|
| H1 | core | 100.0% | 0.5445 → 0.4872 | 0.6654 → 0.6629 |
| H2 | core | 100.0% | 0.3685 → 0.3483 | 0.6096 → 0.5884 |
| H3 | core | 99.7% | 0.1286 → 0.1353 | 0.2209 → 0.2128 |

|Δ| ≤ 0.057, sign inconsistent, 5 of 6 comparisons negative. Not a truncation artifact (84% of complexes keep the full antigen at `max_len=512`).

**Partner-chain context: the earlier "it hurts the small model" claim was mostly an under-training artifact.** At a 10-epoch budget it looked like a solid effect; at 20 epochs two thirds of it disappears:

| loop | Δ at 10 epochs | Δ at 20 epochs (validation-selected) |
|---|---|---|
| H1 | −0.0103 | −0.0031 |
| H2 | −0.0279 | −0.0120 |
| H3 | −0.0086 | −0.0039 |
| L1 | −0.0277 | −0.0080 |
| L2 | −0.0331 | −0.0067 |
| L3 | **−0.0786** | −0.0363 |
| **mean** | **−0.0310** | **−0.0117** |

Adding context lengthens the input and slows convergence: every `+ctx` scratch run peaked at **epoch 18–20**, against 4–14 without it, and five were still improving when the budget ran out — so −0.0117 is an *upper* bound. ESM-2 is unmoved either way (−0.001 to +0.003).

**With an error bar (finally).** Three seeds of `h3_scratch` ± partner chain, 20-epoch budget:

| seed | no ctx | +LC | Δ |
|---|---|---|---|
| 1 | 0.3196 | 0.3154 | −0.0042 |
| 2 | 0.3199 | 0.3123 | −0.0076 |
| 3 | 0.3187 | 0.3089 | −0.0098 |
| | sd **0.0006** | sd **0.0033** | mean −0.0072, sd **0.0028** |

The effect is real but small (2.5σ, all three seeds negative), and the noise floor on a per-loop Δ is **σ ≈ 0.0028**, i.e. ~0.006 at 2σ. Applying that to the table above, only four of the six loops survive:

| loop | Δ | \|Δ\|/σ | |
|---|---|---|---|
| L3 / H2 / L1 | −0.0363 / −0.0120 / −0.0080 | 13.0 / 4.3 / 2.9 | real |
| L2 | −0.0067 | 2.4 | marginal — just past 2σ |
| **H1 (−0.0031), H3 (−0.0039)** | | **1.1 / 1.4** | **not distinguishable from noise** |

So on the loop that actually matters — H3 — partner-chain context has **no measurable effect in either direction**, and the earlier −0.0086 was noise plus under-training. Note also that σ itself is a *seed* spread, and §13 shows run-to-run nondeterminism exceeds that; read the marginal row accordingly. A second observation from the same three seeds: the `+ctx` runs are **5× more variable** than the base runs (sd 0.0033 vs 0.0006) and their best epoch swings between 10 and 20, so context destabilises training as well as slowing it.

**Interpretation is unchanged:** sequence context — partner chain or antigen — buys nothing measurable. Binding information is geometric, not linear-sequence, so the next lever is structure / cross-attention. What changed is that the apparent *harm* was largely our own budget.

#### 5. The homology leak was real but immaterial *for H3*

OAS's original random split leaked clonal near-duplicates (76 of 2,111 test entries, 3.6%, share a CDR3 cluster with train). The clustered-split fix removed it (→0%), and re-running confirmed the H3 core moved by **≤0.004**. (Now the default; see [`preprocess/oas_paired.py`](../cdrseqgen/datasets/preprocess/oas_paired.py).) The important qualifier is *for H3* — the split clusters on heavy CDR3 only, so it controls nothing for the other five loops (§6).

#### 6. The clustered split controls H3 and nothing else — so the ~0.99s are retrieval

`cluster_by_cdr3` groups on heavy CDR3, and whole clusters are held out. Two antibodies with identical H1/H2 land on opposite sides freely as long as their CDR3s differ:

| | unique H1 strings | held-out entries with an **unseen** H1 | with an unseen H2 | recovery |
|---|---|---|---|---|
| OAS | 439 / 21,106 (2.1%) | 65 / 4,222 (**1.5%**) | 83 / 4,222 (2.0%) | ~0.99 |
| SAbDab2 | 895 / 3,984 (22.5%) | 431 / 750 (**57.5%**) | 597 / 750 (79.6%) | 0.33–0.43 |

*OAS is now counted over the three-way file's held-out partitions — validation **and** test, 2,111
each — hence 4,222 rather than the 2,111 of the old two-way split. The fraction is unchanged.*

98% of OAS held-out H1s appear verbatim in training, so H1/H2/L1/L2/L3 under this split are near-lookup tasks — which is why every model scores ~0.99 and why context and pretraining make no difference there. **SAbDab2 is not underperforming; it is the only place the non-H3 loops are honestly measured.** H3 is unaffected (100% of test H3s are unseen by construction). Fixing this needs a per-loop split — cluster on whichever CDR is the target — which is a contained change to `add_split_field`.

#### 7. Recovery and design quality disagree — high AAR can mean a perfect copier

From the post-training generative eval (150 held-out frameworks, 20 samples each, IgBert judge):

| | recovery | diversity | entropy (bits) | judge-ppl |
|---|---|---|---|---|
| H1 scratch | 0.999 | **0.000** | **0.000** | 1.021 |
| H1 ESM-2 | 0.999 | **0.000** | **0.000** | 1.021 |
| H3 scratch | 0.398 | 0.574 | 1.835 | 7.223 |
| H3 ESM-2 | 0.395 | 0.588 | 1.880 | 7.490 |

H1 scores 0.999 AAR while emitting essentially **one** sequence with total certainty (perplexity ≈ 1.0, diversity and per-position entropy both 0.000) — a perfect copier and a useless designer. That is the finding, and the validation-selected batch makes it starker than before: the copier is now exact.

> **The H3 half of this finding no longer holds, and should not be quoted.** Earlier text here said the judge put ESM-2 "1.7× ahead" of scratch on H3 while recovery called them equal. That gap was an artifact of the shared 10-epoch budget (§9 recorded it closing to 1.00× once scratch got 20 epochs), and the prose outlived the table it described. On the validation-selected batch the two metrics **agree**: recovery 0.398 vs 0.395, judge 7.223 vs 7.490, both a hair in scratch's favour. Recovery and design quality can still disagree — H1 above is the proof — but not between these two H3 models.

#### 8. A model is locked to the context regime it was trained in — and `context_dropout` unlocks it

Feeding a checkpoint the *wrong* amount of context is as damaging as feeding it none, in both directions, because for an absolute-position backbone a prepended partner chain shifts every learned position:

| H1 / OAS / scratch | eval WITH ctx | eval WITHOUT |
|---|---|---|
| trained WITHOUT ctx | **0.1588** | 0.9969 |
| trained WITH ctx | 0.9837 | **0.2612** |
| `context_dropout=0.5` | 0.9791 | 0.9931 |

Training a fraction of examples context-free (each context kind drawn independently, deterministic per example) buys one checkpoint that holds up in every regime. Worst-case recovery, 80 held-out frameworks, greedy:

| | specialist, own regime | specialist, wrong regime | `context_dropout=0.5`, worst |
|---|---|---|---|
| OAS H1 | 0.9969 | 0.1588 | **0.9791** |
| OAS H3 | 0.4353 | 0.0535 | **0.4244** |
| SAbDab2 H3 | 0.2271 | 0.1217 | **0.1766** |

**On OAS it is nearly free** (−0.011 to −0.018 against the best specialist, for a ~6× better worst case). **On SAbDab2 it is not** (−0.051): with 3,234 training entries, splitting across four context combinations leaves ~800 per regime, and the model ends up preferring the *no-context* input (0.2384) over the one it was configured for (0.1832). Read that as a data-size limit, not a refutation — dropout costs what the dataset can't afford to split.

Two side benefits on the datasets where it is cheap: the partner-chain and antigen ablations become *within-model* comparisons (same weights, two inputs) instead of comparisons between separately-trained runs with different init and SGD paths; and it removes the silent-degradation failure `generate.py` has to warn about.

#### 9. One un-audited hyperparameter inflated four findings at once

Every config trained for exactly `num_train_epochs: 10`, which suited the pretrained models (ESM-2 median best epoch **4**) and starved the scratch model (median **9**, with 8/18 still improving at the cut-off). Re-running with per-family budgets — 20 epochs for scratch, 10 for pretrained, both with `early_stopping_patience: 3` — moved four separate results, all in the same direction:

| finding | at 10 epochs | at per-family budgets |
|---|---|---|
| §1 pretraining lead, SD2 H1 | +0.231 | **+0.119** |
| §4 partner-chain penalty (mean) | −0.0310 | **−0.0102** |
| §7 judge-ppl gap, H3 scratch vs ESM-2 | 1.74× | **1.00×** |
| §2 scratch `+ctx` recovery | — | +0.007 to +0.055 |

The §7 change is the starkest: scratch's judge-perplexity went **12.42 → 7.22** against ESM-2's 7.23. Recovery and the judge no longer disagree; they were only disagreeing because one model was under-trained. (Not the decoder switch — iterative and independent were measured equivalent, and ESM-2's number barely moved.)

Early stopping cost nothing on the pretrained side — mean Δ over 12 OAS runs was **−0.0000**, and zero runs were cut with their best at the final epoch — while shortening the batch from 295 to 226 minutes.

**The lesson is procedural:** a single budget shared across model families is not a neutral choice, and every distortion it caused ran against the smaller model. Anything compared across families should have its budget justified per family, or the comparison measures the budget.

#### 10. Removing native H3 length exposes a separate unsolved problem

The first length-free run used
the validation-selected scratch checkpoint (`1e-3` learning rate, 0.06 warm-up, 0.01 weight decay,
seed 42). Its frozen sequence backbone supplied the representation for a gap-sentinel length
classifier; neither fitting nor generation received the native length.

| length result | learned head | mode baseline (always 15) |
|---|---:|---:|
| exact accuracy, 2,111 validation frameworks | **9.24%** | 10.52% |
| mean absolute error | **3.64** | 3.28 |

The downstream adapter then produced 500/500 syntactically valid candidates on the frozen
25-framework length-free validation manifest, with zero failures and **30/500 (6.0%)** native-length
matches. That 6.0% is a sampled-candidate rate on a small manifest, whereas the table reports one
argmax decision per example over the full validation split, so they should not be equated.

This is a useful negative result: the current representation/head does not extract more length
information from the framework than a trivial global guess. Fixed-native and length-free results
must remain separate tracks; §11 scores the length-free track against IgLM.

#### 11. Matched length-free comparison: IgLM leads on plausibility and length, and the recovery gap does not resolve

Both models ran the frozen `oas_h3_validation_length_free_v1` manifest — 25
validation frameworks × 20 candidates, one scorer, one IgBert judge. CDRSeqGen predicts a length
from its gap sentinel; IgLM terminates on its own EOS. Neither receives the native length.

| metric | CDRSeqGen (tuned scratch) | IgLM | better |
|---|---:|---:|---|
| whole-span AAR | 0.3324 | **0.3532** | IgLM |
| core AAR | 0.1576 | **0.1838** | IgLM |
| anchor AAR | **0.8611** | 0.8260 | CDRSeqGen |
| length match rate | 6.0% | **8.4%** | IgLM |
| length MAE | 4.51 | **3.69** | IgLM |
| edit similarity | 0.3596 | **0.3890** | IgLM |
| judge perplexity | 9.74 | **4.07** | IgLM (2.4×) |
| diversity (Hamming) | **0.8047** | 0.7784 | CDRSeqGen |
| valid generation rate | 1.00 | 1.00 | tie |
| runtime / retained sample | **0.078 s** | 0.164 s | CDRSeqGen |

**With a paired bootstrap over examples, only half of those margins survive (2026-08-31).** The
example is the resampling unit, not the candidate — 500 draws from 25 frameworks are not 500
observations — and one index draw is applied to both arms, so per-framework difficulty cancels in
the difference. A metric uses only examples where *both* models report it, which for positional AAR
means the 14 frameworks where both produced a native-length candidate. 10,000 resamples, seed 42:

| metric | n | CDRSeqGen | IgLM | difference | 95% CI | resolved |
|---|---:|---:|---:|---:|---|---|
| whole-span AAR | 14 | 0.3288 | 0.3340 | −0.0052 | [−0.0414, +0.0308] | no |
| core AAR | 14 | 0.1540 | 0.1686 | −0.0146 | [−0.0624, +0.0339] | no |
| anchor AAR | 14 | 0.8304 | 0.8065 | +0.0238 | [−0.0238, +0.0685] | no |
| length match rate | 25 | 0.0600 | 0.0840 | −0.0240 | [−0.0520, +0.0040] | no |
| length MAE ↓ | 25 | 4.5100 | 3.6900 | +0.8200 | [+0.3660, +1.2340] | **IgLM** |
| edit similarity | 25 | 0.3596 | 0.3890 | −0.0294 | [−0.0481, −0.0124] | **IgLM** |
| judge perplexity ↓ | 25 | 9.7364 | 4.0709 | +5.6655 | [+5.0044, +6.3255] | **IgLM** |
| diversity (Hamming) | 25 | 0.8047 | 0.7784 | +0.0263 | [+0.0147, +0.0387] | **CDRSeqGen** |

**So the headline should not be "IgLM leads on every quality axis" — that was this section's
original claim and the intervals do not support it.** What survives is narrower and more specific:
IgLM is decisively more *plausible* (judge perplexity, a 5.67 gap whose interval is nowhere near
zero) and decisively better at *length* (MAE, edit similarity). Every **recovery** comparison —
whole-span, core, and CDRSeqGen's apparent anchor advantage alike — is unresolved at n = 25, and the
point estimates sit well inside their intervals. CDRSeqGen's diversity edge is individually resolved but small.

**Correcting for the eight endpoints changes no verdict.** Eight unadjusted 95% intervals carry a
family-wise error near 34% under independence, so the multiplicity caveat is worth stating — but at
Bonferroni-adjusted 99.375% intervals all four resolved metrics still exclude zero: judge perplexity
[+4.75, +6.61], length MAE [+0.16, +1.39], edit similarity [−0.057, −0.006], diversity
[+0.011, +0.044]. Edit similarity sits nearest the boundary and still clears it.

Two readings follow. The recovery axes are not evidence of parity either: an interval spanning zero
at this sample size means *not measured*, not *equal*. And the axis that does resolve is the one
CDRSeqGen loses worst — judged plausibility — which is the same signal §7 and §12 keep returning.
Reproduce with:

```bash
python benchmark/bootstrap.py \
    --scores results/benchmark/oas_h3_validation_length_free_v1/scores/length_free.json --markdown
```

**This inverts the fixed-length result — and the fixed-length track turns out to be unfair to
IgLM.** On the fixed-native smoke manifest CDRSeqGen wins recovery outright (0.3318 vs 0.2155
whole-span, 0.1491 vs 0.1080 core), but that comparison is an artifact of the `constrain` policy.

`constrain` sets `min_length = max_length = native_length` on IgLM's `generate` call, and Hugging
Face's `min_length` suppresses the end-of-span logit until that length is reached — so IgLM is not
merely told the length, it is **forbidden from ending its span** and keeps generating into the
downstream framework. The corruption is directly measurable in the outputs:

| IgLM fixed-length H3 loops | ends in the J-gene motif (`…DV`) | matches the native's last two residues |
|---|---:|---:|
| `constrain` (fixed-length track) | **2%** | **14%** |
| natural EOS (length-free track) | 20% | 76% |
| *natives in the manifest* | *20%* | *—* |

The most common four-residue endings under `constrain` are `YWGQ`, `DYWG`, `GQGT`, `WGQG` — these
are **FR4**, the framework that follows CDR-H3, being counted as loop residues. Naturally IgLM ends
`YMDV`, `YFDY`, `WFDP`, `AFDI`. This is what drives its anchor AAR down to **0.552** against its own
natural-EOS 0.826, while CDRSeqGen's holds at ~0.9 in both tracks.

The tell was visible in the aggregate before the mechanism was found: **IgLM scores worse when handed
the native length (0.2155) than when choosing freely (0.3532)**, and extra information should never
hurt.

**Re-run under the fair protocol, the fixed-length result reverses (2026-08-21).** The adapter's
`filter` mode samples IgLM freely and retains only naturally length-matched draws. On the 20 of 25
frameworks where both models produce native-length loops (matched population, IgBert judge):

| fixed-length, matched population | CDRSeqGen | IgLM `filter` |
|---|---:|---:|
| whole-span AAR | 0.3353 | **0.3569** |
| core AAR | 0.1495 | **0.1868** |
| anchor AAR | **0.8850** | 0.8565 |
| diversity | 0.6097 | 0.6106 |
| per-position entropy (bits) | **1.3008** | 1.0858 |
| judge perplexity ↓ | 9.8244 | **4.7593** |

IgLM's anchors recover from 0.552 to 0.857 — near its natural-EOS 0.826 — confirming that `constrain`,
not the model, produced the collapse. The clearest view is the position curve under the two policies
(`results/figures/benchmark/20260821_regenerated/constrain_vs_filter_position.png`): under `constrain` IgLM's
C-terminal recovery flatlines near 22% while CDRSeqGen climbs to 86%; under `filter` the two curves
converge across the whole loop. **So IgLM leads on both tracks.** The previously reported
fixed-length win was an artifact of the generation policy, and no CDRSeqGen-versus-IgLM recovery
advantage survives a fair protocol.

Two honest counterweights. First, `filter` has its own bias: it drops the frameworks IgLM cannot
match. Five of 25 retained nothing in 50 attempts and they skew long (native lengths 12, 20, 22, 24,
24), so the matched population averages 16.4 residues against 17.2 for the full manifest — both
models are being scored on a slightly easier subset. Neither policy is unbiased: `constrain` corrupts
the output, `filter` shrinks the population. Second, the masked-LM formulation does hold one
structural advantage the recovery numbers hide: **CDRSeqGen produces a native-length loop by
construction, every time**, while IgLM needed 1,081 attempts for 76 usable candidates (7.0%
acceptance) and failed outright on 20% of frameworks within its budget.

*Caveats, all of which must be resolved before this number is quoted outside the repo:* n = 25
examples — AAR is conditioned on length matches, so it rests on 18 CDRSeqGen / 17 IgLM examples
individually and only **14 paired** ones, which is why the recovery intervals above are as wide as
they are. The bootstrap resamples frameworks while holding each framework's finite candidate pool
fixed, so it does not include generation-seed uncertainty; the eight per-metric 95% intervals are
also unadjusted for multiple endpoints. ~~No confidence intervals yet~~ **done 2026-08-31**
([`bootstrap.py`](../benchmark/bootstrap.py));
the remaining blockers are that the scores were produced from a **dirty working tree at the previous
commit** (`repository_commit: 0e7e499`, `repository_dirty: true`), so a clean-tree rescore is still
required, and that IgLM was pretrained on OAS, so this holdout is not a leakage-free estimate for it
(roadmap: overlap audit). The clean-tree rescore is expected to move the point estimates only
slightly — but the intervals above should be regenerated from it rather than carried over.

#### 12. The model's own confidence is nearly useless for selection — an independent judge's is not
Every candidate now carries its own pseudo-perplexity and judge perplexity, so the pairing with its
own recovery survives into `generation_eval.json` instead of being averaged away. That makes a
practical question answerable: given 20 candidates for one framework, does picking the most
confident one beat picking at random? (150 held-out frameworks × 20 candidates, `h3_OAS_scratch`,
IgBert judge; per-framework Spearman, then averaged.)

| ranking signal | Spearman ρ | top-1 recovery | lift over a random draw |
|---|---:|---:|---:|
| random draw (baseline) | — | 0.4066 | — |
| **self**-perplexity | 0.130 ± 0.047 | 0.4176 | **+0.011 ± 0.012 — not significant** |
| **judge** perplexity | 0.254 ± 0.046 | 0.4591 | **+0.053 ± 0.013 (8.0σ)** |

The model's own uncertainty carries a statistically detectable but practically useless signal: ρ is
5.4σ from zero, yet selecting on it does not reliably beat a random draw (95% CI crosses zero), and
its top pick beats the framework mean only 85/150 times — barely better than a coin flip. An
independent model ranks roughly twice as well (Δρ = +0.124, 4.4σ) and its top pick beats the
framework mean 103/150 times, yielding **+12.9% relative recovery for free — no retraining, just
reranking**.

Panel B of `results/figures/training/20260821_confidence/confidence_selection.png` carries the
qualifier that the means hide: the per-framework ρ distributions are wide and both straddle zero.
Self-perplexity ranks *backwards* on **46 of 150** frameworks, the judge on **29 of 150**. So even
the judge is a population-level effect, not a rule you can trust on any single framework — worth
stating before anyone reads "+12.9%" as a guarantee.

> **⚠️ The ordering did not hold on a second checkpoint.** Scored on the 08-25 screen's
> cross-entropy run, self-perplexity *outranked* the judge: top-1 lift **+0.0276 against +0.0186**,
> the reverse of the table above. That run is not a replication — 100 frameworks × 10 candidates
> against 150 × 20, on the tuned config rather than `h3_OAS_scratch` — and halving the candidate
> pool makes top-1 selection substantially noisier. But it is enough that the *ranking* of the two
> signals should be treated as unsettled until one `evaluate.py` run at 150 × 20 on the tuned
> checkpoint says otherwise. The claim that survives either way is the practical one: an
> independent judge is a usable reranker, and reranking beats a better sampler.

**The figure is written automatically.** Any evaluation that produces per-candidate scores also
drops `<output>_confidence.png` beside its JSON — the same convention as the training curves — so
every `train.py` run and every `evaluate.py` call gets one without a second command. It is skipped
silently when there is nothing to plot (`--skip_perplexity` with no judge) and is exception-guarded,
so a plotting problem cannot cost a finished evaluation. To redraw, or to overlay two runs:
```bash
python -m cdrseqgen.visualization.confidence_plot --eval <run>/generation_eval.json
python -m cdrseqgen.visualization.confidence_plot --eval a.json --label scratch --eval b.json --label ESM-2
```

This sits directly alongside the decoder finding: iterative decoding buys nothing because the
learned marginals are near-independent, and now the same model's confidence barely tracks its own
correctness. Both say the masked-LM objective produces poorly *structured* uncertainty, not merely
inaccurate predictions. The practical corollary is that a reranking stage is worth more than a
better sampler — and that the reranker must be an independent model.

#### 13. Re-weighting the loss across loop positions changes nothing — the anchors are not wasted capacity

The 08-25 screen trained the tuned scratch config (`1e-3`, 0.06 warm-up, 0.01 weight
decay, 30 epochs) under four objective settings × 3 seeds; 10 of 15 trials completed before it was
interrupted. Ranked three ways, the winner changes with the metric:

| ranked by | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| judge perplexity ↓ | **cross-entropy** (6.94) | weighted 1/1 (7.04) | weighted 3/1 (7.09) | core-only (31.60) |
| free-running core@L | **weighted 1/1** (0.2294) | cross-entropy (0.2286) | core-only (0.2266) | weighted 3/1 (0.2258) |
| teacher-forced core | **weighted 1/1** (0.3226) | weighted 3/1 (0.3224) | cross-entropy (0.3212) | core-only (0.3207) |

**The two settings trading first place are the same loss function.** `weighted_cross_entropy` at
core/anchor `1.0/1.0` is bit-identical to `cross_entropy` — with equal weights the weighted mean
reduces to the plain token mean (verified to 1e-9). The grid therefore contained only **three**
distinct objectives, not five: `3.0/0.0` is likewise identical to `1.0/0.0`, because the loss
normalises by total weight and only the core-to-anchor *ratio* survives.

That accident supplies the number that settles the comparison. Two runs of an identical objective at
identical seeds differ by **0.096 judge perplexity, 0.0014 teacher-forced core, 0.0009
free-running core** — and every real between-objective gap is at or below that floor. Weighted 3:1,
the only genuinely distinct non-degenerate setting, sits inside it on all three metrics.

**Core-only is the one decisive result, and it is a clean negative.** Setting
`anchor_loss_weight 0` collapses anchor recovery from **0.873 to 0.042**, which is what drives judge
perplexity to 4.5×. It did *not* buy anything for the region it targets: core@L 0.2266 against
cross-entropy's 0.2286. So down-weighting the anchors does not reallocate capacity to the
hypervariable middle — the core stays exactly where it was and the anchors break. The position curve
in `results/runs/20260825_165553_sweep_loss/figures/objective_comparison.png` shows it directly:
core-only tracks every other objective across relative positions 0.2–0.7 and falls away only at the
two ends. Its higher diversity (0.716 against 0.560) is an artifact of emitting implausible anchors,
not better design.

*Caveats: weighted 3:1 has n=1 — the sweep was interrupted before its other two seeds ran, so it is
bounded by the noise floor rather than properly measured. H3/OAS/scratch only, one hyperparameter
setting, generative eval at 100×10 rather than the 150×20 used in §12. One thing strengthens the
conclusion: checkpoint selection ran on `core_accuracy`, the metric a core-weighted objective
directly optimises, and the weighted settings still did not win.*

This is the third independent result pointing one way. Iterative decoding buys nothing because the
marginals are near-independent (Step 4); the model's confidence barely tracks its own correctness
(§12); and now re-weighting tokens within the objective moves neither recovery nor plausibility. The
problem is the objective *class*, not the weighting inside it.

> **Scope of the historical findings:** the ~0.32 core / ~0.46 whole-span plateau describes the tested fixed-native MLM settings. It is not an objective-independent ceiling. The AR/S2S teacher-forced and length-free generation results below use different conditioning and must be reported separately. Non-H3 generalization still requires target-loop-specific splits.

#### 14. Context dropout improves missing-partner robustness across MLM, AR and S2S

The [12-run context screen](../results/runs/20260915_131629_103719_context_dropout/runs.tsv) completed and was reviewed on **2026-09-17**: scratch AR, scratch S2S and ESM-2 S2S × dropout 0/0.5 × seeds 42/43. Each checkpoint generated 20 candidates for the same 150 OAS4 validation frameworks, with and without generator partner context; the IgBert judge always received partner context. Checkpoints were selected by teacher-forced core recovery with partner context present. Two-seed means:

| Model | Train dropout | Edit, partner present ↑ | Edit, partner absent ↑ | Judge PPL, present ↓ | Judge PPL, absent ↓ |
|---|---|---|---|---|---|
| AR scratch | 0.0 | 0.3910 | 0.3815 | 3.261 | 3.890 |
| AR scratch | 0.5 | 0.3900 | 0.3893 | 3.180 | 3.242 |
| S2S scratch | 0.0 | 0.3905 | 0.3829 | 3.208 | 3.405 |
| S2S scratch | 0.5 | 0.3889 | 0.3878 | 3.269 | 3.278 |
| S2S ESM-2 | 0.0 | 0.3932 | 0.3689 | 3.526 | 4.368 |
| S2S ESM-2 | 0.5 | 0.3899 | 0.3872 | 3.531 | 3.558 |

Dropout improves partner-absent edit similarity in **both seeds for all three models**: mean gains **+0.0078 / +0.0050 / +0.0183** for AR / scratch S2S / ESM-2 S2S. Partner-present edit costs are **−0.0010 / −0.0016 / −0.0032**, also consistent in direction across seeds. Partner-absent judge perplexity improves in every pair. This is a robustness trade-off; it does not show that dropout improves every metric or that training without partner context would be worse. The six no-partner training controls are now complete ([October 7 review](../results/reviews/20261007-no-partner-controls.md)); comparisons to the September screen span source/runtime versions.

**Coverage and limits:** 71,999/72,000 candidates were valid. The sole exception was a complete-empty draw from ESM-2/dropout 0/seed 43 without partner context. All 150 frameworks contributed valid candidates in every regime, and judge coverage was complete over valid candidates. Edit similarity excludes that empty draw; generated core recovery is conditional on matching length and has different contributing populations. Two seeds are screening evidence, without a significance or equivalence claim. The existing automatic `batch_tables.md` omits three ESM-2 trials through family-key collisions; use the complete per-seed tables in the [September 15–17 review](../results/misc/reviews/0915-0917-review.md).

**ESM-2 confirmation — audited 2026-10-07.** All four trials at seeds 44/45 and dropout 0/0.5 passed the [completion audit](../results/reviews/20261007-esm2-confirmation.md): eight scored outputs, the same 150 validation frameworks in both regimes, **24,000/24,000 valid draws** and full judge coverage. Final weights match the selected checkpoints; sequence summaries were reproduced from the saved candidates.

| Seed | Dropout | Edit, partner present ↑ | Edit, partner absent ↑ | Δ present (0.5 − 0) | Δ absent (0.5 − 0) |
|---|---|---|---|---|---|
| 44 | 0.0 | 0.3930 | 0.3056 | — | — |
| 44 | 0.5 | 0.3888 | 0.3894 | −0.0042 | +0.0838 |
| 45 | 0.0 | 0.3926 | 0.3285 | — | — |
| 45 | 0.5 | 0.3887 | 0.3901 | −0.0039 | +0.0616 |

The missing-partner robustness direction repeats in both new seeds, with mean changes **+0.0727 absent / −0.0041 present**. The absent-context effects exceed the September screen; keep batches separate because seeds, executable source and runtime differ. These are descriptive validation results, with cluster-aware uncertainty and final held-out benchmarking still pending. All four configs record a dirty checkout; their executable fingerprint is frozen. The interrupted fourth trial restored its optimizer/RNG and early-stopping state, finished at epoch 13, and saved the best epoch-10 weights. Its resumed epoch-12 checkpoint was pruned by normal retention; the recovery journal retains the input hashes. **No rerun is needed for completion.** The two scratch MLM no-partner controls are now complete; their audit and next stage are below.

**MLM no-partner controls — audited 2026-10-07.** Both scratch MLM trials at seeds 42/43 passed the [completion audit](../results/reviews/20261007-mlm-no-partner-controls.md). Each uses the same 150 validation frameworks, a checkpoint-specific inferred-length head, independent residue sampling and 20 returned candidates per framework. Training and generation omit the light chain; the IgBert judge receives it.

| Seed | Edit ↑ | Length match | Length MAE ↓ | Core@L ↑ | Core frameworks | Judge PPL ↓ |
|---|---|---|---|---|---|---|
| 42 | 0.3880 | 7.57% | 4.405 | 0.2006 | 114/150 | 8.062 |
| 43 | 0.3906 | 6.43% | 4.316 | 0.1950 | 94/150 | 7.589 |

The two runs return **6,000 nonempty canonical candidates** with complete judge coverage. MLM does not record an attempt-validity ledger; this is returned-candidate coverage, not a measured attempt-validity rate. Core@L is conditioned on correct length (227/193 candidates). Final model weights match the selected epoch-8/14 checkpoints; training stopped at epoch 11/17. The fitted length heads reach 9.86%/9.73% validation argmax accuracy versus 10.00% for the training-mode baseline; sampled generation length match is a separate quantity. These are descriptive validation results, with dirty-checkout provenance and frozen executable source/runtime preserved. **No rerun is needed.** The four matched partner-enabled MLM dropout trials are now complete, as reviewed below. The six earlier AR/S2S no-partner controls and four ESM-2 confirmations also need no completion rerun.


**MLM ESM-2 no-partner controls — audited 2026-10-22.** The [two new trials](../results/reviews/20261022-mlm-esm2-no-partner-controls.md) add the pretrained MLM system under the same intended control-table evaluation protocol. Both passed checkpoint/head, exact framework membership and saved-metric checks.

| Seed | Edit | Length match % | Length MAE | Core@L | Core frameworks / matched candidates | Judge PP |
|---|---:|---:|---:|---:|---|---:|
| 42 | 0.3891 | 7.07 | 4.035 | 0.2019 | 101 / 212 | 8.117 |
| 43 | 0.3883 | 7.63 | 4.236 | 0.2042 | 98 / 229 | 7.815 |

Training and generation omit the partner; the judge receives it. Both stop at epoch 8 and retain epoch-5 sequence models. The 6,000 returned candidates have complete judge coverage; MLM attempt validity remains unrecorded. The ten-trial control table now contains 30,000 returned candidates, with the six AR/S2S trials retaining their 18,000 valid-attempt accounting. These ESM-2 MLM rows demonstrate pretrained-model integration; no causal pretraining improvement or dropout effect is claimed. They are not copied into the heavy-only external benchmark. **No completion rerun is needed.**

**MLM dropout comparison — audited 2026-10-07.** All four partner-trained scratch MLM trials (dropout 0/0.5, seeds 42/43) passed the [completion audit](../results/reviews/20261007-mlm-dropout-screen.md). Their eight evaluations return **24,000 nonempty canonical candidates** with full judge coverage; attempt validity is unrecorded. They reuse the same 150 validation frameworks, inferred-length/independent decoder and partner-present judge as the two no-partner controls. This completes the six-checkpoint, ten-evaluation MLM context study.

| Seed | Training policy | Edit, partner present ↑ | Edit, partner absent ↑ | Judge PPL, present ↓ | Judge PPL, absent ↓ |
|---|---|---|---|---|---|
| 42 | No partner | — | 0.3880 | — | 8.062 |
| 42 | Partner, dropout 0 | 0.3931 | 0.1575 | 7.945 | 43.863 |
| 42 | Partner, dropout 0.5 | 0.3914 | 0.3907 | 7.906 | 7.856 |
| 43 | No partner | — | 0.3906 | — | 7.589 |
| 43 | Partner, dropout 0 | 0.3923 | 0.1389 | 7.728 | 49.307 |
| 43 | Partner, dropout 0.5 | 0.3936 | 0.3915 | 7.763 | 7.623 |

Dropping partner context only at inference substantially reduces native edit similarity for dropout-0 MLM. Training dropout 0.5 restores partner-absent similarity close to the no-partner-trained controls: gains **+0.2332/+0.2526**, versus the dropout-0 partner-trained models. Partner-present changes are **−0.0017/+0.0013**; unlike the AR/S2S screens, their directions differ. These observations do not establish equivalence, significance or a universal context-present cost. Length remains limited: match rates are 5.70–7.40%, and none of the four length heads beats the training-mode baseline on validation argmax accuracy or MAE. Conditional-recovery populations and all eight rows are in the audit.

Final weights match the selected checkpoints (epochs 16/17/19/19), and all length-head artifacts passed integrity checks. Frozen launch-time source/runtime match the MLM controls. During this batch, MLM annotations changed and the same loss-config/logging statements moved after model loading; all other executable AST nodes match the frozen source. The command-interface refactor followed completion. The audit preserves snapshots and unchanged plan/ledger hashes; byte-identical code at every subprocess launch is not asserted. **No completion rerun is needed.** The matched validation benchmark and its later extensions are now complete (§16–17); final held-out settings remain to be frozen. Additional dropout confirmation seeds/repeats should follow the intended claim and precision, separately from the completed seven-seed no-partner benchmark.

#### 15. ESM-2 learning-rate tuning improves teacher-forced recovery more clearly than generation

The [four-run LR screen](../results/runs/20260915_214327_432282_sweep_lr/summary.tsv) completed at `1e-5` and `1e-4`, seeds 42/43, up to 20 epochs. These are **partner-trained, dropout-0** models evaluated with partner context present; partner-absent evaluation was not part of this sweep. The `3e-5` reference below comes from the context screen. All use the same 150 × 20 generation protocol; means across two seeds:

| ESM-2 LR | Teacher-forced core ↑ | Generated edit ↑ | Generated core@L ↑ | Judge PPL ↓ | Length match | Length MAE ↓ |
|---|---|---|---|---|---|---|
| 1e-05 | 0.3880 | 0.3917 | 0.2137 | 3.941 | 9.07% | 3.834 |
| 3e-05 | 0.3980 | 0.3932 | 0.2260 | 3.526 | 8.53% | 3.813 |
| 1e-04 | 0.4032 | 0.3928 | 0.2104 | 3.345 | 8.62% | 3.762 |

Within the new sweep, **`1e-4` beats `1e-5` in both seeds** on teacher-forced core, edit similarity and judge perplexity. Against the `3e-5` reference, `1e-4` improves mean teacher-forced core by **0.0052** and lowers judge perplexity by **0.180**, but edit similarity is slightly lower on average (**−0.0003**) with opposite seed directions. Generated core@L also falls, but its population differs (107/106 contributing frameworks at `1e-4`, versus 116/109 at `3e-5`). **Keep `1e-4` as a follow-up candidate, not an overall generation winner; no default config was promoted.** Its interaction with dropout 0.5 remains untested.

All **12,000/12,000** LR-sweep candidates were valid with full judge coverage. The final teacher-forced evaluations cover 15,271/15,273 validation examples for core accuracy. Native answer prefixes make these ~0.40 teacher-forced scores different from historical MLM masked recovery; they are not evidence of a direct improvement over its ~0.32 result. All 16 runs across the two batches record dirty-tree provenance, and the batch source hashes differ. Preserve those records and confirm selected comparisons under a fixed clean revision before final paper claims. See the [September 15–17 review](../results/misc/reviews/0915-0917-review.md) for config differences, per-seed results and input hashes.

---

#### 16. Matched OAS4 length-free validation benchmark

The [seven-arm batch](../results/benchmark/oas4_h3_length_free_20261007) completed shared scoring on
**7 October at 20:51 Singapore time** and passed its [8 October artifact audit](../results/reviews/20261008-benchmark-full.md).
All arms use the same 150 validation frameworks, with 20 attempts per framework: **21,000/21,000
valid candidates and full judge coverage**. Scratch MLM/AR/S2S reuse no-partner-trained checkpoints
at seeds 42/43; IgLM is one frozen released model. Heavy-chain flanks are the only generator
context, native length is withheld, and the shared IgBert judge uses heavy-chain context only.
MLM predicts length; AR/S2S/IgLM terminate naturally with EOS. Temperature is 1, top-p 0.9 and the
autoregressive cap 97 new tokens. [Recipe and frozen protocol](../benchmark/recipes/oas4_h3_length_free.md).

| System | Training seed | Edit similarity | Length match % | Length MAE | Conditional core | Core frameworks / matched candidates | Judge PP |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLM | 42 | 0.3905 | 6.83 | 4.351 | 0.2292 | 102 / 205 | 7.676 |
| MLM | 43 | 0.3919 | 7.43 | 4.313 | 0.2200 | 107 / 223 | 7.390 |
| AR | 42 | 0.3880 | 9.17 | 3.931 | 0.2104 | 117 / 275 | 3.150 |
| AR | 43 | 0.3888 | 8.77 | 3.961 | 0.2253 | 116 / 263 | 3.151 |
| S2S | 42 | 0.3870 | 9.70 | 3.855 | 0.2052 | 116 / 291 | 3.244 |
| S2S | 43 | 0.3881 | 8.60 | 3.874 | 0.2173 | 113 / 258 | 3.161 |
| IgLM | — | 0.3880 | 8.87 | 3.707 | 0.1944 | 104 / 266 | 4.047 |
| Empirical training H3 | — | 0.2817 | 6.97 | 4.508 | 0.1208 | 113 / 209 | 5.009 |

Edit, length and judge means average candidates within framework, then all 150 frameworks.
Core recovery uses only native-length candidates and different contributing framework populations.
Mean edit similarity spans 0.3870–0.3919; these small differences do not establish superiority or
significance. IgLM has the lowest observed length MAE, while AR/S2S have lower IgBert perplexity.
The latter is a judge-dependent proxy. Unequal capacities/training exposure and unresolved OAS
pretraining overlap prevent an isolated architecture ranking.
The earlier one-donor IgLM pilot (§11) remains separate. No completion rerun is needed.

**Empirical follow-up — audited 2026-10-08.** The [follow-up audit](../results/reviews/20261008-benchmark-followup.md)
adds 3,000 training-only unconditional H3 draws, giving **24,000 valid/judge-scored candidates** across
eight arms. All seven learned-system-minus-empirical edit contrasts have positive Bonferroni family
intervals (observed gains +0.1052 to +0.1102). All six local-checkpoint-minus-IgLM marginal and family
intervals span zero. These exploratory comparisons fix trained checkpoints/candidate pools; they
do not estimate training/generation variation or isolate framework conditioning. The
[13 contrast rows](../results/benchmark/oas4_h3_followup_20261008/analysis.md) retain separate local seeds.

Recovered groups reproduce every saved corpus split label. All 150 benchmark frameworks have
distinct representative group IDs, so group resampling reduces to paired framework resampling.
Core-recovery paired intersections contain 79–96 frameworks/groups and differ from each arm's
individual conditional population. Representative grouping does not guarantee biological-family
separation. Saved parameter counts are **869,949** for MLM including its 4,900-parameter length head,
**800,256** for AR, **933,120** for S2S and **12,889,600** for released IgLM; the empirical sampler has
zero neural parameters. Actual stopping/selected epochs, optimizer steps and Trainer runtime are
exported, with unavailable historical hardware/pretraining compute left unknown.

The manuscript includes the eight-row validation/accounting tables, paired edit intervals and
vector figures, plus supporting population and capacity tables. Its final held-out table remains
pending. All original and follow-up result artifacts are unchanged; audits verify hashes,
training-only source indices, exact validation membership and independent baseline sequence metrics.

---

#### 17. Expanded validation evidence: seeds, retrieval and capacity

The completed October 9–12 batches extend §16 on the same **150 validation frameworks**. The
[sealed analysis](../results/benchmark/20261012_143836_oas4_h3_variability/variability.md) combines
seven training seeds per scratch method with external and training-only reference pools.
All reported comparisons are exploratory; the cohort and candidate pools are reused.

| System / reference | Training seeds | Native edit similarity | Interpretation |
|---|---:|---:|---|
| MLM scratch | 7 | 0.3918 ± 0.0017 | Mean ± training-seed SD |
| AR scratch | 7 | 0.3868 ± 0.0014 | Mean ± training-seed SD |
| S2S scratch | 7 | 0.3882 ± 0.0013 | Mean ± training-seed SD |
| IgLM | Released checkpoint | 0.3880 | One candidate pool |
| IgCraft, temperature 0.1 | Released checkpoint | 0.3769 | One candidate pool |
| Unconditional training-H3 sampling | — | 0.2817 | No framework conditioning |
| Primary flank retrieval | — | 0.3589 | Last 12 FR3 + first 6 FR4 residues |
| FR4-only retrieval | — | 0.3469 | Sensitivity arm |
| Finer-key retrieval | — | 0.3653 | Sensitivity arm |

**Retrieval is a substantive reference.** The primary lookup captures 70–73% of the scratch
models'/IgLM gain over unconditional sampling and 81% of IgCraft's gain. Scratch models and
IgLM exceed it by 0.028–0.033; IgCraft by 0.018. All five method-level native-edit family
intervals exclude zero in the seed-aware report. Retrieval-key development was exploratory;
the two sensitivity arms are outside that primary contrast family. These findings concern
native sequence similarity under the measured information budget, not binding quality.

**Seed-aware comparisons replace a single ranking.** The analysis resamples framework groups
and each method's training seeds. MLM–AR has a nominal 95% interval of [0.0011, 0.0089], but
its family-adjusted interval is [-0.0008, 0.0108]. All three local-method-versus-IgLM native-edit
intervals include zero. This does not establish equivalence. Anchor-trimmed core edit is a
separate supporting endpoint, defined after the primary analysis; it is not conditional
length-matched core recovery. Candidate-generation variability remains unmeasured.

**Same-seed repeats and capacity are complete.** AR generates identical candidate sets across
all three seed-42 runs; MLM and S2S vary, and one MLM repeat trains 20 epochs rather than 11.
The 18-arm capacity study compares three larger sizes at seeds 42/43 with the small checkpoints:
approximately 0.8–0.9M to 25–29M parameters changes native edit similarity by at most 0.0033.
The learning rate was not retuned for size, so this does not establish a capacity ceiling.
[Capacity figure](../results/benchmark/20261012_143836_oas4_h3_variability/capacity.pdf) ·
[Seed-spread figure](../results/benchmark/20261012_143836_oas4_h3_variability/seed_spread.pdf).

**IgCraft sampling sensitivity is complete.** Temperature 1.0 increases mean loop length from
12.42 to 13.79 residues and diversity from 0.635 to 0.745, while edit similarity decreases
from 0.3769 to 0.3506. The comparison remains method- and setting-specific; temperatures are
not interchangeable across generator families.

The [October 20 independent verification](../../iAIM/verification/REPORT.md) covers the newer
44 arms and 32 seed-aware contrasts, as well as the older paper evidence. An October 21
read-only review also checked 74 saved variability input/output hashes with no mismatch.
These are result checks, not completion of manual code review. Newer batches retain their
recorded source/runtime provenance, including dirty working trees where recorded. Final
held-out evaluation and the manuscript update remain pending; no completion rerun is needed.

---

## ⚠️ Current Issues
*(status reviewed 2026-09-14; historical measurements retain their original batch dates)*

### Open problems
- **Generative evaluation contracts updated (2026-11-06 review).** Self/judge score counts now must match candidates, and non-finite scores become unavailable before means, coverage and candidate records. Supplied-case self-scoring follows the same checks. `confidence_metrics` also filters non-finite pairs and rejects unequal input lengths. Standalone `aggregate()` still assumes sanitized metric records; stratified selection continues to favor first-observed buckets when the framework budget is smaller than the number of non-empty buckets. Historical saved results retain their original scoring source.
- **The `OAS4` corpus confounds donor with disease, by construction.** Jaffe 2022 contains exactly four donors and no more (all 580 catalogued paired runs probed 2026-09-03), and their conditions do not overlap: Donor-1 and Donor-2 are SARS-CoV-2, Donor-3 healthy, Donor-4 CMV. A donor-held-out split is therefore also a disease-held-out split, and neither effect can be attributed separately. Mitigated but not removed by every run being **Naive-B-Cells**: naive repertoires are largely pre-antigen-selection, so disease shapes them far less than it shapes memory or plasmablast compartments. Recorded before any `OAS4` experiment was run.
- **More donors would cost B-cell-type purity.** 96 further donors exist across the other 18 paired studies, but only **one** naive paired run among them (Woodruff_2020); the rest are unsorted, memory, or plasmablast. Somatic hypermutation differs sharply between those compartments, and it lands hardest on the germline-encoded H1/H2 — the loops §6 already shows to be retrieval-bound. Any expansion past four donors must record `BType` per entry (`read_oas_metadata` does not capture it today) and stratify on it, or the heterogeneity becomes invisible in the merged corpus.
- **`OAS4` results are not comparable to any earlier OAS number.** A different donor set is a different test split. The `oas_multi -> OAS4` tag keeps the two apart in run names and batch tables mechanically, but no table should place them on one axis — the same rule the `_OAS` / `_SD2` tags exist to enforce.
- **H3 clustering does not guarantee pairwise separation.** Representative clustering permits same-length training neighbours at ≥85% identity across named splits. The [completed October audit](../results/reviews/20261008-leakage-audit.md) confirms 3/150 affected benchmark frameworks and retains 14,818/15,273 candidate test records after exclusions. Final cohort selection is pending. Different-length/gapped homology and non-H3 target-loop separation remain separate questions; 98% of the older OAS test H1s occur in train (§6).
- **H3 length is close to unpredictable from the framework, and every learned method is below the achievable ceiling.** A non-neural lookup — group training examples by (last 12 residues of FR3 | first 6 of FR4), a crude V/J-gene proxy, and take the modal H3 length — reaches **11.07% exact / 3.01 MAE** on the OAS4 validation split, covering 98.3% of it. Against that: the §10 length head 9.24% / 3.64, causal `<eos>` 7.23% / 4.00, and always predicting the global mode 10.12% / 3.53. So there IS signal (MAE 3.01 against a constant's 3.53, a 15% reduction) but very little, and **neither learned method captures even that much**. Three interventions moved nothing — decoding temperature, a 10x learning-rate range, and 7x more data — which is what a near-flat ceiling predicts. The same verdict holds on the older split the length head was measured on: 9.24% / 3.64 against 10.52% / 3.28 for always predicting length 15 (§10) — below its naive baseline there too. Treat length accuracy as a property to report against these baselines, not a target to optimise; the V/J lookup should replace the global mode as the reference number. Before treating inferred MLM generation as competitive, evaluate probability calibration and top-k length coverage, then test a representation or objective that actually learns the framework-to-length signal.
- **Training still optimizes recovery, not design.** The masked-LM objective rewards reproducing the native loop; the generative metrics only *measure* design quality after the fact — nothing trains for it. Findings §7 shows the two can disagree outright (H1: 0.99 recovery, 0.002 diversity). A design-aligned objective stays open (Phase 2). Re-weighting tokens *within* the masked-LM objective has now been tested and does not help (§13), so this needs a different objective class rather than a better weighting.
- **Training variability has now been measured for the scratch benchmark.** Seven seeds per method, two additional identical-config seed-42 runs per method, and crossed seed/framework intervals are complete (§17). AR repeats produce identical candidates; MLM and S2S do not. Historical loss/context studies retain their own seed counts and uncertainty limits, and the IgBert collapse/non-reproduction remains a separate instability case. Seed spread, fixed-seed repeat variation and evaluation-population uncertainty describe different sources of variation; generation-pool variance is still unmeasured.
- **Loss-sweep selection and ranking are separate.** The shared runner now defaults the `loss` preset to whole-span checkpoint selection (`SELECT_METRIC=accuracy`) and judge-perplexity ranking (`RANK_METRIC=judge_ppl`). This makes anchor collapse visible to selection, but does not train directly for design quality. Historical loss-screen results retain their original checkpoint-selection policy; new summaries label both selection and ranking metrics.
- **Scratch and pretrained S2S have completed initial experiments.** The [September 8 comparison](../results/runs/20260908_173143_compare/batch_tables.md) contains six completed runs. Without partner context, judge perplexity is 7.512 / 3.154 / 3.215 for MLM / AR / S2S, with inferred length for MLM and EOS termination for AR/S2S. This is one training seed, and conditional recovery uses different length-matched subsets. It does not establish an AR-versus-S2S winner. The [first S2S PLM matrix](WORKFLOW.md#s2s-pretrained-encoders-results-and-next-runs) also completed all six runs: judge perplexity is 3.441–4.132, with no improvement over the earlier scratch S2S reference. Unequal capacities/budgets, one seed, and incomplete convergence prevent attributing this to pretraining; IgBert + partner has confirmed encoder representation collapse ([diagnostics](../results/misc/diagnostics/20260914_igbert/REPORT.md)), but the 2026-09-14 replay at the same seed and settings did **not** reproduce it (`20260914_140305_226060_igbert_diagnostic`), so its trigger, incidence and remedy remain open; this rerun did not reproduce collapse within three epochs.
- **The S2S LR screen and shipped defaults use different optimizer settings.** The [six-trial scratch screen](../results/runs/20260907_204721_sweep_lr/summary.tsv) tested LR 1e-4 / 3e-4 / 1e-3 at warmup 0.06 and decay 0.01, seeds 42/43; 3e-4 has the lowest mean judge perplexity in that screen. The fresh comparison and current scratch configs use LR 1e-4 with zero warmup and zero weight decay; the Trainer's linear schedule still decays the LR to zero at the 20-epoch maximum (checked against the logged LR of all 16 scratch/ESM-2 control runs). A sweep does not automatically promote its settings into configs. New PLM configs start at LR 3e-5 with warmup 0.06 and decay 0.01 and still need tuning.
- **The `compare` suite's three arms are matched on protocol but not on capacity.** Corpus, split, evaluation frameworks, candidate count, `top_p`, `max_len`, judge and seed all match; parameter counts do not. `ar` is 800k, and `s2s` is 933k (1.17x) even after pinning its stacks to 2/2 — the closest match that keeps real depth on both sides, since the nearest-by-count alternative needed a single-layer encoder and would have gutted the one thing the method exists to test. The counts are logged in every `train.log`; **quote them in any table**, and do not read a 1.17x-larger model's win as an isolated architecture result. The completed capacity scan (§17) adds evidence under a fixed training recipe, but does not equalize tuning effort or isolate architecture effects.
- **The causal method's plausibility advantage is judge-specific, and a native-referenced metric contradicts it.** On OAS4 H3 the causal model reaches judge perplexity 3.15 against masked 9.35 at equal recovery — but BLOSUM62 global alignment over retained non-empty candidates puts it behind: 0.146 against 0.199. The two disagree because both are confounded by length in opposite directions: the aligned score charges gap penalties for the 93% of causal draws whose length is wrong, while the masked arm is handed the native length and never pays one; judge perplexity is length-blind and native-free, so it is indifferent to exactly what causal is bad at. Neither is a clean method comparison until both arms predict their own length (the `compare` suite). Also worth recording: **core BLOSUM is ~0 for every H3 model** (causal 0.020, masked -0.001, ESM-2 0.003) — core substitutions are on average no better than the matrix expects between unrelated residues, which exact-match core recovery of ~0.20 made look like partial credit.
- **The judge shares the eval distribution.** IgBert was trained on OAS paired repertoires — the same distribution as the OAS eval split it judges. It's applied identically to every model, so the *comparison* is fair, but absolute perplexities are OAS-flavoured; an off-distribution judge would be a stronger check.
- **Generative sweep is H3/OAS only.** The matrix covered CDR-H3 on OAS (150 held-out frameworks × 4 models). L-chain loops, SAbDab2, and the antigen configs have not been swept generatively.
- **Historical iterative decoding showed little improvement.** On the July H3/OAS/ESM-2 comparison, iterative and independent decoding had similar mean recovery (0.4057 vs 0.4072) and judge pseudo-perplexity (7.14 vs 7.23), at roughly 15× the compute for iterative decoding. These observations do not establish statistical equivalence or independence of residue predictions, and do not establish the best decoder for mixed-mask checkpoints. AR/S2S free-running evaluations remain separate comparisons.
- **Three runs hit their epoch ceiling, all of them context runs.** `h1_OAS_LC`, `h1_SD2_LC` and `l3_OAS_HC` (all scratch) peaked at their final epoch, 20 of 20, in [`20260818_192606`](../results/runs/20260818_192606) — the batch the Findings tables quote. Their numbers are lower bounds, so the §4 partner-chain penalty (−0.0102) is an upper bound on the true effect. A scratch-only sweep at 30–40 epochs would settle it (~1 hour; scratch is 60 of the batch's 226 minutes). The earlier 07-29 batch had five such runs; `h2_OAS_LC`, `h3_OAS_LC` and `l1_OAS_HC` no longer hit the ceiling in the 08-18 re-run.
- **Uncertainty coverage depends on the study.** The expanded no-partner scratch benchmark has seven seeds per method and paired seed/framework analysis (§17). Historical context/dropout, loss, LR and PLM comparisons retain their original replication levels. Do not apply the seven-seed benchmark's uncertainty to those studies or claim equivalence from an interval that includes zero.
- **Every batch through 2026-08-17 selected checkpoints on the test split.** Not just the 07-29 runs: `eval_split` defaulted to empty, which fell back to "every non-train entry", and `processed_oas_data.json` has only `train`/`test` — so early stopping, `load_best_model_at_end`, the reported `eval_*` metrics, and `generation_eval.json` all read the final holdout. The 08-17 post-refactor batch (`results/runs/20260817_200051`) is affected too; its logs read `2111 eval (split: all non-train)`. **Fixed 2026-08-18**: `get_datasets` now refuses an unset `eval_split` whenever a `test` split exists, and the 36 OAS configs declare `processed_oas_tuning.json` + `eval_split: validation`. **Resolved for OAS 2026-08-31**: the matrix was re-run as `results/runs/20260818_192606` and every Findings table now quotes it, so the OAS numbers are validation-selected and the test partition is untouched. Two things still stand: the **SAbDab2** rows come from configs declaring `eval_split: test` and remain selection-on-holdout (see the SAbDab2 entry below), and any figure elsewhere in this README still sourced from a pre-08-18 batch — the Step 4 decoding table, the §12 confidence numbers, and the §3 position breakdown — has not been re-derived and stays exploratory.
- **Neither fixed-length policy is unbiased — report both.** *(Re-run 2026-08-21; see Findings §11.)* `constrain` corrupts IgLM's loop endings; `filter` is fair per-example but drops the frameworks IgLM cannot length-match (5 of 25, skewing long), so it scores an easier subset. Current practice: report `filter` on a matched population and state the dropout. A policy that is unbiased in both respects — e.g. raising the attempt budget until every framework is covered, or reporting per-length-bucket — is still open. Historical note on `constrain`: IgLM's `constrain` policy sets `min_length = max_length = native_length`, which suppresses its end-of-span logit until that length is reached — so it writes past its loop into FR4. Only 2% of constrained loops carry the J-gene motif against 20% naturally (Findings §11), and every fixed-length CDRSeqGen-versus-IgLM number is inflated in CDRSeqGen's favour. Re-run with the adapter's `filter` mode (retain only naturally length-matched draws; ~12 attempts per accepted candidate at IgLM's 8.4% match rate) before making any fixed-length claim. The length-free track is unaffected — IgLM uses natural EOS there.
- **SAbDab2 has no validation partition.** Its split comes from the source `ab_ag_split` column, which is `train`/`test` only, so its 12 configs now declare `eval_split: test` — honest, but still selection on the holdout. A three-way SAbDab2 split (or a fixed validation carve-out of its train partition) is needed before any SAbDab2 comparison is reportable.
- **Hyperparameter selection is not yet frozen.** The validation-safe 32-trial scratch-H3 screen completed on 2026-08-11 and placed `1e-3` / 0.06 warm-up / 0.01 weight decay first at 0.32305 mean validation core recovery. Its margin over the runner-up was only 0.00036 across two seeds and changed sign by seed, so additional shared confirmation seeds are still required before touching the final test split.
- **Checkpoint-selection policy is still config-declared.** The current configs select core recovery except the four OAS-L2 configs, which select whole-span because those runs have only 1.4% core coverage. Artifact reporting no longer hard-codes that loop name: it derives the metric from recorded coverage. Training should eventually validate or derive `metric_for_best_model` from the dataset before a run, so a future dataset cannot inherit an obsolete L2 exception.
- **Implementation checks do not establish experimental conclusions.** The suite passed 419 tests as of 2026-09-14, including real tiny ESM/BERT weights, encoder freezing, gradients, separate vocabularies, and exact checkpoint reloads. Actual ESM-2/IgBert GPU smoke runs also exercised training and generation with the real judge. The September 13 PLM batch subsequently completed full ESM-2, IgBert, and ProtBert training and generation in all six settings. Passing checks does not establish split separation, convergence, or biological quality; see [the test guide](../unit_tests/README.md).
- **Structure not yet available.** SAbDab2 preprocessing parses *sequences* only; the Phase-2 structure objective needs a coordinate-parsing step (or predicted structures via IgFold/ABlooper).

### How to read the numbers
- **Report core only when it represents the dataset.** Whole-span recovery is anchor-inflated on H3, but a core score is also misleading when most loops have no residues left after trimming. The shared 50% policy uses `eval_core_accuracy` only when the run's recorded `eval_core_coverage` is at least 50%; otherwise core is `N/A` and the explicitly labeled whole-span score is reported.
- **OAS L2 is an observed case, not a coded exception.** With `ANCHOR_TRIM=2`, 98.8% of its loops are three residues, so only ~30/2,111 evaluation examples (1.4%) contain a core. The reporter reads that 1.4% from `epoch_metrics.csv` and selects whole-span. A future L2 dataset with adequate coverage would use core, while any other short-loop dataset would correctly fall back to whole-span.
- **The historical fixed-native MLM tables supply the true loop length.** Length-free results in §10–11 and the fresh three-method comparison are separate protocols. `dataset_mlm.py` inserts exactly `len(native)` mask tokens for that protocol, so the length is given away by the input's shape. This is standard fixed-length redesign and is required for position-wise AAR to be well-defined, but it is not free information; how much it is worth varies enormously by loop:

  | loop | distinct lengths | mode freq | free info from knowing L |
  |------|------------------|-----------|--------------------------|
  | H3   | 34               | 9.9%      | **4.05 bits**            |
  | L3   | 14               | 43.4%     | 2.13 bits                |
  | L1   | 6                | 54.3%     | 2.01 bits                |
  | H2   | 6                | 62.0%     | 1.24 bits                |
  | H1   | 4                | 83.8%     | 0.73 bits                |
  | L2   | 3                | 98.8%     | **0.10 bits**            |

  On H3 that's ~4 bits handed over per example — roughly 9% of the loop's total information content (a 15-mer at the measured 2.96 bits/position ≈ 44 bits; position-conditioned entropy over the 2,098 length-15 H3 loops). On L2 it's nothing: guess 3 and you're right 98.8% of the time. So **H3 recovery is the number most flattered by this**, and cross-loop comparisons of AAR partly compare how much length information each loop gave away. Length-free results are therefore a separate track with length/edit metrics, not a replacement value in these tables.
- **Teacher-forced accuracy is not generation accuracy, and the core is where they diverge.** `eval_accuracy`/`eval_core_accuracy` are argmax over the true context; the generative eval samples. On `h3_OAS_scratch` (300 frameworks x 10 candidates, fixed-native, T=1.0, independent decoding, all 3,000 candidates length-matched):

  | metric | teacher-forced (argmax) | free-running (sampled) |
  |---|---|---|
  | whole-loop | 0.4604 | 0.3695 |
  | **core** | 0.3180 | **0.2042** |
  | anchor | — | 0.8397 |

  The whole-loop number is carried by the anchors (0.84): they are largely V/J-determined, so a model that had learned nothing about the hypervariable middle would still post a respectable whole-loop AAR. Judge the core column. Against baselines fitted on train and evaluated on validation over the same 26,205 core positions — matched to the sampling protocol, since an argmax baseline is not comparable to a sampled model — sampling from H3's marginal core amino-acid frequencies expects **0.084**, and sampling from a per-`(length, position)` frequency table expects **0.146**. At 0.204 the model beats the lookup table by ~40% relative, which is a real but modest margin, and the honest headline for free-running H3 core design.
- **Decoder choice and training masking are separate settings.** Iterative decoding fills one remaining mask per pass, exposing committed residues to later predictions. Mixed masking includes partially revealed loops during training; full-span masking hides the entire loop. A historical full-span comparison showed lower whole-loop/core recovery with iterative decoding (0.370 → 0.253 / 0.204 → 0.077). Current evaluation, generation and benchmark setup use an explicit decoder request first, then a compatible saved choice, then **independent**. Recorded iterative decoding is honored automatically only with mixed masking; missing masking metadata is treated as full-span. An explicit iterative request remains allowed, and mixed masking alone does not select iterative decoding.
- **Read best-epoch, not last-epoch** — ESM-2 overfits on SAbDab2/H3; `load_best_model_at_end` saves the peak, but the last CSV row can sit well below it.
- **Never plot OAS and SAbDab2 on shared axes** — different test sets; the `_OAS`/`_SD2` run tags enforce it, and per-dataset charts live under `results/figures/training/<batch>/*.png`.
- **`epoch_metrics.csv` and `*_results.json` measure different models — they're *supposed* to disagree.** The CSV is the per-epoch **trajectory** (the model as it stood at each epoch); `eval_results.json`/`all_results.json` come from the post-training `trainer.evaluate()`, which with `load_best_model_at_end` runs on the **best** checkpoint while still being stamped with the final epoch number. On `h3_esm2` the CSV's epoch-10 row reads core **0.3063** (the actual last model) while `eval_results.json` reads **0.3209** (the epoch-4 best model). `save_training_csv` de-duplicates on epoch and **keeps the first** occurrence precisely so the curve keeps the real last-epoch point — keeping the last would silently redraw the best model at x=10 and erase the overfitting decay entirely.
