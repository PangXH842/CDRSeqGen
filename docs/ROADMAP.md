<!-- Note to AI: Edits allowed OUTSIDE of MANUAL PROTECTION ZONES and INSIDE of AI ACCESS ZONES, and DO NOT COMMIT ON MY BEHALF -->
# Project roadmap

[Back to README](../README.md) · [Current handoff](PROJECT_STATUS.md) ·
[Completed evidence](RESULTS.md)

Research milestones, experiment options and feature plans, moved from the README on
**26 October 2026**. Dated entries retain their original scope. Use the current handoff
and results index for completed-run status; checkbox changes remain deliberate edits.
Detailed personal code-review records are kept locally.

<!-- MANUAL PROTECTION ZONE BEGINS -->
## ⏭️ Roadmap

<!-- AI ACCESS ZONE BEGINS -->
> **Current priority — 6 November:** the iAIM abstract is accepted for an in-person poster. The shared backbone and training interfaces are aligned across all three methods; native IgLM replay, zero-shot, both fine-tuning sizes and random-init/size-matched control batches are complete. The selected-checkpoint benchmark smoke is prepared and awaits user execution. Continue manual code review using the [workflow map](../cdrseqgen/methods/common/README.md), and integrate the completed §17 evidence into the framework manuscript and poster. Automated checks and refactors do not complete manual review.

### Medium-term milestone — framework paper and final evaluation

**Main contribution:** a unified framework for antibody CDR infilling and benchmarking, currently implementing MLM, AR and S2S, with explicit method interfaces, allowed conditioning information, length policies, attempt budgets and scoring populations. IgLM, IgCraft and training-only samplers demonstrate the external adapter/scorer contract. Retrieval, variability and context-dropout results provide empirical case studies.

**Scope:** sequence-only H3 infilling, with length-free generation as the primary track. Native-length redesign and earlier MLM studies remain separate diagnostics. Current comparisons are validation evidence; structure-conditioned design, other loops and isolated pretraining effects are follow-ups if the paper retains those claims.

1. **Continue manual reclamation:** review AR `models/transformer` → `train_cli` → `training`, then `eval_cli`/`evaluation`, `gen_cli`/`generation` and `sampling`. Compare the corresponding S2S blocks, including its PLM model and diagnostics, then review the remaining common workflows, callbacks, sampling/loss helpers and logging/artifacts using the workflow map. Automated checks support this review rather than complete it.
2. **Develop the manuscript and poster:** integrate the verified newer evidence, explain the framework contribution and extension contract, expand related work, retain limitations and obtain advisor/coauthor review. The canonical abstract and current PDFs were refreshed on October 23; dated archives need rebuilding when the manuscript is ready to circulate.
3. **Prepare the final evaluation alongside writing:** freeze cohort selection from the audit-eligible pool, common generator support, checkpoint/seed choices, decoding settings, metrics and contrasts. Adapt the validation-only runner and manifest export for the chosen protocol. Reuse existing checkpoints where compatible; disclose external pretraining uncertainty and historical test use.
4. **For the planned final held-out comparison:** run the frozen evaluation and report it separately from exploratory validation. The abstract/poster can present the existing findings with their exploratory scope stated. Add experiments only for a specific unresolved claim; no new broad seed, baseline or capacity sweep is currently required.

### Experiment coverage across methods

Evidence and implementation snapshot: **2026-11-05**. **☑** completed development evidence; **◐** partial/historical evidence; **☐** pending; **—** not applicable to the current implementation. These are experiment statuses, not manual code-review marks. Pending entries are claim-dependent options.

The [results index](RESULTS.md) retains historical categories; the [current handoff](PROJECT_STATUS.md) and [§17](FINDINGS.md#17-expanded-validation-evidence-seeds-retrieval-and-capacity) link the newer evidence directly. Preserve each batch's original provenance and source/runtime differences.

| Experiment category | MLM | AR | S2S |
|---|---|---|---|
| Scratch no-partner benchmark | ☑ Seven seeds, 42–48 | ☑ Seven seeds, 42–48 | ☑ Seven seeds, 42–48 |
| Matched external references | ☑ IgLM, IgCraft, empirical and retrieval | ☑ Same references | ☑ Same references |
| Fixed-seed repeat study | ☑ Seed 42, three runs | ☑ Seed 42, three runs | ☑ Seed 42, three runs |
| Capacity study | ☑ S/M/L/XL; seeds 42/43 | ☑ S/M/L/XL; seeds 42/43 | ☑ S/M/L/XL; seeds 42/43 |
| Seed/framework uncertainty | ☑ No-partner benchmark | ☑ No-partner benchmark | ☑ No-partner benchmark |
| Train with/without partner context | ☑ Matched scratch controls | ◐ Historical paired runs; no-partner controls complete | ◐ Historical paired runs; scratch/ESM-2 no-partner controls complete |
| Context dropout in both inference regimes | ☑ Four trials, seeds 42/43 | ☑ Scratch screen, seeds 42/43 | ☑ Scratch/ESM-2 screens and ESM-2 confirmations at seeds 44/45 |
| Pretrained backbones | ☑ Historical studies and audited ESM-2 no-partner controls | ☑ Native IgLM backbone; replay, zero-shot and both fine-tuning sizes complete | ☑ ESM-2/IgBert/ProtBert encoder matrix |
| Same-architecture pretrained/random control | ☐ Claim-dependent | ☑ IgLM-S random-init pilot at three learning rates; seed 42 | ☐ Claim-dependent |
| Frozen-encoder/collapse diagnosis | ◐ Historical context studies | ◐ Context ablations | ◐ IgBert diagnostic trained; generation/replication pending |
| Decoding and length sensitivity | ☑ Historical decoder/length-head studies | ☑ Historical temperature/length study | ☐ Controlled sampling screen if needed |
| Target-loop-specific light-chain splits | ☐ Historical results only | ☐ | ☐ |
| Final frozen held-out comparison | ☐ | ☐ | ☐ |

The IgLM pilot controls completed 4/4 trials with 12,000 valid, judge-scored
candidates, including the three same-architecture random-init runs and a
size-matched scratch AR. These single-seed validation results remain separate
from the frozen external benchmark. Both larger-IgLM trials also passed the
[November 6 audit](../results/reviews/20261106-iglm-large.md): 6,000 valid candidates; validation core selects
lr 1e-5, epoch 4. The selected-checkpoint GPU benchmark smoke is prepared but unrun; see the
[adaptation protocol](../configs/experiments/20261102_iglm_adaptation.md).

### Active milestone — code review and paper draft

**Evidence is ready for drafting.** The original eight-arm comparison, IgCraft extension, three retrieval variants, seven-seed analysis, same-seed repeats and capacity study are complete. IgCraft temperature sensitivity is also complete. See §16 for the original subset and §17 for the expanded findings. All numerical comparisons remain exploratory; uncertainty varies by study and endpoint.

**Abstract submitted on 23 October.** Continue the code review described in the [current handoff](PROJECT_STATUS.md), and retain the canonical abstract for any later revisions. Remaining writing work is to update the main results and figures, explain the shared workflow/extension contract, expand related work and retain the limits of the data and metrics. Avoid claiming architecture superiority, equivalence or independent biological quality from these comparisons. The local split audit is complete, while final cohort selection and external pretraining membership remain unresolved.

**Deferred historical method backlog.** The following items preserve earlier implementation and research questions. Revisit their relevance as code review and paper claims develop, and distinguish work already covered by §17 from genuinely different settings. They do not require new runs now.

**General**

**Done.** Sampler (temperature/top-k/top-p, independent + iterative Mask-Predict) and generative metrics ([`sampling.py`](../cdrseqgen/methods/mlm/sampling.py), [`metrics.py`](../cdrseqgen/eval/metrics.py)); generative sweep across the H3 model/context matrix (150 held-out frameworks × 4 models, IgBert judge); reporting hygiene (core-vs-whole validity derived from recorded `eval_core_coverage`, both labeled, never branching on loop name); `num_attention_heads`/`intermediate_size` plumbed through `build_model` (`intermediate_size` defaults to 4 × `hidden_size`); three-way clustered OAS splitting with explicit validation/test selection and full run provenance; 32-trial validation-safe H3 screen — top region `1e-3` LR / 0.06 warmup / 0.01 weight decay at **0.32305** mean two-seed core recovery; core/anchor loss screen — **no re-weighting beats plain cross-entropy**, and core-only collapses the anchors without helping the core (§13); `loss` grid canonicalised by core-to-anchor ratio (2026-09-10); policy-aware decoding with `anchor_trim` propagated into generative evaluation (2026-08-31); checkpoint selection separated from ranking (whole-span for selection, judge perplexity for ranking; historical results keep their original policy).

- [ ] **Per-loop split** — cluster on the *target* CDR, not always heavy CDR3, then re-run the OAS matrix. Everything non-H3 in Findings is retrieval-bounded until this lands (§6)
- [ ] Confirm the top two sweep settings with additional shared seeds, freeze the selected config, then evaluate it once on the untouched test split
- **Completed for the current no-partner benchmark:** seven training seeds per method and seed/framework intervals (§17). Further context/dropout replication is claim-dependent and remains separate.
- [ ] Finish the loss screen's two missing `weighted 3.0/1.0` seeds (43, 44) so the only distinct untested setting is measured rather than noise-bounded; skip `3.0/0.0` — it is the same objective as `1.0/0.0`
- [ ] Settle the §12 self-vs-judge ordering with one 150 × 20 `evaluate.py` run on the tuned checkpoint — the 08-25 screen inverted it at 100 × 10
- [ ] Re-run the 07-24 batch under core-based checkpoint selection (those 36 runs predate the fix; their saved weights were picked on whole-span)
- [ ] Lock the Phase-2 objective and primary metric with advisor

**Autoregressive method**

**Done — the length question is closed (through 2026-09-04).** All six `ar` configs retrained under the fixed `LOOP_LENGTH_RESERVE` (`20260902_164544_ar`; `valid_rate` ~1.0 confirms the `<eos>` mechanism is healthy). H1/H2 are perfect and therefore meaningless (§6 retrieval regime); on H3 the causal model reaches **8.3% exact length / 3.80 MAE**, against §10's length head at 9.24%/3.64 and a modal-length baseline at 10.52%/3.28 — `<eos>` does **not** beat the length head, and both sit below the constant baseline on the natural length distribution. Three causes excluded: decoding (temperature sweep moves MAE only 4.31 → 3.48 from T=1.0 to greedy, exact-match never leaves ~8%; mean length well calibrated at 16.11 vs 16.17 native but under-dispersed, correlating 0.396 with the target); learning rate (`20260903_155940_sweep_h3` — teacher-forced core moves 0.0005 across a 10× range against a 0.006 noise floor); data scale (`20260903_192305` — at 7.2× the data the model held at exactly 1.13× its corpus's own modal-length MAE, and a V/J-conditioned lookup puts the achievable ceiling at 11.07% / 3.01). **Length is now a reported property, not an optimisation target.** The causal `generate.py` counterpart landed 2026-09-07 (currently `ar/gen_cli.py` + `ar/case_generation.py`, dispatching through the registry).

- [ ] **Decide the showcase fixture** — blocking the causal CLI (diagnosed 2026-09-07). Every `showcase/*.json` carries one "design only" case with no `target_truth` and an explicit `target_length`, because a masked LM must be told a length when there is no native to derive one from. Both length-free methods reject that key and abort the whole file; dropping it runs all three H3 cases correctly. Either add length-free probe sets, or relax the two CLIs to warn-and-ignore a length request they cannot honour
- **OAS4 scratch capacity study complete:** 18 larger-model runs across MLM/AR/S2S, three sizes and seeds 42/43 (§17). Further capacity/tuning interactions need a specific research question; this scan does not isolate architecture effects or establish a capacity ceiling.
- [ ] **Harden `SpanLayout` before registering a second layout** (review finding 2026-09-04; none of it affects the shipped `CDRSEQGEN_CAUSAL` or any current result): `build_prompt` receives the whole framework dict **including `target`**, so a layout could leak the answer into its own prompt and the training-equals-generation test would still pass; `n_layout_tokens` is unverified metadata that drives truncation; `CausalInfillingDataset` does not validate its layout, so an invalid one trains a whole run before generation rejects it; and `validate_layout` accepts a residue token as the terminator, which would stop decoding at that amino acid
- [ ] Record `span_layout_id` in `run_config.json` and compare it in the checkpoint-compatibility check — otherwise changing a backbone's layout silently reinterprets older checkpoints
- [ ] **A pretrained causal backbone** through `methods/ar/backbones.py` (registry has `scratch`, `iglm` and a generic `hf_decoder` fallback). **6 November 2026:** native IgLM / IgLM-S register as `iglm` ([MODELS.md](MODELS.md#pretrained-ar-backbone-iglm)): their layout, chain/species conditioning fields, sampler hook and output validation are in, replayed draw for draw against the upstream package on a GPU (150/150 frameworks), and the zero-shot, both fine-tuning sizes and random-init/size-matched control pilots are complete ([RESULTS.md](RESULTS.md)); the selected-checkpoint benchmark smoke is prepared but unrun. Still missing: a wired layout on the `hf_decoder` path (it still adds sentinels with random embeddings) and layout-aware extraction for any other native format. Format and checkpoint are separate questions — adopting IgLM's *layout* is clean, fine-tuning IgLM's *weights* is not, since it was pretrained on OAS and is the §11 baseline
- [ ] **Automate one GPU + real-judge smoke run.** Three defects reached live runs past a fully green CPU suite — device mismatch, prompt divergence, and the missing `pseudo_log_likelihood` import — because no test exercised the full GPU training-to-generation path with a real judge. The 2026-09-07 `s2s` landing ran this by hand and caught exactly those three; that is the argument for automating it, not a substitute

**Encoder-decoder method**

**Done.** `s2s` implemented end to end 2026-09-07 (two-stream data view, bidirectional encoder + cross-attending decoder, backbone registry, training, decode-to-`<eos>` sampling, shared-evaluation strategy, CLI, entrypoint, configs, `s2s_heavy` suite, `compare` extended to three arms; 44 new tests), with the full path verified outside the suite on GPU against the real IgBert judge (`valid_generation_rate` 0.975, `judge-cov` 1.000). Scratch S2S LR screen complete ([20260907_204721_sweep_lr](../results/runs/20260907_204721_sweep_lr/summary.tsv) — 3 LRs × 2 seeds; settings not promoted into the scratch configs), as is the three-arm [`compare` suite](../results/runs/20260908_173143_compare/batch_tables.md) on OAS4 validation with length-free generation for all three methods. Pretrained encoders landed 2026-09-10: ESM-2, IgBert and ProtBert feed the shared residue decoder through a learned width projection, with separate encoder tokenizers and `encoder_init`/`freeze_encoder` controls — implementation capability, not evidence of improved generation. The [full PLM matrix](../results/runs/20260913_151152_463868_s2s_plm/batch_tables.md) completed 2026-09-13 (six runs, 18,000 complete candidates, full judge coverage), and the IgBert + partner [collapse was diagnosed](../results/misc/diagnostics/20260914_igbert/REPORT.md) 2026-09-14.

- [x] **Complete the three-arm IgBert diagnostic** (`20260914_140305_226060_igbert_diagnostic`). All arms finished three epochs, with 49 finite probe records and three epoch checkpoints each. The original-LR replay did not reproduce the earlier collapse within this window. Epoch-3 validation loss / core recovery: original 1.6092 / 0.3836; lower LR 1.6954 / 0.3566; frozen 1.6185 / 0.3814. All retained conditioning. This establishes neither failure incidence nor a preventive effect of freezing/lower LR. See the [completed diagnostic report](../results/misc/diagnostics/20260915_igbert_summary/REPORT.md).
- [ ] **Evaluate the three saved diagnostic checkpoints generatively.** Same validation examples and 150 × 20 generation protocol, with fixed partner context for generator and judge; no retraining. [Commands](../configs/experiments/20260915_context_dropout.md#evaluate-the-completed-igbert-diagnostic-checkpoints-first).
- [x] **Run and review the AR/S2S context-dropout screen.** All 12 trials completed, with both evaluation regimes and fixed judge context. Dropout 0.5 improves missing-partner edit similarity in both seeds for every model, at a small partner-present cost (§14; [September 15–17 review](../results/misc/reviews/0915-0917-review.md)). All six no-partner training controls are complete ([October 7 review](../results/reviews/20261007-no-partner-controls.md)), and the selected ESM-2 contrast was audited at seeds 44/45 ([confirmation review](../results/reviews/20261007-esm2-confirmation.md)). Broader stability/uncertainty work remains pending.
- [x] **Preserve trial identity in batch tables.** Fixed 2026-10-06: new tables and dashboards retain every seed/dropout trial and show generator/judge context separately. Artifact-only regression against the completed context batch discovers all 12 runs and both evaluation regimes. Historical reports remain as originally saved.
- [x] **Complete six models trained without partner context.** Scratch AR, scratch S2S and ESM-2 S2S × seeds 42/43, with `include_partner_chain: false` and `context_dropout: 0`. Match the current screen within each model: OAS4 H3 validation split, optimizer, batch size, 20-epoch maximum, early stopping and checkpoint-selection metric. Evaluate the same 150 × 20 examples/candidates without generator partner context and with fixed partner context for the judge. Compare against partner-trained dropout 0 and 0.5 models; removing the partner only at inference does not replace this training control. The [six-trial spec](../configs/experiments/20261006_no_partner_controls.json) completed 2026-10-06: 6/6 trials, 18,000 valid candidates and full judge coverage. The two initial TensorBoard failures have successful AR retries; no rerun is needed. Comparisons to September retain source/runtime differences. See the [completion review](../results/reviews/20261007-no-partner-controls.md) and [next-batch commands](../configs/experiments/20261006_h3_followup.md#commands-for-the-user).
- [x] **Confirm the selected ESM-2 dropout contrast at seeds 44/45.** All four trials in the [spec](../configs/experiments/20261006_esm2_dropout_confirmation.json) passed the [completion audit](../results/reviews/20261007-esm2-confirmation.md), including both generator context regimes and recovered-run checkpoint provenance. Missing-partner edit gains are +0.0838/+0.0616; partner-present costs are −0.0042/−0.0039. Keep seed/source/runtime differences visible and the test split untouched. Seed 46, fixed-seed repeats and cluster-aware uncertainty remain follow-ups; this checkbox does not complete the broader stability or final-evaluation study.
- [x] **Complete the two scratch MLM no-partner controls.** Seeds 42/43 passed the [artifact audit](../results/reviews/20261007-mlm-no-partner-controls.md): fitted length heads, selected weights, the same 150 × 20 returned draws and partner-present judge. No completion rerun is needed; MLM attempt validity remains unrecorded.
- [x] **Complete and review the four matched MLM dropout arms.** All four partner-trained trials and eight evaluations passed the [artifact audit](../results/reviews/20261007-mlm-dropout-screen.md): 24,000 returned candidates, inferred lengths and fixed partner-present judge. Missing-partner edit gains are +0.2332/+0.2526; partner-present changes are −0.0017/+0.0013. Source edits and post-completion command refactoring are documented. No completion rerun is needed; uncertainty, further seeds/repeats and final testing remain open. See the [completed recipe](../configs/experiments/20261007_mlm_context.md).
- [ ] **Separate ESM-2 pretraining from capacity and adaptation.** After reviewing the context screen, compare trainable pretrained, trainable randomly initialized, and frozen pretrained encoders of the same ESM-2 architecture, initially at seeds 42/43. Hold the decoder, corpus, context policy and evaluation protocol fixed; reuse current runs only where settings match. Use `encoder_init`/`freeze_encoder` in a JSON spec. Allow comparable validation tuning effort, since random initialization may require a different learning rate. Report generation quality, validity and contributing populations alongside teacher-forced recovery.
- [x] **Screen ESM-2 learning rates on a longer budget.** Four trials at `1e-5`/`1e-4`, seeds 42/43, up to 20 epochs, completed and reviewed against the `3e-5` context-screen reference (§15; [September 15–17 review](../results/misc/reviews/0915-0917-review.md)). All use dropout 0 and partner-present generation. `1e-4` is a candidate for confirmation, not a promoted default; its dropout interaction and partner-absent behavior remain unmeasured.
- [ ] **Compare tuned scratch and PLM systems under a matched protocol.** Match corpus, split, evaluation and comparable tuning effort; state epoch and compute budgets. The initial matrix's judge-perplexity gap (PLMs 3.441–4.132 versus scratch 3.215/3.250) is historical; the new partner-present ESM-2 `1e-4` mean is 3.345 (§15), and differing capacities and architectures still confound attribution to pretraining. Retain the same-architecture pretrained-versus-random control
- [ ] Calibrate conditioning criteria for the exact 16-example encoder-memory probe and fix them before a confirmation run. This batch began without a numerical pass/fail criterion, so thresholds developed from its own traces are exploratory; the earlier 256-example raw-framework-swap measurements are illustrative references, not calibrated callback thresholds
- [ ] **A full pretrained encoder-decoder backbone**: ProtT5 remains a future adapter. The released checkpoint used BART-like residue masking, **not** original T5 span denoising; validate its real pretraining/input/output contract before designing missing-span extraction ([model card](https://huggingface.co/Rostlab/prot_t5_xl_uniref50))
<!-- AI ACCESS ZONE ENDS -->

- [ ] Project flowchart (0814, outdated)

**Datasets**
- OAS
  - [x] 21,106 paired sequences (`1287169_1_Paired_All.csv`; pre-2026-09-03)
  - [x] OAS4: 152,727 paired sequences (8 CSVs, 4 donors; Naive-B-Cells)
  - [ ] Expand paired data after auditing available donors, B-cell types and attainable scale
  - [ ] Unpaired sequences
- SAbDab2
  - [x] 3,984 paired antibodies with resolved antigen
  - [ ] Inventory and process remaining structures without antigen
- Other sources
  - [ ] SAbDab / Thera-SAbDab

**Preprocessing**
- [x] Bio-Object JSON schema (14 IMGT regions)
- [x] Partner-chain context (light <-> heavy)
- [x] Antigen context (SAbDab2)
- [x] Clustered split (H3) (prevents data leakage)
- [ ] Structure / coordinate parsing
- [ ] Unpaired-OAS preprocessor

**Training / methods**
- Sequence-only
  - General
    - [x] Context dropout (partner chain, antigen)
  - [x] Masked-LM infilling
    - [x] Fixed-native generation
    - [x] Sampling strategies: independent vs iterative
    - [x] Bounded/inferred generation plumbing and checkpoint artifacts
    - [x] Predicts length before infilling residues
    - Models
      - [x] Custom transformer
      - [x] ESM-2
      - [x] IgBert
      - [x] ProtBert
      - [ ] AbLang2
  - [x] Autoregressive infilling
    - Models
      - [x] Custom transformer
  - [x] Encoder-decoder infilling
    - [x] Frozen-encoder and same-architecture random-initialization controls
    - [x] Optional conditioning diagnostics and early stopping that preserves the LR schedule
    - [ ] Establish stable IgBert + partner training with diagnostic controls
    - [ ] Tune and replicate promising settings; compare pretrained and random initialization
    - [ ] Full pretrained encoder-decoder adapter
    - Models
      - [x] Custom transformer
      - [x] ESM-2
      - [x] IgBert
      - [x] ProtBert
- Structure-aware
  - [ ] Invariant Point Attention (IPA)
  - [ ] E(n)-equivariant GNNs
  - [ ] Diffusion (dyMEAN / DiffAb-informed)
  - [ ] Antigen cross-attention / epitope-contact signal

**Evaluation**
- [x] Sequence recovery (AAR) — whole-span, core and anchors
- [x] Self perplexity / pseudo-perplexity and shared judge scoring
- [x] Diversity — pairwise Hamming, unique fraction, per-position entropy
- [x] Novelty — uniqueness versus training set
- [x] Length-free metrics — validity, length match/MAE, edit similarity, conditional length-matched AAR
- [x] Coverage and contributing populations reported alongside conditional recovery
- [ ] Paired bootstrap confidence intervals for the final matched comparisons
- [ ] Structure metrics — RMSD, TM-score

**Benchmark**
- [x] IgLM external benchmark adapter
- [x] Matched fixed-length CDRSeqGen/IgLM validation smoke test
- [ ] Matched length-free CDRSeqGen/IgLM score and plot
- [ ] Final held-out evaluation after model and protocol selection

**Long-term objectives / deliverables**
- Leakage-audited benchmark for CDR-infilling methods
- Structure-conditioned CDR-H3 design
- Integrated sequence + structure design framework
- Benchmark paper and/or a novel antibody-design method

<!-- MANUAL PROTECTION ZONE ENDS -->
