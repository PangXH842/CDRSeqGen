# Project status and next work

[Back to README](../README.md) · [Results index](RESULTS.md) · [Research notes](FINDINGS.md) ·
[Roadmap](ROADMAP.md)

Updated **6 November 2026** for code, IgLM-run and poster progress. Research evidence and
manuscript tasks carry forward the **23 October handoff**; their dated audit and artifact
notes retain their original scope. No experiments or manuscript artifacts were regenerated.

<a id="current-handoff--2026-10-23"></a>

## Current handoff — 2026-11-06

The [completed evidence inventory](RESULTS.md#completed-validation-evidence) links the audited batches. Detailed measurements and historical caveats live in [FINDINGS.md](FINDINGS.md).

**What the paper can now show.** The framework makes allowed inputs, length policy, attempt accounting and scoring populations explicit across methods. The expanded results add a useful reference: primary flank retrieval reaches **0.3589** edit similarity versus **0.2817** for unconditional sampling. Seven-seed scratch means are **0.3918 MLM**, **0.3868 AR** and **0.3882 S2S**; IgLM scores **0.3880** and IgCraft **0.3769**. See [§17](FINDINGS.md#17-expanded-validation-evidence-seeds-retrieval-and-capacity) for the completed analyses and limits. The earlier eight-arm table in §16 is the original subset.

**Abstract accepted for an in-person poster; writing and presentation work:**

The abstract was submitted on **23 October**, as confirmed by the user. The [canonical abstract](../../iAIM/sections/0_abstract.tex), [PDF](../../iAIM/abstract.pdf) and [plain-text copy](../../iAIM/abstract_submission.txt) were refreshed for submission; check their contents together before further distribution. The [coauthor review kit](../../iAIM/COAUTHOR_HANDOFF.md), prepared on October 22, provides a [five-slide overview](../../iAIM/coauthor_brief.pdf), [editable slides](../../iAIM/coauthor_brief.tex) and [email introduction](../../iAIM/coauthor_email.md). Its dated notes and archives precede the final abstract edits.

1. Keep subsequent abstract revisions in the canonical source shared by both LaTeX entry points, and refresh the PDF/plain-text copy together. `abstract_v2.txt` is historical. The [paper outline](../../iAIM/PAPER_OUTLINE.md) maps sections, displays, evidence and remaining work.
2. Bring the manuscript's methods, results, figures and discussion up to the verified October 9–12 evidence. Centre the unified framework; use retrieval, variability and context dropout as empirical demonstrations.
3. Explain the shared workflow and extension contract, expand related work, and obtain advisor/coauthor review. Retain exploratory status, scoring populations, provenance and limitations. Prepare the poster from the resulting evidence narrative.
4. Review the rebuilt [main.pdf](../../iAIM/main.pdf): Table 8 and Figure 5 include MLM ESM-2, while the main external-comparison tables still need the expanded evidence integrated. The [October 22 working archive](../../iAIM/iaim-draft-20261022.zip) and [October 8 archive](../../iAIM/iaim-draft-20261008.zip) are dated snapshots; regenerate a distribution archive after the manuscript updates.

**Remaining experiment decision — final held-out evaluation.** Freeze the cohort, participating systems, checkpoint choices, decoding settings, primary comparisons and reporting before final inference. The 14,818 audit-eligible records are a candidate pool, not a frozen final manifest. Decide common length support, including IgCraft's 30-slot H3 support, and disclose unknown external pretraining membership. The current batch runner still selects validation; the final path needs preparation before execution. Further training follows a specific retained claim or necessary split revision, rather than another broad sweep.

**IgLM adaptation update:** the full GPU replay, both zero-shot evaluations, the fresh four-trial
IgLM-S fine-tuning batch and the four-trial control batch are complete and audited
([fine-tuning](../results/reviews/20261103-iglm-adaptation.md), [controls](../results/reviews/20261105-iglm-controls.md)).
All 12,000 fine-tuned and 12,000 control candidates are valid and judge-scored; exported weights match
the selected checkpoints. The validation-core criterion selects lr 3e-5, epoch 11 (core 0.407213;
edit similarity 0.389715 versus zero-shot IgLM-S 0.385881). The random-init IgLM-S controls reach core
0.400739–0.406348, all at the 20-epoch cap with convergence unestablished, and a scratch AR of the same size 0.403734;
at the highest learning rate random-init matches the pretrained run on core accuracy.
Both [larger-IgLM trials](../results/reviews/20261106-iglm-large.md) are also complete and audited:
6,000 valid, judge-scored candidates; validation core selects lr 1e-5, epoch 4 (core 0.409725,
edit similarity 0.386728). No clear generated-edit improvement over the selected IgLM-S is shown.
All adaptation training recipes are complete. The two-checkpoint benchmark smoke is prepared
and its CPU plan passes, but GPU execution remains pending
(see the [adaptation protocol](../configs/experiments/20261102_iglm_adaptation.md)).
Keep these seed-42, partner-present-judge pilot results separate from the frozen external benchmark.
Earlier CPU parity diagnostics used some test-split records. Document that development exposure
when freezing the final evaluation; no final comparative held-out run has occurred.

The current matched OAS4 benchmark has used validation; historical test-selection caveats remain documented below. All current contrasts are exploratory. The newer analysis resamples training seeds and framework groups, while older intervals fix checkpoints; candidate-generation variability is not estimated by either. The seven-seed study covers no-partner scratch checkpoints, **not seven-seed dropout experiments**. Historical MLM training evaluations lack attempt-validity accounting; the benchmark adapters record every attempt. Different-length/gapped homology and [external pretraining membership](../benchmark/pretraining_overlap.md) remain unresolved.

**Manual code review — finish the shared training helpers, then AR/S2S evaluation and generation.**
The MLM, AR and S2S training blocks and the root [train.py](../train.py) dispatcher are reviewed.
Continue with the shared modules they call: [logging](../cdrseqgen/methods/common/logging.py),
[run setup](../cdrseqgen/methods/common/run_setup.py), [callbacks](../cdrseqgen/methods/common/callbacks.py),
[losses](../cdrseqgen/methods/common/losses.py) and the AR/S2S
[decoding module](../cdrseqgen/methods/common/eos_decoding.py). Then review AR's evaluation, generation
and sampling blocks and the native [IgLM model](../cdrseqgen/methods/ar/models/iglm.py), compare S2S, and
revisit shared evaluation and sampling.

**Current shared implementation:**

- All three methods wire `prepare_training()` and `finish_training()` into
  `common.training.run_training_pipeline()`; standalone evaluation/generation
  likewise use shared lifecycles.
- `common.arguments` owns base fields, config/CLI merging and shared validation.
  Explicit overrides preserve unrelated JSON nulls, lists and dictionaries.
  See the [argument reference](ARGUMENTS.md).
- `common.backbones` owns `Backbone`, `BackboneRegistry` and loading helpers.
  Method registries remain separate; AR extends tokenizer construction for its
  span-layout checks, while S2S owns encoder/decoder assembly.
- MLM's `length.py` owns policies, prompts, the fitted head, span sampling and
  evaluation strategies. Its `generation.py` includes per-case scoring; the old
  predictor/variable-evaluation modules and `case_generation.py` are retired.

```text
train.py    → method/train_cli.main() → common.training.run_training_pipeline()
                                    → method prepare_training() / finish_training()
generate.py → method/gen_cli.main()   → common.generation.run_generation_pipeline()
                                    → method/generation.generate_case()
evaluate.py → method/eval_cli.main()  → common.evaluation.run_evaluation_pipeline()
                                    → method/evaluation.run_generative_eval()
```

Use the [workflow map](../cdrseqgen/methods/common/README.md) to follow stage ownership.
The remaining shared logging, run-setup, callback, loss, decoding, evaluation and sampling
modules, and the training-artifact export in `visualization/`, still need manual review. Passing tests or completing a refactor
does not complete a manual review. Detailed personal review records are kept locally.

**Reading older notes:** dated recipes, the baseline-scope memo and the results index contain preparation-time statements that precede these completions. Use this handoff, §17 and the linked saved outputs for current run status. No new broad training batch is needed before drafting. Experiment execution remains user-run.

Conference milestones: **abstract submitted 23 October and accepted for an in-person poster**, submission **9941**;
conference **3–5 December**, poster sessions **4–5 December**. The [poster template](../../iAIM/poster.tex)
and [preparation guide](../../iAIM/POSTER_GUIDE.md) are ready. Assigned session and poster specifications
remain pending; the acceptance email requires registration within two weeks of receipt.
