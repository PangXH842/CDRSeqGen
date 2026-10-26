# Project status and next work

[Back to README](../README.md) · [Results index](RESULTS.md) · [Research notes](FINDINGS.md) ·
[Roadmap](ROADMAP.md) · [Operation Reclamation](RECLAMATION.md)

This guide carries forward the **23 October 2026 handoff**, reorganized on 26 October.
Submission status is user-confirmed; dated audit and artifact notes retain their original
scope. This documentation cleanup did not rerun experiments or regenerate manuscript artifacts.

## Current handoff — 2026-10-23

The [completed evidence inventory](RESULTS.md#completed-validation-evidence) links the audited batches. Detailed measurements and historical caveats live in [FINDINGS.md](FINDINGS.md).

**What the paper can now show.** The framework makes allowed inputs, length policy, attempt accounting and scoring populations explicit across methods. The expanded results add a useful reference: primary flank retrieval reaches **0.3589** edit similarity versus **0.2817** for unconditional sampling. Seven-seed scratch means are **0.3918 MLM**, **0.3868 AR** and **0.3882 S2S**; IgLM scores **0.3880** and IgCraft **0.3769**. See [§17](FINDINGS.md#17-expanded-validation-evidence-seeds-retrieval-and-capacity) for the completed analyses and limits. The earlier eight-arm table in §16 is the original subset.

**Abstract submitted; writing and presentation work:**

The abstract was submitted on **23 October**, as confirmed by the user. The [canonical abstract](../../iAIM/sections/0_abstract.tex), [PDF](../../iAIM/abstract.pdf) and [plain-text copy](../../iAIM/abstract_submission.txt) were refreshed for submission; check their contents together before further distribution. The [coauthor review kit](../../iAIM/COAUTHOR_HANDOFF.md), prepared on October 22, provides a [five-slide overview](../../iAIM/coauthor_brief.pdf), [editable slides](../../iAIM/coauthor_brief.tex) and [email introduction](../../iAIM/coauthor_email.md). Its dated notes and archives precede the final abstract edits.

1. Keep subsequent abstract revisions in the canonical source shared by both LaTeX entry points, and refresh the PDF/plain-text copy together. `abstract_v2.txt` is historical. The [paper outline](../../iAIM/PAPER_OUTLINE.md) maps sections, displays, evidence and remaining work.
2. Bring the manuscript's methods, results, figures and discussion up to the verified October 9–12 evidence. Centre the unified framework; use retrieval, variability and context dropout as empirical demonstrations.
3. Explain the shared workflow and extension contract, expand related work, and obtain advisor/coauthor review. Retain exploratory status, scoring populations, provenance and limitations. Prepare the poster from the resulting evidence narrative.
4. Review the rebuilt [main.pdf](../../iAIM/main.pdf): Table 8 and Figure 5 include MLM ESM-2, while the main external-comparison tables still need the expanded evidence integrated. The [October 22 working archive](../../iAIM/iaim-draft-20261022.zip) and [October 8 archive](../../iAIM/iaim-draft-20261008.zip) are dated snapshots; regenerate a distribution archive after the manuscript updates.

**Remaining experiment decision — final held-out evaluation.** Freeze the cohort, participating systems, checkpoint choices, decoding settings, primary comparisons and reporting before final inference. The 14,818 audit-eligible records are a candidate pool, not a frozen final manifest. Decide common length support, including IgCraft's 30-slot H3 support, and disclose unknown external pretraining membership. The current batch runner still selects validation; the final path needs preparation before execution. Further training follows a specific retained claim or necessary split revision, rather than another broad sweep.

The current matched OAS4 benchmark has used validation; historical test-selection caveats remain documented below. All current contrasts are exploratory. The newer analysis resamples training seeds and framework groups, while older intervals fix checkpoints; candidate-generation variability is not estimated by either. The seven-seed study covers no-partner scratch checkpoints, **not seven-seed dropout experiments**. Historical MLM training evaluations lack attempt-validity accounting; the benchmark adapters record every attempt. Different-length/gapped homology and [external pretraining membership](../benchmark/pretraining_overlap.md) remain unresolved.

**Manual code review — continue with MLM generation.** The [manual checklist](RECLAMATION.md) now also marks [eval_cli](../cdrseqgen/methods/mlm/eval_cli.py) and [evaluation](../cdrseqgen/methods/mlm/evaluation.py) reclaimed on **23 October**. Their docstrings map CLI setup, framework selection, generation strategies, shared scoring and reporting. Continue with [gen_cli](../cdrseqgen/methods/mlm/gen_cli.py) → [generation.run_generation](../cdrseqgen/methods/mlm/generation.py), then `case_generation`, `sampling`, `length`, `length_predictor` and `variable_length_evaluation`. The shared masked-span dispatcher now lives in `mlm/sampling.py` and serves generation and evaluation.

```text
generate.py → mlm/gen_cli.main() → mlm/generation.run_generation()
evaluate.py → mlm/eval_cli.main() → common/evaluation.run_evaluation_pipeline()
                               → mlm/evaluation.run_generative_eval()
```

Both workflows call shared loading/setup lifecycles in `methods/common/`. Follow those dependencies while reading, then compare method implementations through MLM → AR → S2S → common using the [workflow map](../cdrseqgen/methods/common/README.md). Shared code remains pending in the manual checklist; passing tests and completed refactors do not mark it reclaimed. The manual checkboxes remain the record of your own review.

**Reading older notes:** dated recipes, the baseline-scope memo and the results index contain preparation-time statements that precede these completions. Use this handoff, §17 and the linked saved outputs for current run status. No new broad training batch is needed before drafting. Experiment execution remains user-run.

Conference milestones: **abstract submitted 23 October** (deadline: 1 November); **poster presentations 4–5 December**. Acceptance and presentation instructions remain subject to the conference's notification.
