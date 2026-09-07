# Tests worth keeping

The suite protects behavior that could silently change an experiment or waste a training run. It is not a checklist proving the research conclusions. Tests run with synthetic data on CPU; external pretrained models are stubbed, while small scratch models, serialization, and shell orchestration are exercised directly.

From the repository root:

```bash
./scripts/unit_tests.sh                       # complete suite
./scripts/unit_tests.sh test_model_roundtrip  # real CPU update/save/reload, both methods
./scripts/unit_tests.sh test_runner           # batch launch settings and failures
./scripts/unit_tests.sh 'test_*length*'        # length policies and evaluation
```

Use the project Python environment (`conda activate cdrprj` if needed). No GPU, downloaded checkpoint, OAS/SAbDab2 corpus, network access, or active Conda installation is required by the test fixtures. The runner sets `PYTHONPATH` and changes to the repository root. Shell tests replace environment activation and expensive subprocesses, leaving the real runner and its JSON preflight checks intact; their files stay in temporary directories.

## Scientific guarantees and where to look

| What must not silently change | Main coverage |
|---|---|
| Native residues or native length must not leak into length-free generation | `test_variable_length_eval`, `test_length_policy`, `test_dataset_ar`, `test_span_layout` |
| Future answer tokens and padding must not change earlier causal predictions | `test_ar_model` (real attention, including the padded branch) |
| Prompt layout, EOS, and answer-only supervision agree between training and generation | `test_dataset_ar`, `test_span_layout`, `test_ar_sampling`, `test_ar_training` |
| Training/evaluation select explicitly named partitions | `test_eval_split_selection`, `test_ar_training` |
| Model weights actually update, then survive serialization | `test_model_roundtrip` (real dataset/collator/model/optimizer/save/reload for MLM and AR) |
| Failed, empty, wrong-length, or unscored draws cannot silently improve a metric | `test_ar_evaluation`, `test_metrics`, `test_confidence_metrics`, `test_sweep_tables`, benchmark scorer/track tests |
| A matched batch receives its intended corpus, length policy, sampling settings, and devices | `test_runner` (actual sequential and sharded shell execution) |
| Results preserve data/checkpoint provenance and resume compatibility | `test_preprocess_provenance`, `test_artifacts`, adapter and manifest tests |
| Missing values are not plotted as zero; tables state actual length policy and recovery coverage | `test_plotting`, `test_batch_tables`, `test_benchmark_plotter` |

Tests in other files cover related tokenizers, losses, configuration parsing, statistical calculations, and artifact handling. The table is a reading guide, not a complete test inventory.

## What passing does not establish

- **No sequence leakage.** Selecting disjoint named partitions is different from verifying cross-partition similarity. The existing greedy representative clustering does not guarantee pairwise separation at 85% identity. The September audit found 3/150 affected OAS4 H3 evaluation cases. This remains an open data-protocol issue; it is not hidden behind an expected-failure test or asserted to be safe. See [the experiment review](../results/reviews/0903-0904-review.md) and its data-dependent audit script.
- **Real pretrained-model/GPU correctness.** CPU scratch tests do not cover external tokenizer quirks, downloaded weights, device/dtype interactions, or a live IgBert judge. Run a small full integration experiment before a costly batch.
- **Learning quality or statistical validity.** A real optimizer step checks plumbing, not convergence. A correct arithmetic mean may still describe a biased population. Training-seed replication and matched experimental design remain necessary.
- **Complete shell automation coverage.** `test_runner` covers `run.sh`; it does not execute the full `sweep.sh` workflow, real Conda activation, training, or plotting subprocesses.

## How to maintain the suite

Keep a test when you can name a plausible consequential failure it would catch. For example, perturb a future token and check earlier logits, use a deliberately mismatched candidate population, or change the RNG seed before reloading weights. Test the behavior through the relevant public boundary when practical.

Use mocks at expensive/external boundaries, with small deterministic fixtures to isolate scoring and edge cases. Keep at least one real path through internal components; mocking every layer only proves the mocks agree. Temporary directories should clean up on assertion failures too.

Do not pin source-code spelling, internal parameter names, exact config-file counts, palette choices, or historical suite names unless those are themselves an intentional public contract. Do not delete useful numerical/reporting tests just to lower the count. Prefer a few meaningful scenarios and subtests over duplicate assertions.

The September 7 curation replaced 12 config/source-text checks with 5 behavioral batch-runner tests, removed an API-signature test already covered by layout behavior, consolidated two AR integration tests into two real MLM/AR optimizer-and-reload tests, and added two reporting regressions. The reporting regressions accompany a fix: generated design tables now state the saved length policy and number of frameworks contributing recovery instead of assuming MLM always knows native length. Existing result artifacts were not regenerated.
