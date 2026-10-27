# Tests worth keeping

The suite protects behavior that could silently change an experiment or waste a training run. It is not a checklist proving the research conclusions. Tests run with synthetic data on CPU; small scratch models and tiny local HF ESM/BERT checkpoints exercise real attention, training, serialization, and shell orchestration without downloading weights.

From the repository root:

```bash
./scripts/unit_tests.sh                       # complete suite
./scripts/unit_tests.sh test_model_roundtrip  # real CPU update/save/reload, both methods
./scripts/unit_tests.sh test_s2s_plm         # real tiny ESM/BERT encoder + decoder contracts
./scripts/unit_tests.sh test_runner           # suite/sweep plans, execution, failures, and resume
./scripts/unit_tests.sh 'test_*length*'        # length policies and evaluation
```

Use the project Python environment (`conda activate cdrprj` if needed). No GPU, downloaded checkpoint, OAS/SAbDab2 corpus, network access, or active Conda installation is required by the test fixtures. The runner sets `PYTHONPATH` and changes to the repository root. Runner tests select a test interpreter and replace training/reporting boundaries, leaving both shell entry points and the common planner/executor intact; their files stay in temporary directories.

## Scientific guarantees and where to look

| What must not silently change | Main coverage |
|---|---|
| Root commands reach each method through one main(), while compatibility, context, case order and output checks survive delegation | `test_command_dispatch`, `test_inference_entrypoints`, `test_method_integration`, `test_evaluate_contract` |
| Native residues or native length must not leak into length-free generation | `test_variable_length_eval`, `test_length_policy`, `test_dataset_ar`, `test_span_layout` |
| Future answer tokens and padding must not change earlier causal predictions | `test_ar_model` (real attention, including the padded branch) |
| Prompt layout, EOS, and answer-only supervision agree between training and generation | `test_dataset_ar`, `test_span_layout`, `test_ar_sampling`, `test_ar_training` |
| Training/evaluation select explicitly named partitions | `test_eval_split_selection`, `test_ar_training` |
| A decoder name is never silently replaced by another decoder, and sampling runs without autograd | `test_shared_contracts` (training config, standalone evaluation/generation and the sampler dispatcher) |
| The public flags of `evaluate.py`/`generate.py` cannot change silently, and flags shared with training keep the same defaults | `test_shared_contracts` (flag sets, one shared definition, mirrored `gen_*` defaults including the 150-example budget, either spelling of the checkpoint flag, bare `main()` parsing `sys.argv`) |
| A config the dataset or model would reject cannot leave a run directory behind, a valid one is not rejected, and every shipped config still passes its method's setup | `test_arguments_contracts` (all three methods' `setup_args()` against a mocked `initialize_run`: unknown loop or backbone, heads that cannot split the width, early stopping that cannot engage; pretrained backbones exempt from the scratch rules; list/dict config values with trailing overrides; every shipped training config and experiment-spec trial; help text and `Optional` declarations) |
| PLM encoder/decoder token ids, pretrained weights, gradients, frozen dropout, and offline reload agree | `test_s2s_plm` (real local ESM/BERT models and Trainer) |
| Model weights actually update, then survive serialization | `test_model_roundtrip` (real dataset/collator/model/optimizer/save/reload for MLM and AR) |
| Failed, empty, wrong-length, or unscored draws cannot silently improve a metric | `test_ar_evaluation`, `test_metrics`, `test_confidence_metrics`, `test_sweep_tables`, benchmark scorer/track tests |
| A matched batch receives its intended corpus, length policy, sampling settings, and devices | `test_runner` (legacy/named commands, plan equivalence, sequential/sharded execution, explicit grids, dataset gates, frozen-plan resume, artifact verification and active locks) |
| Results preserve data/checkpoint provenance and resume compatibility | `test_preprocess_provenance`, `test_artifacts`, adapter and manifest tests |
| Run logs retain Trainer events/exceptions, latest preserves real directories, and epoch CSVs retain the training trajectory | `test_run_recording`, `test_artifacts` (including the provenance checkout root) |
| Missing values are not plotted as zero; tables state actual length policy and recovery coverage | `test_plotting`, `test_batch_tables`, `test_benchmark_plotter` |

Tests in other files cover related tokenizers, losses, configuration parsing, statistical calculations, and artifact handling. The table is a reading guide, not a complete test inventory.

## What passing does not establish

- **No sequence leakage.** Selecting disjoint named partitions is different from verifying cross-partition similarity. The existing greedy representative clustering does not guarantee pairwise separation at 85% identity. The September audit found 3/150 affected OAS4 H3 evaluation cases. This remains an open data-protocol issue; it is not hidden behind an expected-failure test or asserted to be safe. See [the experiment review](../results/misc/reviews/0903-0904-review.md) and its data-dependent audit script (`audit_0903_0904.py`, same folder).
- **Real pretrained-model/GPU correctness.** Tiny CPU model tests do not establish correctness for every published checkpoint, device/dtype interaction, or a live IgBert judge. Run a small full integration experiment before a costly batch.
- **Learning quality or statistical validity.** A real optimizer step checks plumbing, not convergence. A correct arithmetic mean may still describe a biased population. Training-seed replication and matched experimental design remain necessary.
- **External execution boundaries.** `test_runner` executes both launchers and the common suite/sweep lifecycle, with training and reporting replaced. Real Conda activation, pretrained training, and GPU execution still need integration checks.

## How to maintain the suite

Keep a test when you can name a plausible consequential failure it would catch. For example, perturb a future token and check earlier logits, use a deliberately mismatched candidate population, or change the RNG seed before reloading weights. Test the behavior through the relevant public boundary when practical.

Use mocks at expensive/external boundaries, with small deterministic fixtures to isolate scoring and edge cases. Keep at least one real path through internal components; mocking every layer only proves the mocks agree. Temporary directories should clean up on assertion failures too.

Do not pin source-code spelling, internal parameter names, exact config-file counts, palette choices, or historical suite names unless those are themselves an intentional public contract. Do not delete useful numerical/reporting tests just to lower the count. Prefer a few meaningful scenarios and subtests over duplicate assertions.

The September 7 curation replaced 12 config/source-text checks with 5 behavioral batch-runner tests, removed an API-signature test already covered by layout behavior, consolidated two AR integration tests into two real MLM/AR optimizer-and-reload tests, and added two reporting regressions. The reporting regressions accompany a fix: generated design tables now state the saved length policy and number of frameworks contributing recovery instead of assuming MLM always knows native length. Existing result artifacts were not regenerated.
