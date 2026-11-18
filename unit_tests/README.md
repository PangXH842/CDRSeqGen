# Tests worth keeping

The suite protects behavior that could silently change an experiment or waste a training run. It is not a checklist proving the research conclusions. Tests run with synthetic data on CPU; small scratch models and tiny local HF ESM/BERT checkpoints exercise real attention, training, serialization, and shell orchestration without downloading weights.

From the repository root:

```bash
./scripts/unit_tests.sh                       # complete suite
./scripts/unit_tests.sh test_models  # model builders, backbone registries, flags, and real CPU update/save/reload
./scripts/unit_tests.sh test_s2s            # S2S model and sampling, real tiny ESM/BERT encoder + decoder contracts
./scripts/unit_tests.sh test_runner           # suite/sweep plans, execution, failures, and resume
./scripts/unit_tests.sh 'test_*length*'        # length policies and evaluation
```

Each `test_<topic>.py` groups what used to be several files. A banner comment at the head of every section names the original file and keeps its docstring; helper names that two files used differently carry their original file's name as a prefix.

Use the project Python environment (`conda activate cdrprj` if needed). No GPU, downloaded checkpoint, OAS/SAbDab2 corpus, network access, or active Conda installation is required by the test fixtures. The runner sets `PYTHONPATH` and changes to the repository root. Runner tests select a test interpreter and replace training/reporting boundaries, leaving both shell entry points and the common planner/executor intact; their files stay in temporary directories.

## Scientific guarantees and where to look

| What must not silently change | Main coverage |
|---|---|
| Root commands reach each method through one main(), while compatibility, context, case order and output checks survive delegation | `test_commands`, `test_training` |
| Native residues or native length must not leak into length-free generation | `test_length`, `test_datasets` |
| Future answer tokens and padding must not change earlier causal predictions | `test_ar` (real attention, including the padded branch) |
| Early-stopping callbacks receive the requested patience/threshold and are attached to the Trainer, AR/S2S prepare/train/save/finish run in process (scratch and IgLM, with layout recorded), run names/provenance/latest-link rules hold, the three methods report the same recovery for the same errors, and a framework whose every draw fails is still counted | `test_training` |
| Prompt layout, EOS, and answer-only supervision agree between training and generation | `test_datasets`, `test_ar` |
| Training/evaluation select explicitly named partitions | `test_evaluation`, `test_ar` |
| Native IgLM loads unchanged and speaks its own format: native ids/EOS/PAD, `[chain][species] left [MASK] right [SEP]` prompts, original-release versus converted-checkpoint recognition, fixed-attempt sampling with the upstream token restrictions, and chain/species conditioning from training through generation records | `test_iglm` (format, registry and sampler contract; release recognition, species names, generation records, recorded sampling protocol) |
| IgLM rejects incomplete weights and changed source contracts; generation defaults and RNG state cannot silently change its protocol; raw checkpoints route explicitly to AR and every failed attempt remains traceable | `test_iglm` (tiny GPT-2 weights, legacy buffers, Transformers tokenizer variants, configuration isolation, source checks, routing, malformed versus unterminated outputs) |
| The IgLM experiment specs differ from the completed scratch AR control only as declared, the random-init control is seeded and safely replaceable, and the replay harness rejects missing attempts as well as changed draws | `test_experiments`; `test_arguments_contracts` validates every spec trial whose local IgLM checkpoint exists (missing local sources are skipped, so use the explicit preflight in the adaptation protocol) |
| A pretrained AR/S2S checkpoint can be a local benchmark arm without changing any scratch plan or prompt, and its chain/species conditioning comes only from the manifest and the checkpoint | `test_benchmark_adapters` (prompts, pinned files, arm rules, recipe generation) |
| Backbone registries remain method-local, IDs are unique, fallback order is valid, and tokenizer loading preserves vocabulary provenance and I/O failures | `test_models` (dispatch, duplicate IDs, fallback rules, source selection and tokenizer errors; real local-model reloads), `test_ar`, `test_datasets` (AR vocabulary/layout), `test_s2s` |
| A decoder name is never silently replaced by another decoder, and sampling runs without autograd | `test_models` (training config, standalone evaluation/generation and the sampler dispatcher) |
| The public flags of `evaluate.py`/`generate.py` cannot change silently, and flags shared with training keep the same defaults | `test_models` (flag sets, one shared definition, mirrored `gen_*` defaults including the 150-example budget, either spelling of the checkpoint flag, bare `main()` parsing `sys.argv`) |
| A config the dataset or model would reject cannot leave a run directory behind, a valid one is not rejected, and every shipped config still passes its method's setup | `test_arguments_contracts` (all three methods' `setup_args()` against a mocked `initialize_run`: unknown loop or backbone, heads that cannot split the width, early stopping that cannot engage; pretrained backbones exempt from the scratch rules; JSON nulls/empty lists/dictionaries with explicit CLI overrides, boolean aliases and `.args` precedence; every shipped training config and experiment-spec trial; help text and `Optional` declarations; `docs/ARGUMENTS.md` lists exactly the argument fields) |
| PLM encoder/decoder token ids, pretrained weights, gradients, frozen dropout, and offline reload agree | `test_s2s` (real local ESM/BERT models and Trainer) |
| Splitting a batch into gradient-accumulation steps cannot rescale the AR or S2S gradient, and Hugging Face label smoothing cannot silently replace their own loss | `test_losses` (the Trainer's own `training_step` with real model code, and the `compute_loss_func` seam a selectable loss would use), `test_arguments_contracts` (smoothing rejected for AR/S2S, accepted for MLM) |
| Model weights actually update, then survive serialization | `test_models` (real dataset/collator/model/optimizer/save/reload for MLM, AR and S2S) |
| Custom-model builders and direct constructors reject invalid dimensions, and renamed custom backbones retain architecture validation | `test_models`, `test_arguments_contracts` (positive integer dimensions/depths, divisibility, rotary head width and family-owned validation) |
| Failed, empty, wrong-length, or unscored draws cannot silently improve a metric | `test_ar`, `test_metrics`, `test_reporting`, benchmark scorer/track tests |
| A matched batch receives its intended corpus, length policy, sampling settings, and devices | `test_runner` (legacy/named commands, plan equivalence, sequential/sharded execution, explicit grids, dataset gates, frozen-plan resume, artifact verification and active locks) |
| Results preserve data/checkpoint provenance and resume compatibility | `test_preprocess_provenance`, `test_run_files`, adapter and manifest tests |
| Run logs retain Trainer events/exceptions, latest preserves real directories, and epoch CSVs retain the training trajectory | `test_run_files` (including the provenance checkout root) |
| Missing values are not plotted as zero; tables state actual length policy and recovery coverage | `test_reporting`, `test_benchmark_analysis` |

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
