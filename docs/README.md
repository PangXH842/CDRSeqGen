# Documentation

[Back to the project README](../README.md)

Project-wide guides live here. Run shell commands from the repository root unless
a guide explicitly says otherwise; moving a document does not change command paths.
READMEs for configs, tests, methods and benchmarks remain beside their implementations.

| Guide | Purpose |
|---|---|
| [Environment](ENVIRONMENT.md) | Dependencies, verification and server restoration |
| [Workflow](WORKFLOW.md) | Preprocessing, training, generation, evaluation and command history |
| [Architecture](ARCHITECTURE.md) | Repository map, data/model conventions and extension points |
| [Model specifications](MODELS.md) | Custom-model architecture, native IgLM backbone, defaults, benchmark settings and paper-reporting notes |
| [Model integration](MODEL_INTEGRATION.md) | Add model families, preserve native protocols and choose backbone/method/benchmark extension points |
| [Arguments](ARGUMENTS.md) | Training fields, defaults, validation and JSON/CLI override rules |
| [Results](RESULTS.md) | Completed validation evidence and source artifacts |
| [Findings](FINDINGS.md) | Detailed measurements, historical findings and limitations |
| [Project status](PROJECT_STATUS.md) | Current handoff, manuscript work and final-evaluation decisions |
| [Roadmap](ROADMAP.md) | Research milestones, experiment options and feature plans |
| [Changelog](CHANGELOG.md) | Detailed AI-assisted implementation and audit history |

The roadmap's protection markers and AI access zones remain in force. Personal manual-review
notes are kept locally and are not part of the published guides. Passing tests or moving
files does not itself complete a manual review.

For implementation-specific guidance, see the [method workflow map](../cdrseqgen/methods/common/README.md),
[experiment guide](../configs/experiments/README.md), [benchmark guide](../benchmark/README.md)
and [test guide](../unit_tests/README.md).
