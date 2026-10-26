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
| [Results](RESULTS.md) | Completed validation evidence and source artifacts |
| [Findings](FINDINGS.md) | Detailed measurements, historical findings and limitations |
| [Project status](PROJECT_STATUS.md) | Current handoff, manuscript work and final-evaluation decisions |
| [Roadmap](ROADMAP.md) | Research milestones, experiment options and feature plans |
| [Operation Reclamation](RECLAMATION.md) | Manual code-review checklist and review marks |
| [Changelog](CHANGELOG.md) | Detailed AI-assisted implementation and audit history |

The protection markers and AI access zones in the roadmap and reclamation checklist
remain in force. Checkmarks record manual review; passing tests or moving files does
not complete a review item.

For implementation-specific guidance, see the [method workflow map](../cdrseqgen/methods/common/README.md),
[experiment guide](../configs/experiments/README.md), [benchmark guide](../benchmark/README.md)
and [test guide](../unit_tests/README.md).
