# experiments/

Everything for configuring and running experiments, in one place. Run it from the repository root:

```bash
python -m experiments list
python -m experiments plan mlm_no_partner_controls --gpus 0
python -m experiments run mlm_no_partner_controls --gpus 0
python -m experiments status results/runs/<batch>
python -m experiments resume results/runs/<batch> --gpus 0
python -m experiments 0,1 compare                 # legacy forms: <gpus> [suite ...] | sweep ... | spec FILE
python -m experiments benchmark followup plan     # forwarded to the benchmark tools
```

The first argument picks the interface, exactly as `scripts/run.sh` does. Recipe, suite and sweep usage
is described in [specs/README.md](specs/README.md).

## Layout

| Path | What it holds |
|---|---|
| `__main__.py` | The entry point: named commands, legacy commands, `benchmark` forwarding |
| `core.py` | Fingerprints, source identity, completion checks, frozen-plan reading, batch status |
| `spec.py` | Suites, sweep presets and spec files expanded into trials; the corpus gate |
| `plan.py` | Trials into a frozen plan; the re-check before a resume |
| `execute.py` | Lock, GPU shards, journal, resume, then the report |
| `report.py` | Tables and plots after a batch |
| `recipes.py` | Named recipes, plan display, refusal of hidden legacy overrides |
| `specs/` | The experiment definitions; `configs/experiments` is a link to this folder |
| `tools/` | `queue_specs.py`, `make_iglm_randinit.py`, `iglm_replay.py`, `organize_results.py` |

The base training configs stay in `configs/{ar,mlm,s2s}`. Specs refer to them, and to `configs/experiments/bases/`,
by repository-relative path, so the `configs/experiments` link has to stay (see below).

## Where each piece came from

| Before | Now |
|---|---|
| `cdrseqgen/experiments.py` | `core.py`, `spec.py`, `plan.py`, `execute.py`, `report.py`, and `legacy_main` in `__main__.py` |
| `scripts/experiments.py` | `core.py` (status), `plan.py` (resume check), `recipes.py`, and `named_main` in `__main__.py` |
| `configs/experiments/` | `specs/` |
| `scripts/{queue_specs,make_iglm_randinit,iglm_replay,organize_results}.py` | `tools/` (only the repository-root depth and the spec paths changed) |

The functions were moved, not rewritten. Of the 35 moved definitions, 30 are AST-identical to the originals; the
other five differ only in where the package is found, the spec folder, the `prog` shown in `--help`, and the two
`main` names.

## Status: the old files are gone

On 20 November 2026 the originals were deleted: `cdrseqgen/experiments.py`, `scripts/experiments.py`, the four
scripts now in `tools/`, and the real `configs/experiments/` folder. In their place:

- `configs/experiments` is a symlink to `../experiments/specs`. Keep it: every spec names its base config as
  `configs/experiments/bases/...`, and the older doc and command paths use it too. Python 3.11's `rglob` does not
  descend into symlinked folders, so code that lists specs has to read `experiments/specs` directly
  (`test_arguments_contracts` does).
- `scripts/run.sh` only selects the environment and runs `python -m experiments "$@"`; `sweep.sh` and
  `weekend_queue.sh` go through it, and the queue's preflight calls `tools/queue_specs.py`.
- `unit_tests/` import from `experiments.core`, `.spec` and `.plan`; `test_runner.py` builds its temporary
  repository with this package and the link.

Deleting `cdrseqgen/experiments.py` changed `source_identity()` (it hashes every `.py` file under `cdrseqgen/`),
so plans frozen before that day cannot be resumed; finished batches are unaffected. `experiments/` itself is
outside both source hashes.

## How it was checked

- **Same behaviour.** The old implementation's output for 129 cases (every suite, sweep preset, spec file and named
  recipe, error paths, `status` and `resume` on a frozen batch, `--help`) was captured before the deletion. After it,
  the same cases through `scripts/run.sh` and the `configs/experiments` link give 0 differences: 25 byte-identical, and
  104 equal once hashes and trial ids (which changed with the source hash), the program name, the spec folder and
  help wrapping are normalised. The 150 shipped configs and spec trials that `test_arguments_contracts` validates are
  the same set with the same contents, and the same 964 tests pass.
- **Before the deletion** the package matched the originals on 129 cases (121 byte-identical, 8 differing only in
  program name, spec path or help wrapping), 30 of 35 moved definitions were AST-identical (the other five differ only
  as listed above), and `queue_specs.py` regenerated the 12 weekend specs byte for byte.
