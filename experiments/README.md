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
| `specs/` | The experiment definitions (a byte-identical copy of `configs/experiments/`) |
| `tools/` | `queue_specs.py`, `make_iglm_randinit.py`, `iglm_replay.py`, `organize_results.py` |

The base training configs stay in `configs/{ar,mlm,s2s}`. Specs refer to them, and to `configs/experiments/bases/`,
by repository-relative path, so `configs/experiments` has to keep resolving (see below).

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

## Status: copy first

Both sets exist. Nothing under `cdrseqgen/`, `benchmark/`, `scripts/` or `configs/` was changed, so `run.sh`,
the tests and every earlier batch still use the old files, and the source identities are unchanged
(`experiments/` is outside both hashes). Retiring the old files is a separate step:

1. Delete the four moved scripts, `scripts/experiments.py`, `cdrseqgen/experiments.py` and `configs/experiments/`.
2. Make `configs/experiments` a symlink to `../experiments/specs`, so every spec path and doc link keeps working.
3. Reduce `scripts/run.sh` to `exec python -m experiments "$@"` and point `unit_tests/test_runner.py` and
   `test_experiments.py` at the new modules.

Step 1 changes `source_identity()` (it hashes every `.py` file under `cdrseqgen/`), so plans frozen before it
cannot be resumed afterwards. Finished batches are unaffected.

## How it was checked

- **Same plans.** 129 old-versus-new cases (every suite, sweep preset and spec file, error paths, status and resume
  on a frozen batch, the named CLI): 121 byte-identical, including every plan's trial ids and source hash; the other
  8 differ only in program name, spec path or help wrapping.
- **Same behaviour.** The 18 runner tests of `test_runner.py` pass unchanged against this package in the
  post-migration layout (symlinked `configs/experiments`, one-line `run.sh`).
- **Tools.** Each copy differs from its original only in the root depth and the spec paths; `queue_specs.py`
  regenerates the 12 weekend specs byte for byte, and the benchmark forwarding prints the same `--help` as before.
