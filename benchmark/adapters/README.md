# Writing a benchmark adapter

An adapter turns a frozen manifest into candidate loops for one generator and writes them as JSON. The
scorer, plots and paired analysis read only that JSON, so a generator may live in any environment and
share no code with the rest of the repository. Five producers already meet this contract:
`cdrseqgen.py` (trained checkpoints), `iglm.py` and `igcraft.py` (frozen external models), `../empirical.py`
(training-only reference, no model) and `retrieval.py` (flank-keyed training reference: no parameters, so the
registry runs it in the main environment and the merge treats it as a reference pool rather than a generator). `igcraft.py` is the worked example for a generator that needs inputs the
manifest does not carry: it takes IMGT flank regions from a table built once by `../export_regions.py` (flanks
only, re-verified against `pt1`/`pt2`), rather than numbering strings or reading the corpus at run time.

## 1. What the runner passes

`run.py` starts `PYTHON ADAPTER.py` with these arguments, then the adapter's own (`output`, model choice, caps):

    --manifest PATH --num-samples N --temperature T --top-p P --seed S --device cuda [--resume]

`--resume` is added when the output exists; the adapter must continue it and refuse to if the model,
sampling or manifest changed (`assert_resume_compatible` in `_common.py`). Derive each framework's seed with
`stable_example_seed(seed, example_id)` so a resumed run reproduces the draws it would have made.

## 2. What it may read (information budget)

A manifest record carries `pt1`, `native`, `pt2`, `native_length`, `infill_range`, `parent_sequence`,
identifiers and any conditioning tokens the manifest declares. The adapter must echo `native` and
`native_length` for the scorer. `parent_sequence` contains the native loop, so an adapter that builds its
prompt from it (as `iglm.py` does) must mask the span at `infill_range` before the model sees it.

- **length_free track:** read `pt1`, `pt2` and declared conditioning only. Do not read the native loop, its
  length, or look the record up in a corpus by `source_index` / `sequence_id`.
- **fixed_native_length track:** the native length is given by design and the retention rules below change.

`blind_check.py` tests this rather than trusting it: it runs the adapter under identical seeds on the
manifest, on the manifest again, and on a copy with different hidden fields, and compares the candidates.
It cannot see a lookup keyed on something it does not change (for example, matching `pt1` against a corpus).

## 3. What it writes

One JSON object, rewritten atomically after each framework (`atomic_write_json`):

| Field | Requirement |
|---|---|
| `schema_version` | `ADAPTER_SCHEMA_VERSION` from `_common.py` (`cdrseqgen.adapter_output.v1`) |
| `adapter`, `benchmark_id` | adapter name; must equal the manifest's `benchmark_id` |
| `manifest` | `{path, sha256, schema_version}` of the exact manifest file used |
| `model` | free-form provenance: versions, checkpoint hash, `execution_source_identity()` |
| `sampling` | must include `track` (equal to the manifest's), `num_samples`, `temperature`, `top_p`, `base_seed` |
| `records[]` | one per framework, below |
| `started_at`, `finished_at`, `summary` | timing and totals |

`assert_comparable` rejects a comparison if `track`, `num_samples`, `temperature`, `top_p` or `base_seed`
differ between arms. A generator without those controls (the empirical sampler, or a non-token sampler) writes
the requested values and sets `sampling.sampling_controls_applied: false` with a `note`; record its real
controls under `model` or `sampling` so a reader is not misled. The scorer cannot yet express "not applicable".

Each record needs `example_id`, `native`, `native_length` (both equal to the manifest's), `samples`,
`attempts`, `valid_attempts`, `invalid_attempts`, `length_matches`, `length_mismatches`, `runtime_seconds`
and `complete`, with these identities enforced by `validate_adapter_record`:

- `attempts == valid_attempts + invalid_attempts` and `valid_attempts == length_matches + length_mismatches`;
- every sample is non-empty and uses the 20 amino-acid letters;
- on `fixed_native_length`, retained samples are length matches and `len(samples) <= length_matches`.

Keep `raw_generations` (every valid draw, duplicates included) and `failures` (code and detail per failed
attempt). Do not replenish failed attempts: the budget is exactly `--num-samples`, and failures count against it.
An example with `attempts == 0` is a crashed example; more than 5% of them fails the run.

## 4. Check it without a GPU or judge

    python benchmark/scorer.py --manifest M.json --input name=out.json --output /tmp/check.json

Without `--judge`, the scorer runs every validator above and scores on CPU. Use `--allow-partial` for an
unfinished run. Then run `blind_check.py` on a length-free manifest (it executes the model, so it is user-run):

    python benchmark/blind_check.py --manifest M.json --limit 4 -- \
        PYTHON ADAPTER.py --manifest {manifest} --output {output} --num-samples 3 --seed 42 --device cuda

## 5. Register it

1. Add an `External` to `EXTERNALS` in `../externals.py`: the script, the conda environment, a function that
   builds the adapter's own arguments, the packages to fingerprint, and a `check_record` for any input it cannot serve.
2. List it in a recipe: `"externals": [{"label": "name", "adapter": "name", "options": {...}}]`. A recipe with
   no `arms` runs only the externals, which adds a generator without regenerating finished arms.
3. Pass its interpreter with `--external-python name=PATH`, or create the environment the registry names.

4. List any input files the plan should freeze or pin (checkpoint, config, tables) under the `inputs` hook; a
   digest that is not its pin stops the plan before any model runs.

To put the new arm beside a finished comparison, run its recipe as a batch of its own (externals, no `arms`), then
`scripts/run.sh benchmark extend run --added BATCH`: it merges the scores only if the manifest, judge revision and
scoring policy match, runs the paired intervals over `planned_pairs(labels, externals=..., references=...)`, and
extends the capacity table, leaving the base batch untouched. The base may be an earlier extension, so they chain.
`--reference LABEL` makes an added arm a parameter-free reference pool compared with every system, and `--exclude LABEL`
merges and scores it but keeps it out of the contrast family (a sensitivity check). A worked protocol is `../recipes/oas4_h3_igcraft.md`.
