# CDR-H3 benchmark manifests

The manifest is the frozen, model-independent input shared by CDRSeqGen and external baselines.
Model adapters may tokenize these strings differently, but must not reconstruct or reselect the
examples themselves.

## Included sample

`manifests/oas_h3_test_sample_v1.json` contains a deterministic 25-record sample of the 2,111-record
clustered OAS test split. Its companion `.sha256` file pins the exact artifact.

Each record contains:

- `example_id`: stable join key for model outputs and paired statistics
- `pt1`, `native`, `pt2`: framework-before, native CDR-H3, and framework-after strings
- `parent_sequence`: validated as `pt1 + native + pt2`
- `native_length`: fixed-length benchmark target
- `infill_range`: zero-indexed, end-exclusive coordinates for `IgLM.infill`
- IgLM chain/species conditioning tokens

The sample excludes light-chain and antigen context because pretrained IgLM's infill API consumes a
single antibody chain. Species and source-run provenance come from the processed records, and the
exporter verifies the processed file against its preprocessing metadata sidecar before writing a
manifest. It fails instead of guessing when species or stable source provenance is missing.

## Regenerate

From the repository root:

```bash
python benchmark/export_manifest.py \
  --input dataset/data/processed/processed_oas_tuning.json \
  --output benchmark/manifests/oas_h3_test_sample_v1.json \
  --sample-size 25 \
  --benchmark-id oas_h3_test_sample_v1
```

Omit `--sample-size 25` and change the benchmark/output names to export the full test split after the
sampling and scoring adapters pass the smoke experiment.

## Adapter contract

Both adapters should emit the same model-independent record shape:

```json
{
  "example_id": "oas:...",
  "native": "...",
  "samples": ["...", "..."],
  "attempts": 20,
  "failures": []
}
```

For the fixed-length track, retain only samples whose length equals `native_length`, while preserving
the total attempt count and length-mismatch failures. Do not alter the manifest to store generated
outputs.
