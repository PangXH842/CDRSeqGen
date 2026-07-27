"""Offline preprocessing: raw downloads -> the processed JSON that dataset.py consumes.

Runs once per dataset, from the repo root:

    bash cdrseqgen/data/preprocess/download_oas_paired.sh   # fetch  -> dataset/data/raw/
    python -m cdrseqgen.data.preprocess.oas_paired          # build  -> dataset/data/processed/

  * oas_paired.py  -- OAS paired repertoires (sequence-only, no antigen) + the leakage-safe split
  * sabdab2.py     -- SAbDab2 structures (adds an 'antigen' block; split ships with the data)
  * common.py      -- what both share: validation, chain assembly, CLI args, funnel, JSON output
  * download_*.sh  -- fetch the raw inputs into dataset/data/raw/ (CWD-independent)

Both scripts emit the same entry schema, so downstream code never branches on dataset; only the
extraction step differs (OAS lifts ready-made region columns, SAbDab2 folds up per-residue
numbering). The runtime half that reads this output is ../dataset.py.
"""
