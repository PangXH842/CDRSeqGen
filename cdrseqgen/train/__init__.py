"""Training subsystem — what the root train.py entrypoint is assembled from.

  * arguments.py -- the three config dataclasses + setup_args (parse, name the run, make its dir)
  * workspace.py -- logging setup, train.log, run directories, results/latest, run_meta.json
  * pipeline.py  -- get_datasets, the train/eval driver, post-training generative eval
  * reporting.py -- epoch_metrics.csv and the four curve PNGs

train.py at the repo root stays the entrypoint and holds only main(): resolve a backbone, build the
datasets and model, hand them to a Trainer, run the pipeline. The recovery metrics it passes as
compute_metrics live in ../eval/recovery.py, beside the design-side metrics they pair with.

Order matters at import: arguments imports workspace (setup_args needs the run-dir helpers), and
pipeline imports reporting. Nothing imports arguments except the entrypoints, so there is no cycle.
"""
