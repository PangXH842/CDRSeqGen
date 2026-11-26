"""Post-batch reporting: tables and plots for a finished suite or sweep."""
from __future__ import annotations

import subprocess
import sys


def report(plan, output):
    jobs = ([('batch_table', '--batch', str(output), '--output'),
             ('batch_figure', '--batch', str(output))] if plan["mode"] == "suite" else
            [('sweep_table', '--sweep', str(output), '--rank_metric', plan["rank_metric"]),
             ('sweep_figure', '--sweep', str(output))])
    for module, *args in jobs:
        code = subprocess.call([sys.executable, '-m', f'cdrseqgen.visualization.{module}', *args])
        if code:
            print(f"Reporting failed ({module}); training artifacts are in {output}", flush=True)
