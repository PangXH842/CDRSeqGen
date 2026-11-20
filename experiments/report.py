"""Post-batch reporting: tables and plots for a finished suite or sweep."""
from __future__ import annotations

import subprocess
import sys


def report(plan, output):
    jobs = ([('batch_tables', '--batch', str(output), '--output'),
             ('batch_plots', '--batch', str(output))] if plan["mode"] == "suite" else
            [('sweep_tables', '--sweep', str(output), '--rank_metric', plan["rank_metric"]),
             ('sweep_plots', '--sweep', str(output))])
    for module, *args in jobs:
        code = subprocess.call([sys.executable, '-m', f'cdrseqgen.visualization.{module}', *args])
        if code:
            print(f"Reporting failed ({module}); training artifacts are in {output}", flush=True)
