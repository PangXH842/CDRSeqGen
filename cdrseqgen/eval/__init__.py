"""Shared sequence metrics, generative scoring and evaluation constants.

``metrics`` defines sequence comparisons used by model evaluation and benchmarks.
``generative`` selects frameworks, scores strategy-provided candidates, aggregates
metrics, and provides reporting and output helpers. Method packages supply
decoding strategies, self-perplexity implementations, teacher-forced recovery,
validity accounting and final evaluation payload assembly.

``ANCHOR_TRIM`` is the default number of residues excluded at each loop end when
scoring the core; it is read from the environment at import time (default 2).
``MIN_CORE_COVERAGE`` is the reporting threshold for the fraction of examples
with a non-empty core, not a filter on generated candidates.
"""

import os


# Threshold for warnings when too few examples have residues after anchor trimming.
# Shared by Trainer warnings and artifact-only reporting.
MIN_CORE_COVERAGE = 0.5


# Residues at each loop end treated as V/J-determined anchors rather than hypervariable core.
# Neutral: the boundary is a property of the loop, not of how a method generates it. Overridable by
# environment so a different numbering scheme can move it without editing code.
ANCHOR_TRIM = int(os.environ.get("ANCHOR_TRIM", "2"))
