import os

"""Architecture-neutral evaluation primitives.

``metrics.py`` contains sequence-level measures shared by native evaluation and
cross-method benchmarks. Objective-specific recovery, decoding, perplexity,
and evaluation orchestration belong to the corresponding package under
``cdrseqgen.methods``.
"""


# A core metric represents the dataset only when at least this fraction of examples has residues
# remaining after anchor trimming. Shared by Trainer warnings and artifact-only reporting.
MIN_CORE_COVERAGE = 0.5


# Residues at each loop end treated as V/J-determined anchors rather than hypervariable core.
# Neutral: the boundary is a property of the loop, not of how a method generates it. Overridable by
# environment so a different numbering scheme can move it without editing code.
ANCHOR_TRIM = int(os.environ.get("ANCHOR_TRIM", "2"))
