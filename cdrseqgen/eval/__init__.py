"""Architecture-neutral evaluation primitives.

``metrics.py`` contains sequence-level measures shared by native evaluation and
cross-method benchmarks. Objective-specific recovery, decoding, perplexity,
and evaluation orchestration belong to the corresponding package under
``cdrseqgen.methods``.
"""


# A core metric represents the dataset only when at least this fraction of examples has residues
# remaining after anchor trimming. Shared by Trainer warnings and artifact-only reporting.
MIN_CORE_COVERAGE = 0.5
