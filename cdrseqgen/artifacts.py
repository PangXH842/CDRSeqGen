"""Shared names for files persisted in a CDRSeqGen run directory."""

# New runs write the canonical name. Readers use the ordered tuple so current artifacts win while
# older checkpoints remain readable. On a future rename, move the previous canonical name into
# RUN_CONFIG_LEGACY_FILENAMES; consumers do not need to change.
RUN_CONFIG_FILENAME = "run_config.json"
RUN_CONFIG_LEGACY_FILENAMES = ("run_meta.json",)
RUN_CONFIG_FILENAMES = (
    RUN_CONFIG_FILENAME,
    *RUN_CONFIG_LEGACY_FILENAMES,
)

# The length predictor is a deliberately separate artifact: old sequence checkpoints remain valid
# for fixed-length generation, while bounded/inferred generation fails clearly when these are absent.
LENGTH_CONFIG_FILENAME = "length_config.json"
LENGTH_HEAD_FILENAME = "length_head.safetensors"
