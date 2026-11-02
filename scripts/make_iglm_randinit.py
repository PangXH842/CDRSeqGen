#!/usr/bin/env python3
"""Create the random-initialised IgLM control: the released architecture with untrained weights.

The directory is a valid `iglm:` model source (GPT-2 config of the release, native 33-token vocabulary,
`cdrseqgen_backbone` marker), so the experiment runner trains it through exactly the code path of the
fine-tuned models; only the weights differ. Initialisation is seeded, and RANDOM_INIT.json records the
seed and the weight hash so the control can be recognised later.

  python scripts/make_iglm_randinit.py [--source DIR] [--output DIR] [--seed N] [--force]

Lives outside the training package, so running it does not change any batch's source identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = Path.home() / "miniconda3/envs/iglm/lib/python3.11/site-packages/iglm/trained_models/IgLM-S"
DEFAULT_OUTPUT = ROOT / "results/controls/iglm_s_random_init"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(source, output, seed, force=False):
    """Write the control to `output` and return its provenance record."""
    import torch
    import transformers
    from transformers import GPT2Config, GPT2LMHeadModel
    from cdrseqgen.datasets.dataset_ar import span_layout_metadata
    from cdrseqgen.methods.ar.models import iglm

    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must not overlap")
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"{output} exists and is not a directory")
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"{output} exists and is not empty; pass --force to replace it")
    # Validate every source input before replacing an existing control, including with --force.
    config = GPT2Config.from_pretrained(source, local_files_only=True)
    iglm._check_config(config.to_dict())            # the release's GPT-2 shape and native vocabulary size
    config.bos_token_id, config.eos_token_id, config.pad_token_id = None, iglm.CLS_ID, iglm.PAD_ID
    config.forced_eos_token_id = None
    source_hash = sha256(source / "config.json")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = GPT2LMHeadModel(config)
    model.config.cdrseqgen_backbone = "iglm"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "new"
        staging.mkdir()
        model.config.save_pretrained(staging)
        (staging / "vocab.txt").write_text("\n".join(iglm.NATIVE_VOCAB) + "\n")
        tokenizer = iglm.load_tokenizer(staging)
        layout = span_layout_metadata(tokenizer)
        model.config.span_layout = layout
        model.config.causal_sampling = dict(tokenizer.causal_sampling_metadata)
        model.save_pretrained(staging)
        tokenizer.save_pretrained(staging)
        (staging / 'span_layout.json').write_text(json.dumps(layout, indent=2) + '\n')
        record = dict(
            purpose="random-initialised control with the released IgLM architecture",
            source_config=str(source / "config.json"), source_config_sha256=source_hash,
            seed=seed, parameters=sum(p.numel() for p in model.parameters()),
            weights_sha256=sha256(staging / "model.safetensors"), torch=torch.__version__,
            transformers=transformers.__version__)
        (staging / "RANDOM_INIT.json").write_text(json.dumps(record, indent=2) + "\n")
        backup = Path(temporary) / "previous"
        if output.exists():
            output.rename(backup)
        try:
            staging.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="released IgLM-S directory (architecture)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    record = create(args.source, args.output, args.seed, args.force)
    print(f"Wrote {args.output} ({record['parameters']:,} parameters, seed {record['seed']}, "
          f"weights {record['weights_sha256'][:12]})")


if __name__ == "__main__":
    main()
