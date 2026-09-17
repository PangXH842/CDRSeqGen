#!/usr/bin/env bash
# Downloads and extracts full SAbDab2 data splits tar.gz:
#
# Usage:  bash cdrseqgen/datasets/preprocess/download_sabdab2.sh

set -euo pipefail

# --- Download ---------------------------------------------------------------------------------
# Resolve the destination from this script; it works from any working directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ ! -d "$ROOT/cdrseqgen" || ! -f "$ROOT/train.py" ]]; then
    echo "❌ Resolved repo root '$ROOT' does not look like CDRSeqGen (no cdrseqgen/ + train.py)."
    echo "   Refusing to create a data tree there."
    exit 1
fi
DEST="$ROOT/data/raw/sabdab2/"

mkdir -p "$DEST"
echo "⬇️  Downloading SAbDab2 splits into: $DEST"
wget -O "$DEST/splits.tar.gz" https://zenodo.org/records/20083995/files/splits.tar.gz
tar -xzf "$DEST/splits.tar.gz" -C "$DEST"

echo "✅ Done. Next:"
echo "     python -m cdrseqgen.datasets.preprocess.sabdab2"
