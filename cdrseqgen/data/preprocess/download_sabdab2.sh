#!/usr/bin/env bash
# Downloads and extracts full SAbDab2 data splits tar.gz:
#
# Usage:  bash cdrseqgen/data/preprocess/download_sabdab2.sh

set -euo pipefail

# --- Download ---------------------------------------------------------------------------------
# Set destination directory (script to be run inside the preprocess/ directory)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ ! -d "$ROOT/cdrseqgen" || ! -f "$ROOT/train.py" ]]; then
    echo "❌ Resolved repo root '$ROOT' does not look like CDRSeqGen (no cdrseqgen/ + train.py)."
    echo "   Refusing to create a data tree there."
    exit 1
fi
DEST="$ROOT/dataset/data/raw/sabdab2/" # [to edit]

mkdir -p "$DEST"
echo "⬇️  Downloading SAbDab2 splits into: $DEST"
wget -O "$DEST/splits.tar.gz" https://zenodo.org/records/20083995/files/splits.tar.gz
tar -xzf "$DEST/splits.tar.gz" -C "$DEST"

echo "✅ Done. Next:"
echo "     python -m cdrseqgen.data.preprocess.sabdab2"
