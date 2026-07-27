#!/usr/bin/env bash
# Download + extract the SAbDab2 ML-grade structural splits (~876MB, one-time).
#
# Writes into the gitignored data blob at <repo>/dataset/data/raw/sabdab2/, resolved from THIS
# script's own location — so it works from any working directory. (It previously did a bare
# `mkdir -p raw/sabdab2`, relative to CWD: run as documented from the repo root, that silently
# dumped the download into <repo>/raw/sabdab2/, where no config or preprocessor looks.)
#
# Usage:  bash cdrseqgen/data/preprocess/download_sabdab2.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/dataset/data/raw/sabdab2/"

mkdir -p "$DEST"
echo "⬇️  Downloading SAbDab2 splits into: $DEST"
wget -O "$DEST/splits.tar.gz" https://zenodo.org/records/20083995/files/splits.tar.gz
tar -xzf "$DEST/splits.tar.gz" -C "$DEST"

echo "✅ Done. Next:"
echo "     python -m cdrseqgen.data.preprocess.sabdab2"
