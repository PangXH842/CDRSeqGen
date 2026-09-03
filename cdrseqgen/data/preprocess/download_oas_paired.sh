#!/usr/bin/env bash
# Bulk-download the OAS human paired repertoires into the gitignored data blob at
# <repo>/dataset/data/raw/, resolved from THIS script's own location so it works from any CWD.
#
# The wget lines below are bare (no -O/-P), so they write into the working directory — hence the
# `cd`. Without it, running this from the repo root scattered .csv.gz files across <repo>/.
# No `set -e` on purpose: one dead URL should not abort the remaining downloads.
#
# Usage:  bash cdrseqgen/data/preprocess/download_oas_paired.sh    (downloads AND extracts)
#   then: python -m cdrseqgen.data.preprocess.oas_paired \
#             --input_file dataset/data/raw/oas/*_Paired_All.csv \
#             --output_file dataset/data/processed/processed_oas_multi.json \
#             --split cluster --test_size 0.1 --validation_size 0.1 --seed 42
#
# The selection below is Jaffe 2022, its FOUR donors (the whole study -- there is no fifth), two
# naive runs each so the donors stay balanced for a future donor-held-out split. Every run is
# Naive-B-Cells on purpose: memory and plasmablast repertoires carry heavy somatic hypermutation,
# which would shift the germline-encoded H1/H2 distribution far more than the disease labels do.
# Probed 2026-09-03: outside Jaffe only ONE naive paired run exists in this catalogue
# (Woodruff_2020), so more donors means accepting mixed B-cell types.

# THREE levels up: this script lives at <repo>/cdrseqgen/data/preprocess/, so ../.. only reaches
# <repo>/cdrseqgen. It used to, and silently downloaded into <repo>/cdrseqgen/dataset/data/raw/ --
# which .gitignore does not cover, since its `dataset/data/` pattern is anchored to the repo root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit 1
if [[ ! -d "$ROOT/cdrseqgen" || ! -f "$ROOT/train.py" ]]; then
    echo "❌ Resolved repo root '$ROOT' does not look like CDRSeqGen (no cdrseqgen/ + train.py)."
    echo "   Refusing to create a data tree there."
    exit 1
fi
DEST="$ROOT/dataset/data/raw/oas/"
mkdir -p "$DEST" || exit 1
cd "$DEST" || exit 1
echo "⬇️  Downloading OAS paired repertoires into: $DEST"

# Donor 1: SARS-COV-2 (19627, 19085)
wget -c https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1287203_1_Paired_All.csv.gz
wget -c https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1287201_1_Paired_All.csv.gz
# Donor 2: SARS-COV-2 (19068, 16401)
wget -c https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1287186_1_Paired_All.csv.gz
wget -c https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1287187_1_Paired_All.csv.gz
# Donor 3: none (19839, 19303)
wget -c https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1287154_1_Paired_All.csv.gz
wget -c https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1287155_1_Paired_All.csv.gz
# Donor 4: CMV (21839, 21106)
wget -c https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1287171_1_Paired_All.csv.gz
wget -c https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Jaffe_2022/csv/1287169_1_Paired_All.csv.gz

# --- Extract ----------------------------------------------------------------------------------
# The preprocessor reads plain CSV, so unpack here rather than leaving it as a step to remember.
# `-k` keeps the .gz: it is a tenth of the size of the CSV and means a corrupted extraction never
# costs another 154 MB download. Every stage is skippable, so re-running the script is cheap.
echo ""
echo "📦 Extracting..."
extracted=0
skipped=0
failed=0
for archive in *.csv.gz; do
    [[ -e "$archive" ]] || { echo "⚠️  Nothing downloaded — no .csv.gz in $DEST"; break; }
    csv="${archive%.gz}"
    # NOT `$csv -nt $archive`: gunzip -k copies the archive's mtime onto the output, so the two are
    # equal and "strictly newer" is false -- which silently re-extracted 1.6 GB on every run. Ask
    # the question that actually matters instead: has the archive changed since we unpacked it?
    if [[ -s "$csv" && ! "$archive" -nt "$csv" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    # Verify before unpacking: a truncated download would otherwise become a truncated CSV, and
    # pandas reads a truncated OAS CSV without complaining.
    if ! gunzip -t "$archive" 2>/dev/null; then
        echo "  ❌ $archive is corrupt — delete it and re-run to fetch again"
        failed=$((failed + 1))
        continue
    fi
    gunzip -kf "$archive" && extracted=$((extracted + 1)) || failed=$((failed + 1))
done
echo "   extracted $extracted, already present $skipped, failed $failed"

if [[ $failed -gt 0 ]]; then
    echo "❌ $failed archive(s) could not be extracted; fix those before preprocessing."
    exit 1
fi

echo ""
echo "✅ Done. $(ls -1 *.csv 2>/dev/null | wc -l) CSV(s), $(du -sh . | cut -f1) total in $DEST"
echo "   Next: python -m cdrseqgen.data.preprocess.oas_paired \\"
echo "             --input_file dataset/data/raw/oas/*_Paired_All.csv \\"
echo "             --output_file dataset/data/processed/processed_oas_multi.json \\"
echo "             --split cluster --test_size 0.1 --validation_size 0.1 --seed 42"
