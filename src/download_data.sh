#!/bin/bash
set -e
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
source ~/.bashrc
conda activate "${CONDA_ENV:-base}" 2>/dev/null || true
DATA="${WORKDIR:-.}/data"
mkdir -p "$DATA"
cd "$DATA"
echo "START $(date)"
pip install -q -U hf_transfer 2>&1 | tail -2
echo "INSTALLED"
# 1) TinyStories (small smoke test)
hf download roneneldan/TinyStories --repo-type dataset --local-dir "$DATA/tinystories"
echo "TINYSTORIES_DONE $(date)"
du -sh "$DATA/tinystories"
# 2) FineWeb sample-10BT (~28GB)
hf download HuggingFaceFW/fineweb --repo-type dataset --include "sample/10BT/*" --local-dir "$DATA/fineweb"
echo "FINEWEB_DONE $(date)"
du -sh "$DATA/fineweb"
echo "ALL_DONE $(date)"
