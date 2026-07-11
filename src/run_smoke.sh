#!/bin/bash
# Single-node smoke test: tiny model on TinyStories.
# Run as the inter-node trust user inside the torch conda env.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

DATA="${WORKDIR:-.}/data/tinystories_tok"
OUT="${WORKDIR:-.}/checkpoints/smoke"

# 1) tokenize once if not done
if [ ! -f "$DATA/meta.json" ]; then
  python tokenize_data.py --dataset roneneldan/TinyStories --split train \
      --out "$DATA" --num_proc 32
fi

# 2) train
torchrun --standalone --nproc_per_node=4 train.py \
    --model tiny --data_dir "$DATA" --out_dir "$OUT" \
    --micro_bsz 16 --grad_accum 2 --max_steps 500 --no_compile
