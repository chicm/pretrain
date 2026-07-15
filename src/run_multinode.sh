#!/bin/bash
# Multi-node pretraining across 8 nodes x 8 MI300X (=64 GPUs).
# This script is meant to be launched on EACH node with NODE_RANK set,
# or via a launcher that loops nodes (see launch_multinode.sh).
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

NNODES=${NNODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:?set NODE_RANK}
MASTER_ADDR=${MASTER_ADDR:?set MASTER_ADDR}
MASTER_PORT=${MASTER_PORT:-29500}

DATA="${WORKDIR:-.}/data/fineweb_tok"
OUT="${WORKDIR:-.}/checkpoints/fineweb_1b"

torchrun \
  --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  train.py --model 1b --data_dir "$DATA" --out_dir "$OUT" \
    --micro_bsz 8 --grad_accum 8 --max_steps 20000
