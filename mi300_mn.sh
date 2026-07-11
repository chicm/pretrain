#!/bin/bash
# Multi-node 1B launcher (code-sync workflow) for an 8-GPU-per-node AMD (MI300X) cluster.
# Each node: git clone/pull repo -> LOCAL fast disk -> run from there.
# Data stays on the shared mount (large seq reads, fast+consistent). ckpt -> shared.
#
# Config via env (set real values at run time; defaults are placeholders):
#   REPO      : git URL of this repo
#   LOCAL     : per-node LOCAL disk checkout dir (strong consistency, avoids FUSE cache)
#   SHARED    : cross-node shared mount for data/ckpt/logs  (REQUIRED)
#   CONDA_ENV : conda env with ROCm torch
REPO=${REPO:-https://github.com/chicm/pretrain.git}
LOCAL=${LOCAL:-/scratch/code}
SHARED=${SHARED:?set SHARED to the shared work dir}
CONDA_ENV=${CONDA_ENV:-base}
DATA=$SHARED/data/tinystories_tok
OUT=$SHARED/checkpoints/mn_1b
LOGDIR=$SHARED/logs
mkdir -p "$OUT" "$LOGDIR"
rm -f "$LOGDIR"/mn_node*.log
NODES=(node-0 node-1 node-2 node-3)
for i in 0 1 2 3; do
  n=${NODES[$i]}
  ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no $n "
    set -e
    # --- sync code to LOCAL disk (avoids network-FS cache issues) ---
    git config --global --add safe.directory $LOCAL 2>/dev/null || true
    if [ -d $LOCAL/.git ]; then cd $LOCAL && git fetch -q origin && git reset -q --hard origin/main; else rm -rf $LOCAL && git clone -q $REPO $LOCAL; fi
    echo \\\"[$n] code HEAD: \\\$(cd $LOCAL && git rev-parse --short HEAD)\\\"
    cd $LOCAL/src
    source /opt/conda/etc/profile.d/conda.sh; conda activate $CONDA_ENV
    export HF_HOME=/scratch/hf_local OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
    export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=eth0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
    nohup torchrun --nnodes=4 --nproc_per_node=8 --node_rank=$i \
      --rdzv_id=mn1b --rdzv_backend=c10d --rdzv_endpoint=node-0:29500 --rdzv_conf=timeout=900 \
      train.py --model 1b --data_dir $DATA --out_dir $OUT \
      --micro_bsz 8 --grad_accum 2 --max_steps 500 --no_compile \
      > $LOGDIR/mn_node${i}.log 2>&1 &
  " &
done
wait
echo "launched 4 nodes x 8 GPU (code synced to LOCAL disk per node)"
