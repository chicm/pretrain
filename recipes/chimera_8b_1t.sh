#!/usr/bin/env bash
# Exact launcher recipe for the Chimera-8B 1T run started in July 2026.
#
# This recipe intentionally records the real cluster topology, paths, code
# revision, environment, and training arguments. It is the reproducibility
# record for this experiment; generic code under scripts/ stays parameterized.
#
# Exact detached launch command used from the cluster login/node-0 shell:
#   su aiscuser -c 'setsid bash /scratch/code/recipes/chimera_8b_1t.sh > /scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/chec/pretrain/logs/mn_launch.log 2>&1 < /dev/null &'
#
# The original live run was started from the equivalent shared-disk launcher
# `_launch_1t_v1.sh`. This tracked recipe pins its effective configuration.
set -euo pipefail

case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

RECIPE_FILE=$(readlink -f "${BASH_SOURCE[0]}")

# Historical source revision used to start the original 1T run. Future resumes
# use the checked-out immutable revision below; every node must match it. The
# runtime manifest records that resolved full SHA for each launch.
ORIGINAL_RUN_CODE_REV="23ab530a531fe84ba25aaeafebfec4dabc61c59b"
REPO_URL="https://github.com/chicm/pretrain.git"
LOCAL_REPO="/scratch/code"
CODE_REV=$(git -C "$LOCAL_REPO" rev-parse HEAD)
CODE_SYNC_MODE="verify"          # deploy the same revision to every node first

# Exact 8-node MI300X topology.
NODES=(node-0 node-1 node-2 node-3 node-4 node-5 node-6 node-7)

# Exact runtime environment.
CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV="py_3.10"
HF_HOME="/scratch/hf_local"

# Exact shared storage layout.
SHARED="/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/chec/pretrain"
DATA="$SHARED/data/tinystories_tok"
DATA_ROOT="$SHARED/data"
OUT="$SHARED/checkpoints/chimera_1t"
LOGDIR="$SHARED/logs"
TB_DIR="/scratch/azureml/cr/j/9c09a3062dda4b66a76667ef14ead331/exe/wd"

# Exact environment exported on every worker node.
REMOTE_ENV=(
  "HF_HOME=$HF_HOME"
  "OMP_NUM_THREADS=8"
  "TOKENIZERS_PARALLELISM=false"
  "NCCL_DEBUG=WARN"
  "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
  "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900"
)

# Exact torchrun rendezvous configuration.
TORCHRUN_COMMON_ARGS=(
  "--nnodes=8"
  "--nproc_per_node=8"
  "--rdzv_id=chimera1t"
  "--rdzv_backend=c10d"
  "--rdzv_endpoint=node-0:29500"
  "--rdzv_conf=timeout=900"
)

# Exact effective train.py invocation. Explicit positive FP8/fused-CE flags are
# recorded even though revision 23ab530 enables them by default. Compilation is
# ON because --no_compile is intentionally absent.
TRAIN_ARGS=(
  "train.py"
  "--model" "8b"
  "--data_dir" "$DATA"
  "--out_dir" "$OUT"
  "--micro_bsz" "4"
  "--grad_accum" "4"
  "--max_steps" "238000"
  "--data_mix" "mix_1t"
  "--data_root" "$DATA_ROOT"
  "--resume" "latest"
  "--keep_last_ckpts" "3"
  "--tb_dir" "$TB_DIR"
  "--fp8"
  "--fp8_recipe" "tensorwise"
  "--fused_ce"
)

# shellcheck source=../scripts/launch_multinode.sh
source "/scratch/code/scripts/launch_multinode.sh"
launch_multinode
