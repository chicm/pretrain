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

# Migrated 15-node MI300X topology (120 GPUs).
NODES=(node-{0..14})

# ROCm 7.1 image runtime environment.
VENV_DIR="/opt/venv"
HF_HOME="/scratch/hf_local"

# Exact shared storage layout.
SHARED="/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/chec/pretrain"
DATA="$SHARED/data/tinystories_tok"
DATA_ROOT="$SHARED/data"
OUT="${OUT_OVERRIDE:-$SHARED/checkpoints/chimera_1t}"
LOGDIR="${LOGDIR_OVERRIDE:-$SHARED/logs}"
TB_DIR="${TB_DIR_OVERRIDE:-/scratch/azureml/cr/j/454bfccaf2e34030b93119d8f6a55b99/exe/wd}"
MAX_STEPS="${MAX_STEPS_OVERRIDE:-252667}"
RESUME_FROM="${RESUME_OVERRIDE:-latest}"
RDZV_ID="${RDZV_ID_OVERRIDE:-chimera1t}"

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
  "--nnodes=${#NODES[@]}"
  "--nproc_per_node=8"
  "--rdzv_id=$RDZV_ID"
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
  "--grad_accum" "2"
  # ROCm 7.1 can deadlock when DataLoader workers fork after NCCL/FSDP init.
  # Token blocks are memory-mapped; synchronous loading is cheap relative to a step.
  "--data_workers" "0"
  "--max_steps" "$MAX_STEPS"
  "--lr_warmup_tokens" "6291456000"
  "--lr_schedule_total_tokens" "998244352000"
  # ckpt_18000 predates consumed-token checkpoint metadata. New checkpoints
  # persist it and take precedence over this one-time migration fallback.
  "--resume_consumed_tokens" "75501666304"
  "--data_mix" "mix_1t"
  "--data_root" "$DATA_ROOT"
  "--resume" "$RESUME_FROM"
  "--keep_last_ckpts" "3"
  "--tb_dir" "$TB_DIR"
  "--fp8"
  "--fp8_recipe" "tensorwise"
  "--fused_ce"
  # Validated 64-GPU FSDP2 winner: synchronize/reshard only on the final
  # accumulation microbatch. Steady throughput 637K tok/s vs 600K baseline.
  "--fsdp_sync_last_micro"
  "--fsdp_reshard_last_micro"
)
if [[ ${NO_SAVE_FINAL_OVERRIDE:-0} == 1 ]]; then
  TRAIN_ARGS+=("--no_save_final")
fi

# shellcheck source=../scripts/launch_multinode.sh
source "/scratch/code/scripts/launch_multinode.sh"
launch_multinode
