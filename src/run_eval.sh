#!/bin/bash
# Evaluate a Chimera checkpoint on base-model likelihood tasks (lm-eval-harness).
# Run on a single GPU inside the training conda env.
#
#   bash run_eval.sh <ckpt_path> [extra args...]
#   bash run_eval.sh $OUT/ckpt_2000.pt
#   bash run_eval.sh $OUT/ckpt_2000.pt --mmlu
#   bash run_eval.sh $OUT/ckpt_2000.pt --limit 200      # quick smoke
set -e
cd "$(dirname "$0")"
CKPT="${1:?usage: run_eval.sh <ckpt_path> [extra args]}"
shift || true

export TOKENIZERS_PARALLELISM=false
# pin to one GPU (respect an already-set visible-devices, else use 0)
: "${HIP_VISIBLE_DEVICES:=${CUDA_VISIBLE_DEVICES:-0}}"
export HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -c "import lm_eval" 2>/dev/null || {
  echo "[run_eval] installing lm-eval..."
  pip install --user "lm-eval>=0.4.3"
}

python eval.py --ckpt "$CKPT" --batch_size "${BATCH_SIZE:-16}" "$@"
