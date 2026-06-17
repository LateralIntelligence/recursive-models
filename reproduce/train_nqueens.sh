#!/usr/bin/env bash
#
# train_nqueens.sh — Train a plain FLM (algo=flm) on the generated N-Queens
# dataset in the in-place *infilling* formulation: the model sees the solution
# board and a conditioning_mask over the clue queens (model=nqueens_infill,
# length 64 for 8x8). Tokens: 0=pad, 1=empty, 2=queen.
#
# Usage:
#   ./reproduce/train_nqueens.sh [extra hydra overrides...]
#   N=10 MODEL_LENGTH=100 DATA=nqueens-10 ./reproduce/train_nqueens.sh   # 10x10
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N="${N:-8}"
DATA="${DATA:-nqueens}"                 # nqueens (8x8) or nqueens-10 (10x10)
MODEL_LENGTH="${MODEL_LENGTH:-$((N * N))}"
NUM_TRAIN="${NUM_TRAIN:-15000}"
NUM_VALID="${NUM_VALID:-2000}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MAX_STEPS="${MAX_STEPS:-100001}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-128}"
RUN_DIR="${RUN_DIR:-outputs/nqueens-${N}x${N}}"

python main.py \
  data="${DATA}" \
  data.nqueens_n="${N}" \
  data.num_train="${NUM_TRAIN}" \
  data.num_valid="${NUM_VALID}" \
  data.infill_loss_region=board \
  model=nqueens_infill \
  model.length="${MODEL_LENGTH}" \
  algo=flm \
  algo.infill=true \
  algo.diffusion_forcing=true \
  loader.global_batch_size="${GLOBAL_BATCH}" \
  sampling.steps="${SAMPLING_STEPS}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
  trainer.check_val_every_n_epoch=null \
  hydra.run.dir="${RUN_DIR}" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${CKPT_EVERY}" \
  "$@"

echo ""
echo "=== done. checkpoints under ${RUN_DIR}/checkpoints ==="
