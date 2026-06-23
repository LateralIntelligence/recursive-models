#!/usr/bin/env bash
#
# eval_infill_cell.sh — evaluate the FINAL (last.ckpt) checkpoint of ONE infill
# ablation cell. Root-agnostic: point ABLATIONS_ROOT at the local ./ablations
# folder or at a mounted bucket (e.g. /artifacts) and pass RUN_SUBDIR (the path
# of the run relative to that root). One SkyPilot job runs one cell -> the
# sweep parallelizes by fanning out cells.
#
# Required env:
#   TASK         sudoku | nqueens
#   RUN_SUBDIR   <ABLATION_TAG>/sweep-.../<run_name>   (relative to ABLATIONS_ROOT)
#
# sudoku env:  DIFFICULTY, INFILL_LOSS_REGION  (+ optional GLOBAL_BATCH/NUM_TRAIN/NUM_VALID)
# nqueens env: DATA, NQUEENS_N, MODEL_LENGTH    (+ optional NUM_SAMPLES/NUM_PUZZLES)
#
# Results land next to the checkpoint at
#   <ABLATIONS_ROOT>/<RUN_SUBDIR>/{sudoku_eval,nqueens_eval}/last/results.json
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${TASK:?set TASK=sudoku|nqueens}"
: "${RUN_SUBDIR:?set RUN_SUBDIR=<tag>/sweep-.../<run_name>}"

ABLATIONS_ROOT="${ABLATIONS_ROOT:-ablations}"
SAMPLING_STEPS="${SAMPLING_STEPS:-128}"

RUN_DIR="${ABLATIONS_ROOT}/${RUN_SUBDIR}"
CKPT="${RUN_DIR}/checkpoints/last.ckpt"
[[ -f "${CKPT}" ]] || { echo "ERROR: no last.ckpt at ${CKPT}"; exit 1; }
ABS_CKPT="$(realpath "${CKPT}")"

echo "=== eval cell: TASK=${TASK}  RUN_DIR=${RUN_DIR} ==="

if [[ "${TASK}" == "sudoku" ]]; then
  : "${DIFFICULTY:?set DIFFICULTY}"
  : "${INFILL_LOSS_REGION:?set INFILL_LOSS_REGION}"
  python main.py \
    mode=sudoku_eval \
    data=sudoku-gen \
    data.difficulty="${DIFFICULTY}" \
    data.num_train="${NUM_TRAIN:-10000}" \
    data.num_valid="${NUM_VALID:-2000}" \
    data.infill_loss_region="${INFILL_LOSS_REGION}" \
    model=small_infill \
    algo=flm \
    algo.infill=true \
    algo.diffusion_forcing=true \
    loader.global_batch_size="${GLOBAL_BATCH:-32}" \
    trainer.devices=1 \
    sampling.steps="${SAMPLING_STEPS}" \
    sampling.override_algo_steps=true \
    eval.checkpoint_path="${ABS_CKPT}"
  RESULT="${RUN_DIR}/sudoku_eval/last/results.json"

elif [[ "${TASK}" == "nqueens" ]]; then
  : "${DATA:?set DATA}"
  : "${NQUEENS_N:?set NQUEENS_N}"
  : "${MODEL_LENGTH:?set MODEL_LENGTH}"
  python main.py \
    mode=nqueens_eval \
    data="${DATA}" \
    data.nqueens_n="${NQUEENS_N}" \
    model=nqueens_infill \
    model.length="${MODEL_LENGTH}" \
    algo=flm \
    algo.infill=true \
    algo.diffusion_forcing=true \
    trainer.devices=1 \
    sampling.steps="${SAMPLING_STEPS}" \
    sampling.override_algo_steps=true \
    eval.nqueens_num_samples="${NUM_SAMPLES:-20}" \
    eval.nqueens_num_puzzles="${NUM_PUZZLES:-750}" \
    eval.checkpoint_path="${ABS_CKPT}"
  RESULT="${RUN_DIR}/nqueens_eval/last/results.json"

else
  echo "ERROR: unknown TASK=${TASK} (want sudoku|nqueens)"; exit 1
fi

[[ -f "${RESULT}" ]] || { echo "ERROR: results.json not written at ${RESULT}"; exit 1; }
echo "=== cell complete: ${RESULT} ==="
