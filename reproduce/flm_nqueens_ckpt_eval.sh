#!/usr/bin/env bash
#
# flm_nqueens_ckpt_eval.sh — Run N-Queens eval (mode=nqueens_eval) on one or many
# checkpoints, then render the GRAM-style accuracy/coverage-vs-#solutions plots.
#
# For each puzzle we draw eval.nqueens_num_samples completions and report:
#   - accuracy (fraction of samples satisfying all constraints + clues)
#   - coverage (distinct valid solutions found / total valid completions)
# Results land in <run>/nqueens_eval/<ckpt_stem>/results.json, with the plots
# written next to them.
#
# Usage:
#   ./reproduce/flm_nqueens_ckpt_eval.sh CHECKPOINT_PATH [extra hydra overrides...]
#   N=10 MODEL_LENGTH=100 DATA=nqueens-10 ./reproduce/flm_nqueens_ckpt_eval.sh CKPT
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "error: no checkpoint path provided." >&2
  echo "usage: ./reproduce/flm_nqueens_ckpt_eval.sh CHECKPOINT_PATH [overrides...]" >&2
  exit 1
fi

N="${N:-8}"
DATA="${DATA:-nqueens}"
MODEL_LENGTH="${MODEL_LENGTH:-$((N * N))}"
SAMPLING_STEPS="${SAMPLING_STEPS:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
NUM_PUZZLES="${NUM_PUZZLES:-200}"

# A single .ckpt -> just that; a directory -> every *.ckpt inside, version-sorted.
CHECKPOINTS=()
if [[ -f "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINTS=("${CHECKPOINT_PATH}")
elif [[ -d "${CHECKPOINT_PATH}" ]]; then
  while IFS= read -r ckpt; do CHECKPOINTS+=("${ckpt}"); done \
    < <(find "${CHECKPOINT_PATH}" -maxdepth 1 -type f -name '*.ckpt' | sort -V)
  [[ ${#CHECKPOINTS[@]} -gt 0 ]] || { echo "error: no .ckpt in ${CHECKPOINT_PATH}" >&2; exit 1; }
else
  echo "error: checkpoint path not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then shift; fi
EXTRA_OVERRIDES=("$@")

for ckpt in "${CHECKPOINTS[@]}"; do
  echo ""
  echo "============================================================"
  echo "==> nqueens_eval checkpoint: ${ckpt}"
  echo "============================================================"
  abs_ckpt="$(realpath "${ckpt}")"
  python main.py \
    mode=nqueens_eval \
    data="${DATA}" \
    data.nqueens_n="${N}" \
    model=nqueens_infill \
    model.length="${MODEL_LENGTH}" \
    algo=flm \
    algo.infill=true \
    algo.diffusion_forcing=true \
    sampling.steps="${SAMPLING_STEPS}" \
    sampling.override_algo_steps=true \
    eval.nqueens_num_samples="${NUM_SAMPLES}" \
    eval.nqueens_num_puzzles="${NUM_PUZZLES}" \
    eval.checkpoint_path="${abs_ckpt}" \
    "${EXTRA_OVERRIDES[@]}" \
    || { echo "FAILED (eval): ${ckpt}"; continue; }

  # Plot next to the results.json this run wrote.
  stem="$(basename "${ckpt%.*}")"
  run_dir="$(dirname "$(dirname "${abs_ckpt}")")"
  results="${run_dir}/nqueens_eval/${stem}/results.json"
  if [[ -f "${results}" ]]; then
    python plot_nqueens.py "${results}"
  else
    echo "WARN: results.json not found at ${results}; skipping plot"
  fi
done
