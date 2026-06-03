#!/usr/bin/env bash
#
# sample_eval.sh — Reproduce the sampling-based evaluation on Sudoku-Extreme.
#
# Usage:
#   ./sample_eval.sh CHECKPOINT_PATH [extra hydra overrides...]
#   CHECKPOINT_PATH=/path/to/model.ckpt ./sample_eval.sh
#
# Examples:
#   ./reproduce/flm_sudoku_baseline.sh outputs/sudoku-extreme/2026.05.31/224049/checkpoints/1-25000.ckpt
#   ./reproduce/flm_sudoku_baseline.sh model.ckpt eval.num_steps=16 eval.batch_size=64
#
set -euo pipefail

# --- Resolve repo root so the script works from any directory ---------------
# This script lives in <repo>/reproduce/, but main.py is at the repo root,
# so we go up one level from the script's own directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Read the checkpoint from the first arg, or fall back to the env var ----
CHECKPOINT_PATH="${1:-${CHECKPOINT_PATH:-}}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  cat >&2 <<'EOF'
error: no checkpoint path provided.

Usage:
  ./sample_eval.sh CHECKPOINT_PATH [extra hydra overrides...]
  CHECKPOINT_PATH=/path/to/model.ckpt ./sample_eval.sh

Example:
  ./sample_eval.sh outputs/sudoku-extreme/.../checkpoints/1-25000.ckpt
EOF
  exit 1
fi

# --- Build the list of checkpoints to evaluate ------------------------------
# A single .ckpt file -> evaluate just that one.
# A directory         -> evaluate every *.ckpt inside it, version-sorted so
#                        e.g. 1-5000 < 1-25000 < 1-100000.
CHECKPOINTS=()
if [[ -f "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINTS=("${CHECKPOINT_PATH}")
elif [[ -d "${CHECKPOINT_PATH}" ]]; then
  while IFS= read -r ckpt; do
    CHECKPOINTS+=("${ckpt}")
  done < <(find "${CHECKPOINT_PATH}" -maxdepth 1 -type f -name '*.ckpt' | sort -V)
  if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "error: no .ckpt files found in directory: ${CHECKPOINT_PATH}" >&2
    exit 1
  fi
else
  echo "error: checkpoint path not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

# If a checkpoint was passed as $1, drop it so the rest are hydra overrides.
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_OVERRIDES=("$@")

# --- Run --------------------------------------------------------------------
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  echo "==> overrides: ${EXTRA_OVERRIDES[*]}"
fi
echo "==> evaluating ${#CHECKPOINTS[@]} checkpoint(s)"

for ckpt in "${CHECKPOINTS[@]}"; do
  echo ""
  echo "============================================================"
  echo "==> checkpoint: ${ckpt}"
  echo "============================================================"
  python main.py \
    mode=sample_eval \
    eval.checkpoint_path="${ckpt}" \
    "${EXTRA_OVERRIDES[@]}"
done