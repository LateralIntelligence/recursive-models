#!/usr/bin/env bash
#
# sweep_flm_infill_nqueens.sh — Train a sweep of plain FLM (algo=flm) on the
# generated N-Queens dataset in the in-place *infilling* formulation, then run
# nqueens_eval on every checkpoint each run produces and render the GRAM-style
# accuracy/coverage-vs-#solutions plots.
#
# This fuses the two previous scripts into one train -> eval pipeline:
#   - train_nqueens.sh          (the training sweep over conditioning_prob_clean)
#   - flm_nqueens_ckpt_eval.sh  (per-checkpoint nqueens_eval + plot_nqueens.py)
# The combined structure mirrors sweep_flm_infill_sudoku_gen.sh.
#
# Formulation: the model sees the solution board and a conditioning_mask over the
# clue queens (model=nqueens_infill, length N*N for an NxN board). Tokens:
# 0=pad, 1=empty, 2=queen. Loss region is the whole board.
#
# What this sweep fixes vs. varies:
#   - algo / formulation : flm + algo.infill=true, in-place board infilling.
#   - loss region        : whole board (data.infill_loss_region=board).
#   - VARIED             : algo.conditioning_prob_clean over CONDITIONING_PROBS_CLEAN.
#
# As in the sudoku sweep, we pin hydra.run.dir to a per-run path keyed by
# run_name, so checkpoints land in a known location
# (outputs/<sweep>/<run_name>/checkpoints) instead of hydra's timestamped dir,
# which makes them trivial to find and evaluate afterwards.
#
# Eval: for each puzzle we draw eval.nqueens_num_samples completions and report
# accuracy (fraction satisfying all constraints + clues) and coverage (distinct
# valid solutions found / total valid completions). Results land in
# <run>/nqueens_eval/<ckpt_stem>/results.json, with plots written next to them.
#
# Usage:
#   ./reproduce/sweep_flm_infill_nqueens.sh [extra hydra TRAIN overrides...]
#   EVAL_ONLY=1 ./reproduce/sweep_flm_infill_nqueens.sh        # skip training, eval existing ckpts
#   N=10 MODEL_LENGTH=100 DATA=nqueens-10 ./reproduce/sweep_flm_infill_nqueens.sh   # 10x10
#   CONDITIONING_PROBS_CLEAN="0.5 1.0" ./reproduce/sweep_flm_infill_nqueens.sh
#
set -euo pipefail

# --- Resolve repo root so the script works from any directory ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Board / data config (all overridable via env) --------------------------
N="${N:-10}"
DATA="${DATA:-nqueens-10}"                 # nqueens (8x8) or nqueens-10 (10x10)
MODEL_LENGTH="${MODEL_LENGTH:-$((N * N))}"
NUM_TRAIN="${NUM_TRAIN:-15000}"
NUM_VALID="${NUM_VALID:-2000}"

# --- Train config -----------------------------------------------------------
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MAX_STEPS="${MAX_STEPS:-100001}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-128}"

conditioning_time_random="${CONDITIONING_TIME_RANDOM:-true}"
read -r -a conditioning_prob_clean_values <<< "${CONDITIONING_PROBS_CLEAN:-0.2 0.5 1.0}"

# --- Eval config ------------------------------------------------------------
NUM_SAMPLES="${NUM_SAMPLES:-20}"        # eval.nqueens_num_samples
NUM_PUZZLES="${NUM_PUZZLES:-750}"       # eval.nqueens_num_puzzles

SWEEP_ROOT="${SWEEP_ROOT:-outputs/sweep-flm-infill-nqueens-${N}}"
EVAL_ONLY="${EVAL_ONLY:-0}"

# Extra hydra overrides go to TRAINING only (matching train_nqueens.sh). They are
# intentionally not forwarded to eval, since trainer.* / loader.* overrides don't
# apply there and could error.
EXTRA_OVERRIDES=("$@")

for cprob in "${conditioning_prob_clean_values[@]}"; do
  run_name="FLM_Infill_NQueens_board_N:${N}_cond_time_random:${conditioning_time_random}_clean_prob:${cprob}"
  run_dir="${SWEEP_ROOT}/${run_name}"
  echo ""
  echo "=== ${run_name} ==="

  # --- Train ----------------------------------------------------------------
  if [[ "${EVAL_ONLY}" != "1" ]]; then
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
      algo.conditioning_time_random="${conditioning_time_random}" \
      algo.conditioning_prob_clean="${cprob}" \
      algo.diffusion_forcing=true \
      loader.global_batch_size="${GLOBAL_BATCH}" \
      sampling.steps="${SAMPLING_STEPS}" \
      trainer.max_steps="${MAX_STEPS}" \
      trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
      trainer.check_val_every_n_epoch=null \
      hydra.run.dir="${run_dir}" \
      callbacks.checkpoint_every_n_steps.every_n_train_steps="${CKPT_EVERY}" \
      "wandb.name='${run_name}'" \
      "${EXTRA_OVERRIDES[@]}" \
      || { echo "!!! run failed (train) for clean_prob=${cprob}, continuing"; continue; }
  fi

  # --- Eval every checkpoint from this run ----------------------------------
  ckpt_dir="${run_dir}/checkpoints"
  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "WARN: no checkpoints dir for ${run_name} (${ckpt_dir}); skipping eval"
    continue
  fi

  mapfile -t ckpts < <(find "${ckpt_dir}" -maxdepth 1 -type f -name '*.ckpt' | sort -V)
  if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "WARN: no .ckpt files in ${ckpt_dir}; skipping eval"
    continue
  fi

  echo "--> nqueens_eval on ${#ckpts[@]} checkpoint(s) for ${run_name}"
  for ckpt in "${ckpts[@]}"; do
    echo "    ckpt: ${ckpt}"
    # Architecture overrides (model/algo/infill/diffusion_forcing) must match
    # training so the checkpoint loads and the eval consumes the infill batch
    # layout. Results land in ${run_dir}/nqueens_eval/<ckpt_stem>/results.json
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
      || { echo "FAILED (eval): ${run_name} :: ${ckpt}"; continue; }

    # Plot next to the results.json this eval wrote.
    stem="$(basename "${ckpt%.*}")"
    results="${run_dir}/nqueens_eval/${stem}/results.json"
    if [[ -f "${results}" ]]; then
      python plot_nqueens.py "${results}"
    else
      echo "WARN: results.json not found at ${results}; skipping plot"
    fi
  done
done

echo ""
echo "=== done. results under ${SWEEP_ROOT}/<run_name>/nqueens_eval/<ckpt>/results.json ==="