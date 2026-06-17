#!/usr/bin/env bash
#
# sweep_flm_infill_sudoku_gen.sh — Train a sweep of plain FLM (algo=flm) on the
# generated sudoku dataset in the in-place *infilling* formulation, over a range
# of (deterministic) training-set sizes and difficulty levels, then run
# sudoku_eval on every checkpoint produced by each run.
#
# What this sweep fixes vs. varies:
#   - algo            : flm (with algo.infill=true)
#   - formulation     : in-place infilling -> model sees the solution grid and a
#                       conditioning_mask over the given clues (model=small_infill,
#                       length 81).
#   - loss region     : whole board (data.infill_loss_region=board), so the loss
#                       covers the clamped clue cells as well as the blanks.
#   - VARIED          : difficulty in {easy, medium, hard} and training-set size
#                       N in {1000, 10000}, selected deterministically via
#                       data.train_subset_n so the exact same N examples are
#                       reused across runs/seeds.
#
# All sizes subset from one shared generated pool of `NUM_TRAIN` examples (keyed
# by difficulty/seed and cached on disk), so generation happens once per
# difficulty level. A size equal to NUM_TRAIN uses the whole pool (no subsetting).
#
# Key trick (from the discrete-loop sweep): we pin `hydra.run.dir` to a per-run
# path keyed by run_name, so checkpoints land in a known location
# (outputs/<sweep>/<difficulty>/<run_name>/checkpoints) instead of hydra's
# timestamped dir, which makes them trivial to find and evaluate afterwards.
#
# Usage:
#   ./reproduce/sweep_flm_infill_sudoku_gen.sh
#   EVAL_ONLY=1 ./reproduce/sweep_flm_infill_sudoku_gen.sh          # skip training, eval existing ckpts
#   DIFFICULTIES="easy hard" SUBSET_SIZES="100 1000" ./reproduce/sweep_flm_infill_sudoku_gen.sh
#
set -euo pipefail

# --- Resolve repo root so the script works from any directory ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Sweep / run configuration (all overridable via env) --------------------
# Difficulty levels to sweep.
read -r -a difficulties <<< "${DIFFICULTIES:-hard}"

# Training-set sizes to sweep (the deterministic subset size).
read -r -a subset_sizes <<< "${SUBSET_SIZES:-1000 10000}"

# Shared generated pool size. Must be >= the largest subset size above; a subset
# equal to this uses the full pool.
NUM_TRAIN="${NUM_TRAIN:-10000}"
NUM_VALID="${NUM_VALID:-2000}"

GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
MAX_STEPS="${MAX_STEPS:-160001}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
# Plain FLM has no discrete timestep grid, so sudoku generation uses this many
# Euler steps (kept modest so the per-eval cost stays tractable).
SAMPLING_STEPS="${SAMPLING_STEPS:-128}"

conditioning_time_random="${CONDITIONING_TIME_RANDOM:-true}"
read -r -a conditioning_prob_clean_values <<< "${CONDITIONING_PROBS_CLEAN:-0.5 1.0}"

# Sweep root groups all difficulties together; difficulty is reflected in
# run_dir so outputs stay organized as:
#   <SWEEP_ROOT>/<difficulty>/<run_name>/checkpoints/
SWEEP_ROOT="${SWEEP_ROOT:-outputs/sweep-flm-infill-sudoku}"
EVAL_ONLY="${EVAL_ONLY:-0}"

for difficulty in "${difficulties[@]}"; do
  for cprob in "${conditioning_prob_clean_values[@]}"; do
    for N in "${subset_sizes[@]}"; do
      run_name="FLM_Infill_Sudoku_${difficulty}_board_N:${N}_cond_time_random:${conditioning_time_random}_clean_prob:${cprob}"
      run_dir="${SWEEP_ROOT}/${difficulty}/${run_name}"
      echo ""
      echo "=== ${run_name} ==="

      # --- Train --------------------------------------------------------------
      if [[ "${EVAL_ONLY}" != "1" ]]; then
        python main.py \
          data=sudoku-gen \
          data.difficulty="${difficulty}" \
          data.num_train="${NUM_TRAIN}" \
          data.num_valid="${NUM_VALID}" \
          data.train_subset_n="${N}" \
          data.infill_loss_region=board \
          model=small_infill \
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
          || { echo "FAILED (train): ${run_name}"; continue; }
      fi

      # --- Eval every checkpoint from this run --------------------------------
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

      echo "--> sudoku_eval on ${#ckpts[@]} checkpoint(s) for ${run_name}"
      for ckpt in "${ckpts[@]}"; do
        echo "    ckpt: ${ckpt}"
        # Architecture overrides (model/algo/infill) must match training so the
        # checkpoint loads and the eval consumes the infill batch layout. The
        # subset size doesn't affect generation, so it's omitted here. Results
        # land in ${run_dir}/sudoku_eval/<ckpt_stem>/results.json
        abs_ckpt="$(realpath "${ckpt}")"
        python main.py \
          mode=sudoku_eval \
          data=sudoku-gen \
          data.difficulty="${difficulty}" \
          data.num_train="${NUM_TRAIN}" \
          data.num_valid="${NUM_VALID}" \
          data.infill_loss_region=board \
          model=small_infill \
          algo=flm \
          algo.infill=true \
          algo.diffusion_forcing=true \
          loader.global_batch_size="${GLOBAL_BATCH}" \
          sampling.steps="${SAMPLING_STEPS}" \
          sampling.override_algo_steps=true \
          eval.checkpoint_path="${abs_ckpt}" \
          hydra.run.dir="${run_dir}" \
          || echo "FAILED (eval): ${run_name} :: ${ckpt}"
      done
    done
  done
done

echo ""
echo "=== done. results under ${SWEEP_ROOT}/<difficulty>/<run_name>/sudoku_eval/<ckpt>/results.json ==="