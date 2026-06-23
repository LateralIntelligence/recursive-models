#!/usr/bin/env bash
#
# sweep_infill_nqueens_ablation.sh — generalized infill-FLM N-Queens sweep used
# by the conditioning-noise ablations. It is a superset of
# reproduce/sweep_infill_nqueens.sh: same train -> checkpoint -> nqueens_eval ->
# plot flow, with two extra knobs so the same script drives the ablations from
# environment variables (mirrors reproduce/sweep_infill_sudoku_ablation.sh):
#
#   INFILL_LOSS_REGION  board | fill   (data.infill_loss_region)
#       - board: loss over the whole grid (clue + blank cells)        [default]
#       - fill : loss over the blank cells only (predict response only)
#   SEPARATE_COND_TIME  true | false   (algo.separate_conditioning_time)
#       - true : conditioning tokens get their own independent noise level t'
#                (conditioning_prob_clean is then ignored at train time)
#
# Everything else (board size N/DATA/MODEL_LENGTH, batch, steps, conditioning
# clean probabilities, eval sample/puzzle counts) is identical to the base
# nqueens sweep and overridable via env.
#
# Usage examples:
#   # Ablation 3 (separate conditioning noise level), N-Queens 8x8:
#   N=8 DATA=nqueens MODEL_LENGTH=64 SEPARATE_COND_TIME=true \
#     CONDITIONING_PROBS_CLEAN="1.0" ./reproduce/sweep_infill_nqueens_ablation.sh
#
#   # Ablation 3, N-Queens 10x10:
#   N=10 DATA=nqueens-10 MODEL_LENGTH=100 SEPARATE_COND_TIME=true \
#     CONDITIONING_PROBS_CLEAN="1.0" ./reproduce/sweep_infill_nqueens_ablation.sh
#
#   EVAL_ONLY=1 ... ./reproduce/sweep_infill_nqueens_ablation.sh  # eval existing ckpts
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

# --- Ablation knobs ---------------------------------------------------------
INFILL_LOSS_REGION="${INFILL_LOSS_REGION:-board}"   # board | fill
SEPARATE_COND_TIME="${SEPARATE_COND_TIME:-false}"   # true | false

conditioning_time_random="${CONDITIONING_TIME_RANDOM:-true}"
read -r -a conditioning_prob_clean_values <<< "${CONDITIONING_PROBS_CLEAN:-0.2 0.5 1.0}"

# --- Eval config ------------------------------------------------------------
NUM_SAMPLES="${NUM_SAMPLES:-20}"        # eval.nqueens_num_samples
NUM_PUZZLES="${NUM_PUZZLES:-750}"       # eval.nqueens_num_puzzles

SWEEP_ROOT="${SWEEP_ROOT:-outputs/sweep-flm-infill-nqueens-${N}}"
EVAL_ONLY="${EVAL_ONLY:-0}"

# Extra hydra overrides go to TRAINING only (matching sweep_infill_nqueens.sh).
EXTRA_OVERRIDES=("$@")

echo "=== nqueens ablation sweep config ==="
echo "  N=${N}  DATA=${DATA}  MODEL_LENGTH=${MODEL_LENGTH}"
echo "  GLOBAL_BATCH=${GLOBAL_BATCH}  MAX_STEPS=${MAX_STEPS}"
echo "  INFILL_LOSS_REGION=${INFILL_LOSS_REGION}  SEPARATE_COND_TIME=${SEPARATE_COND_TIME}"
echo "  CONDITIONING_PROBS_CLEAN=${conditioning_prob_clean_values[*]}"
echo "  SWEEP_ROOT=${SWEEP_ROOT}"
echo "====================================="

for cprob in "${conditioning_prob_clean_values[@]}"; do
  run_name="FLM_Infill_NQueens_N:${N}_region:${INFILL_LOSS_REGION}_septime:${SEPARATE_COND_TIME}_cond_time_random:${conditioning_time_random}_clean_prob:${cprob}"
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
      data.infill_loss_region="${INFILL_LOSS_REGION}" \
      model=nqueens_infill \
      model.length="${MODEL_LENGTH}" \
      algo=flm \
      algo.infill=true \
      algo.conditioning_time_random="${conditioning_time_random}" \
      algo.conditioning_prob_clean="${cprob}" \
      algo.separate_conditioning_time="${SEPARATE_COND_TIME}" \
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
    # Eval always feeds clean conditioning during sampling, so the ablation
    # train-time flags don't change generation; we still pass the architecture
    # overrides so the checkpoint loads and the infill batch layout matches.
    # Results land in ${run_dir}/nqueens_eval/<ckpt_stem>/results.json
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
