#!/usr/bin/env bash
#
# sweep_infill_sudoku_ablation.sh — generalized infill-FLM sweep used by the
# conditioning-noise ablations. It is a superset of
# reproduce/sweep_infill_sudoku_gen{,_hard}.sh: same train -> checkpoint ->
# sudoku_eval flow, with two extra knobs so the same script drives all three
# ablations from environment variables:
#
#   INFILL_LOSS_REGION  board | fill   (data.infill_loss_region)
#       - board: loss over the whole grid (clue + blank cells)        [default]
#       - fill : loss over the blank cells only (predict response only)
#   SEPARATE_COND_TIME  true | false   (algo.separate_conditioning_time)
#       - true : conditioning tokens get their own independent noise level t'
#                (conditioning_prob_clean is then ignored at train time)
#
# Everything else (difficulty, batch, steps, subset sizes, conditioning clean
# probabilities) is identical to the base sweeps and overridable via env.
#
# Usage examples:
#   # Ablation 1 (zero conditional training), sudoku hard, p=0:
#   DIFFICULTY=hard GLOBAL_BATCH=256 MAX_STEPS=40001 \
#     CONDITIONING_PROBS_CLEAN="0" SUBSET_SIZES="10000" \
#     ./reproduce/sweep_infill_sudoku_ablation.sh
#
#   # Ablation 2 (predict response only), sudoku easy, p-sweep, N=10k:
#   DIFFICULTY=easy GLOBAL_BATCH=256 MAX_STEPS=40001 INFILL_LOSS_REGION=fill \
#     CONDITIONING_PROBS_CLEAN="0 0.2 0.5 1.0" SUBSET_SIZES="10000" \
#     ./reproduce/sweep_infill_sudoku_ablation.sh
#
#   # Ablation 3 (separate conditioning noise level), sudoku easy, N=10k:
#   DIFFICULTY=easy GLOBAL_BATCH=256 MAX_STEPS=40001 SEPARATE_COND_TIME=true \
#     CONDITIONING_PROBS_CLEAN="1.0" SUBSET_SIZES="10000" \
#     ./reproduce/sweep_infill_sudoku_ablation.sh
#
#   EVAL_ONLY=1 ... ./reproduce/sweep_infill_sudoku_ablation.sh  # eval existing ckpts
#
set -euo pipefail

# --- Resolve repo root so the script works from any directory ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Sweep / run configuration (all overridable via env) --------------------
read -r -a subset_sizes <<< "${SUBSET_SIZES:-10000}"

DIFFICULTY="${DIFFICULTY:-hard}"          # easy / medium / hard
NUM_TRAIN="${NUM_TRAIN:-10000}"
NUM_VALID="${NUM_VALID:-2000}"

GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MAX_STEPS="${MAX_STEPS:-40001}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-128}"

# --- Ablation knobs ---------------------------------------------------------
MODEL="${MODEL:-small_infill}"                      # infill model config (e.g. single_infill)
INFILL_LOSS_REGION="${INFILL_LOSS_REGION:-board}"   # board | fill
SEPARATE_COND_TIME="${SEPARATE_COND_TIME:-false}"   # true | false

conditioning_time_random="${CONDITIONING_TIME_RANDOM:-true}"
read -r -a conditioning_prob_clean_values <<< "${CONDITIONING_PROBS_CLEAN:-0.2 0.5 1.0}"

SWEEP_ROOT="${SWEEP_ROOT:-outputs/sweep-flm-infill-sudoku-${DIFFICULTY}}"
EVAL_ONLY="${EVAL_ONLY:-0}"

echo "=== ablation sweep config ==="
echo "  DIFFICULTY=${DIFFICULTY}  GLOBAL_BATCH=${GLOBAL_BATCH}  MAX_STEPS=${MAX_STEPS}"
echo "  MODEL=${MODEL}  INFILL_LOSS_REGION=${INFILL_LOSS_REGION}  SEPARATE_COND_TIME=${SEPARATE_COND_TIME}"
echo "  CONDITIONING_PROBS_CLEAN=${conditioning_prob_clean_values[*]}  SUBSET_SIZES=${subset_sizes[*]}"
echo "  SWEEP_ROOT=${SWEEP_ROOT}"
echo "============================="

for cprob in "${conditioning_prob_clean_values[@]}"; do
  for N in "${subset_sizes[@]}"; do
    run_name="FLM_Infill_Sudoku_${DIFFICULTY}_region:${INFILL_LOSS_REGION}_septime:${SEPARATE_COND_TIME}_N:${N}_cond_time_random:${conditioning_time_random}_clean_prob:${cprob}"
    run_dir="${SWEEP_ROOT}/${run_name}"
    echo ""
    echo "=== ${run_name} ==="

    # --- Train ----------------------------------------------------------------
    if [[ "${EVAL_ONLY}" != "1" ]]; then
      python main.py \
        data=sudoku-gen \
        data.difficulty="${DIFFICULTY}" \
        data.num_train="${NUM_TRAIN}" \
        data.num_valid="${NUM_VALID}" \
        data.train_subset_n="${N}" \
        data.infill_loss_region="${INFILL_LOSS_REGION}" \
        model="${MODEL}" \
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
        || { echo "FAILED (train): ${run_name}"; continue; }
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

    echo "--> sudoku_eval on ${#ckpts[@]} checkpoint(s) for ${run_name}"
    for ckpt in "${ckpts[@]}"; do
      echo "    ckpt: ${ckpt}"
      # Eval always feeds clean conditioning during sampling, so the ablation
      # train-time flags don't change generation; we still pass the architecture
      # overrides so the checkpoint loads and the infill batch layout matches.
      abs_ckpt="$(realpath "${ckpt}")"
      python main.py \
        mode=sudoku_eval \
        data=sudoku-gen \
        data.difficulty="${DIFFICULTY}" \
        data.num_train="${NUM_TRAIN}" \
        data.num_valid="${NUM_VALID}" \
        data.infill_loss_region="${INFILL_LOSS_REGION}" \
        model="${MODEL}" \
        algo=flm \
        algo.infill=true \
        algo.diffusion_forcing=true \
        loader.global_batch_size="${GLOBAL_BATCH}" \
        sampling.steps="${SAMPLING_STEPS}" \
        sampling.override_algo_steps=true \
        eval.checkpoint_path="${abs_ckpt}" \
        || echo "FAILED (eval): ${run_name} :: ${ckpt}"
    done
  done
done

echo ""
echo "=== done. results under ${SWEEP_ROOT}/<run_name>/sudoku_eval/<ckpt>/results.json ==="
