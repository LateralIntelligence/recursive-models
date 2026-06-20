#!/usr/bin/env bash
#
# train_tinygsm.sh — Train the default FLM (algo=flm) on the TinyGSM grade-school
# math benchmark, following https://github.com/jdeschena/s-flm (flm_default).
#
# Each example is [BOS] question <sep> answer [EOS] padded to model.length and
# tokenized with SmolLM-135M. The question (prompt) is the conditioning region;
# the loss covers the answer (+ padding, train_on_pad=True).
#
# Experiment knob: unlike s-flm (which always keeps the prompt clean), we can
# noise the prompt tokens for a fraction (1 - conditioning_prob_clean) of
# sequences. Sweep CONDITIONING_PROBS_CLEAN to vary it; clean_prob=1.0
# reproduces the s-flm clean-prompt setup.
#
# Usage:
#   ./reproduce/train_tinygsm.sh [extra hydra overrides...]
#   CONDITIONING_PROBS_CLEAN="0.5 1.0" ./reproduce/train_tinygsm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TOKENIZER="${TOKENIZER:-HuggingFaceTB/SmolLM-135M}"
MODEL_LENGTH="${MODEL_LENGTH:-512}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-250000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-10000}"
CKPT_EVERY="${CKPT_EVERY:-10000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/flm-tinygsm}"

conditioning_time_random="${CONDITIONING_TIME_RANDOM:-true}"
read -r -a conditioning_prob_clean_values <<< "${CONDITIONING_PROBS_CLEAN:-0.2 0.5 1.0}"

for cprob in "${conditioning_prob_clean_values[@]}"; do
  run_name="FLM_TinyGSM_len:${MODEL_LENGTH}_cond_time_random:${conditioning_time_random}_clean_prob:${cprob}"
  run_dir="${SWEEP_ROOT}/${run_name}"
  echo ""
  echo "=== ${run_name} ==="

  python main.py \
    data=tinygsm \
    data.tokenizer_name_or_path="${TOKENIZER}" \
    data.wrap=False \
    data.train_on_prompt=False \
    data.train_on_pad=True \
    data.filter_too_long=True \
    model=tinygsm \
    model.length="${MODEL_LENGTH}" \
    algo=flm \
    algo.infill=true \
    algo.diffusion_forcing=true \
    algo.conditioning_time_random="${conditioning_time_random}" \
    algo.conditioning_prob_clean="${cprob}" \
    loader.global_batch_size="${GLOBAL_BATCH}" \
    loader.batch_size="${BATCH_SIZE}" \
    loader.num_workers="${NUM_WORKERS}" \
    eval.generate_samples=False \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.val_check_interval="${VAL_CHECK_INTERVAL}" \
    trainer.limit_val_batches="${LIMIT_VAL_BATCHES}" \
    trainer.check_val_every_n_epoch=null \
    hydra.run.dir="${run_dir}" \
    "wandb.name='${run_name}'" \
    callbacks.checkpoint_every_n_steps.every_n_train_steps="${CKPT_EVERY}" \
    "$@" || { echo "!!! run failed for clean_prob=${cprob}, continuing"; continue; }
done

echo ""
echo "=== done. checkpoints under ${SWEEP_ROOT}/<run_name>/checkpoints ==="