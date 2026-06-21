#!/usr/bin/env bash
#
# eval_tinygsm_gsm8k.sh — Score a TinyGSM-trained FLM checkpoint on the GSM8K
# test benchmark, following https://github.com/jdeschena/s-flm (sample/tinygsm).
#
# Conditionally generates the answer region for each GSM8K test prompt (the
# model emits Python code defining `simple_math_problem`), executes it in
# sandbox_gsm8k, and compares to the gold answer. Writes results.json with the
# accuracy and per-example records.
#
# Usage:
#   CKPT_PATH=/path/to/checkpoint.ckpt ./reproduce/eval_tinygsm_gsm8k.sh
#   CKPT_PATH=... STEPS=64 ./reproduce/eval_tinygsm_gsm8k.sh [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CKPT_PATH="${CKPT_PATH:?set CKPT_PATH to the checkpoint to evaluate}"
TOKENIZER="${TOKENIZER:-HuggingFaceTB/SmolLM-135M}"
MODEL_LENGTH="${MODEL_LENGTH:-512}"
STEPS="${STEPS:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BOOTSTRAP_SIZE="${BOOTSTRAP_SIZE:-0}"
TIMEOUT="${TIMEOUT:-5.0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

cmd=(python main.py
  mode=gsm8k_eval
  eval.checkpoint_path="${CKPT_PATH}"
  data=gsm8k-test
  data.tokenizer_name_or_path="${TOKENIZER}"
  model=tinygsm
  model.length="${MODEL_LENGTH}"
  algo=flm
  algo.infill=true
  sampling.steps="${STEPS}"
  sampling.override_algo_steps=true
  loader.eval_batch_size="${EVAL_BATCH_SIZE}"
  loader.num_workers="${NUM_WORKERS}"
  gsm8k.timeout="${TIMEOUT}"
  gsm8k.bootstrap_size="${BOOTSTRAP_SIZE}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(gsm8k.output_dir="${OUTPUT_DIR}")
fi

"${cmd[@]}" "$@"