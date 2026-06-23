#!/usr/bin/env bash
#
# eval_ablation_last_ckpts.sh — evaluate the final (last.ckpt) checkpoint of
# every ablation run under ablations/, matching each run's training params
# (read off the run-dir name / config_tree.txt). Results land next to each
# checkpoint at <run>/{sudoku_eval,nqueens_eval}/last/results.json.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SAMPLING_STEPS=128

eval_sudoku () {
  local run_dir="$1" difficulty="$2" region="$3" gb="$4"
  local ckpt="${run_dir}/checkpoints/last.ckpt"
  [[ -f "${ckpt}" ]] || { echo "SKIP (no last.ckpt): ${run_dir}"; return; }
  [[ -f "${run_dir}/sudoku_eval/last/results.json" && "${FORCE:-0}" != "1" ]] && { echo "SKIP (already evaluated): ${run_dir}"; return; }
  local abs_ckpt; abs_ckpt="$(realpath "${ckpt}")"
  echo ""
  echo "=== SUDOKU eval: ${run_dir} (diff=${difficulty} region=${region}) ==="
  python main.py \
    mode=sudoku_eval \
    data=sudoku-gen \
    data.difficulty="${difficulty}" \
    data.num_train=10000 \
    data.num_valid=2000 \
    data.infill_loss_region="${region}" \
    model=small_infill \
    algo=flm \
    algo.infill=true \
    algo.diffusion_forcing=true \
    loader.global_batch_size="${gb}" \
    trainer.devices=1 \
    sampling.steps="${SAMPLING_STEPS}" \
    sampling.override_algo_steps=true \
    eval.checkpoint_path="${abs_ckpt}" \
    || echo "FAILED (sudoku eval): ${run_dir}"
}

eval_nqueens () {
  local run_dir="$1" data="$2" n="$3" length="$4"
  local ckpt="${run_dir}/checkpoints/last.ckpt"
  [[ -f "${ckpt}" ]] || { echo "SKIP (no last.ckpt): ${run_dir}"; return; }
  [[ -f "${run_dir}/nqueens_eval/last/results.json" && "${FORCE:-0}" != "1" ]] && { echo "SKIP (already evaluated): ${run_dir}"; return; }
  local abs_ckpt; abs_ckpt="$(realpath "${ckpt}")"
  echo ""
  echo "=== NQUEENS eval: ${run_dir} (data=${data} n=${n} len=${length}) ==="
  python main.py \
    mode=nqueens_eval \
    data="${data}" \
    data.nqueens_n="${n}" \
    model=nqueens_infill \
    model.length="${length}" \
    algo=flm \
    algo.infill=true \
    algo.diffusion_forcing=true \
    trainer.devices=1 \
    sampling.steps="${SAMPLING_STEPS}" \
    sampling.override_algo_steps=true \
    eval.nqueens_num_samples=20 \
    eval.nqueens_num_puzzles=750 \
    eval.checkpoint_path="${abs_ckpt}" \
    || echo "FAILED (nqueens eval): ${run_dir}"
}

A1=ablations/ablation1-zerocond
A2=ablations/ablation2-fillonly
A3=ablations/ablation3-septime

# --- Ablation 1: zero conditional training (region=board, clean_prob=0) ------
eval_sudoku "${A1}/sweep-flm-infill-sudoku-easy/FLM_Infill_Sudoku_easy_region:board_septime:false_N:10000_cond_time_random:true_clean_prob:0" easy board 32
eval_sudoku "${A1}/sweep-flm-infill-sudoku-hard/FLM_Infill_Sudoku_hard_region:board_septime:false_N:10000_cond_time_random:true_clean_prob:0" hard board 256

# --- Ablation 2: predict response only (region=fill), sudoku easy p-sweep -----
for p in 0 0.2 0.5 1.0; do
  eval_sudoku "${A2}/sweep-flm-infill-sudoku-easy/FLM_Infill_Sudoku_easy_region:fill_septime:false_N:10000_cond_time_random:true_clean_prob:${p}" easy fill 32
done

# --- Ablation 3: separate conditioning noise level (septime=true, p=1.0) ------
eval_sudoku "${A3}/sweep-flm-infill-sudoku-easy/FLM_Infill_Sudoku_easy_region:board_septime:true_N:10000_cond_time_random:true_clean_prob:1.0" easy board 32
eval_sudoku "${A3}/sweep-flm-infill-sudoku-hard/FLM_Infill_Sudoku_hard_region:board_septime:true_N:10000_cond_time_random:true_clean_prob:1.0" hard board 256
eval_nqueens "${A3}/sweep-flm-infill-nqueens-8/FLM_Infill_NQueens_N:8_region:board_septime:true_cond_time_random:true_clean_prob:1.0" nqueens 8 64
eval_nqueens "${A3}/sweep-flm-infill-nqueens-10/FLM_Infill_NQueens_N:10_region:board_septime:true_cond_time_random:true_clean_prob:1.0" nqueens-10 10 100

echo ""
echo "=== done. results under <run>/{sudoku_eval,nqueens_eval}/last/results.json ==="
