#!/usr/bin/env bash
#
# sweep_disc_flm_sudoku_gen.sh — Train a (gamma x timesteps x backprop) sweep
# of DiscreteLoopFLM on the generated sudoku dataset, then run sudoku_eval on
# every checkpoint produced by each run.
#
# Key trick: we pin `hydra.run.dir` to a per-run path keyed by run_name, so the
# checkpoints land in a known location (outputs/sweep/<run_name>/checkpoints)
# instead of hydra's default timestamped dir. That makes them trivial to find
# and evaluate afterwards.
#
# Usage:
#   ./reproduce/sweep_disc_flm_sudoku_gen.sh
#   EVAL_ONLY=1 ./reproduce/sweep_disc_flm_sudoku_gen.sh   # skip training, just eval existing ckpts
#
set -euo pipefail

# --- Resolve repo root so the script works from any directory ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
gammas=(0.0 0.5)
timesteps=(6 12)
backprop=(6 12)
discrete_time=(False)

SWEEP_ROOT="outputs/sweep"
EVAL_ONLY="${EVAL_ONLY:-0}"

for gamma in "${gammas[@]}"; do
  for T in "${timesteps[@]}"; do
    for bp in "${backprop[@]}"; do
      for disc in "${discrete_time[@]}"; do
        if (( bp != T )); then
          echo "skip: backprop=${bp} > timesteps=${T}"
          continue
        fi
        run_name="Cont_FLM_Looped_Sudoku_gen_g:${gamma}_T:${T}_bp:${bp}"
        run_dir="${SWEEP_ROOT}/${run_name}"
        echo ""
        echo "=== ${run_name} ==="

        # --- Train --------------------------------------------------------------
        if [[ "${EVAL_ONLY}" != "1" ]]; then
          python main.py \
            data=sudoku-gen \
            model=small \
            algo=discrete_loop_flm \
            algo.gamma="${gamma}" \
            algo.num_timesteps="${T}" \
            algo.backprop_steps="${bp}" \
            algo.discrete_denoiser_time="${disc}" \
            loader.global_batch_size=32 \
            trainer.max_steps=120001 \
            trainer.val_check_interval=10000 \
            trainer.check_val_every_n_epoch=null \
            hydra.run.dir="${run_dir}" \
            callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
            "wandb.name='${run_name}'" \
            || { echo "FAILED (train): ${run_name}"; continue; }
        fi

        # --- Eval every checkpoint from this run --------------------------------
        ckpt_dir="${run_dir}/checkpoints"
        if [[ ! -d "${ckpt_dir}" ]]; then
          echo "WARN: no checkpoints dir for ${run_name} (${ckpt_dir}); skipping eval"
          continue
        fi

        mapfile -t ckpts < <(find "${ckpt_dir}" -maxdepth 1 -type f -name 'best_nll.ckpt' | sort -V)
        if [[ ${#ckpts[@]} -eq 0 ]]; then
          echo "WARN: no .ckpt files in ${ckpt_dir}; skipping eval"
          continue
        fi

        echo "--> sudoku_eval on ${#ckpts[@]} checkpoint(s) for ${run_name}"
        for ckpt in "${ckpts[@]}"; do
          echo "    ckpt: ${ckpt}"
          # Architecture overrides (model/algo) must match training so the
          # checkpoint loads; gamma/T/bp don't affect generation, so they're
          # omitted. Results land in ${run_dir}/sudoku_eval/<ckpt_stem>/results.json
          abs_ckpt="$(realpath "${ckpt}")"
          python main.py \
            mode=sudoku_eval \
            data=sudoku-gen \
            model=small \
            loader.global_batch_size=32 \
            algo=discrete_loop_flm \
            eval.checkpoint_path="${abs_ckpt}" \
            || echo "FAILED (eval): ${run_name} :: ${ckpt}"
        done
      done
    done
  done
done

echo ""
echo "=== done. results under ${SWEEP_ROOT}/<run_name>/sudoku_eval/<ckpt>/results.json ==="