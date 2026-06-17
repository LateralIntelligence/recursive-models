#!/usr/bin/env bash
set -euo pipefail

gammas=(0.0 0.5 1.0)
timesteps=(6 12 24)
backprop=(2 6 12)

for gamma in "${gammas[@]}"; do
  for T in "${timesteps[@]}"; do
    for bp in "${backprop[@]}"; do
      if (( bp > T )); then
        echo "skip: backprop=${bp} > timesteps=${T}"
        continue
      fi
      run_name="Disc_FLM_Looped_g=${gamma}_T=${T}_bp=${bp}"
      echo "=== ${run_name} ==="
      python main.py \
        data=mnist \
        model=mnist \
        algo=discrete_loop_flm \
        algo.gamma="${gamma}" \
        algo.num_timesteps="${T}" \
        algo.backprop_steps="${bp}" \
        loader.global_batch_size=16 \
        trainer.max_steps=20001 \
        "wandb.name='${run_name}'" \
        || echo "FAILED: ${run_name}"
    done
  done
done