#!/usr/bin/env bash
set -euo pipefail

gammas=(0.0 0.5 1.0)

for gamma in "${gammas[@]}"; do
    run_name="Disc_FLM_Looped_gamma=${gamma}"
    echo "=== gamma=${gamma} | run_name=${run_name} ==="
    python main.py \
        data=mnist \
        model=mnist \
        algo=discrete_loop_flm \
        algo.gamma="${gamma}" \
        algo.num_timesteps=12 \
        loader.global_batch_size=16 \
        trainer.max_steps=20001 \
        "wandb.name='${run_name}'"
done