#!/bin/bash

CHECKPOINT="/home/jsjung00/Desktop/Code/recursive-models/outputs/incorrect-cont-sweep/Continuous_FLM_Looped_Sudoku_gen_hard_g:0/checkpoints/167-120000.ckpt"

for steps in 6 12 32 64 128 256; do
    echo "Running evaluation with sampling.steps=${steps}"

    python main.py \
        mode=sudoku_eval \
        data=sudoku-gen \
        model=small \
        loader.global_batch_size=256 \
        algo=discrete_loop_flm \
        eval.checkpoint_path="${CHECKPOINT}" \
        sampling.override_algo_steps=True \
        sampling.steps="${steps}"
done