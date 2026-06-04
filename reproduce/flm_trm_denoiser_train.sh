#!/usr/bin/env bash
# --- Resolve repo root so the script works from any directory ---------------
# This script lives in <repo>/reproduce/, but main.py is at the repo root,
# so we go up one level from the script's own directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python main.py \
    model=trm \
    algo.backbone=trm \
    wandb.name=flm_trm_denoiser_training 