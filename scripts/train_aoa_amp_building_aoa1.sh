#!/bin/bash
# Training script for first-path (AoA+Amp) 2-channel building experiment

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHON_BIN="python3"
if [ -x "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
fi

mkdir -p ./checkpoints/aoa_amp_building_path1
mkdir -p ./logs/aoa_amp_building_path1

echo "Starting first-path training (AoA1 + Amp1, 2 channels)..."

"${PYTHON_BIN}" train_aoa_amp_building.py \
    --model_config configs/aoa_amp_building_aoa1_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --checkpoint_dir ./checkpoints/aoa_amp_building_path1 \
    --log_dir ./logs/aoa_amp_building_path1 \
    --gpu 0 \
    "$@"

echo "First-path training completed."
