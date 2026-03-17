#!/bin/bash
# Training script for three-path (AoA+Amp) 6-channel building experiment

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PYTHON_BIN="python3"
if [ -x "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
fi

mkdir -p ./checkpoints/aoa_amp_building_path3
mkdir -p ./logs/aoa_amp_building_path3

echo "Starting three-path training (AoA1-3 + Amp1-3, 6 channels)..."

"${PYTHON_BIN}" train_aoa_amp_building.py \
    --model_config configs/aoa_amp_building_aoa3_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --checkpoint_dir ./checkpoints/aoa_amp_building_path3 \
    --log_dir ./logs/aoa_amp_building_path3 \
    --gpu 0 \
    "$@"

echo "Three-path training completed."
