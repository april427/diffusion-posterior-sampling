#!/bin/bash
# Training script for AoA/Amplitude diffusion model with buildings

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories
mkdir -p ./checkpoints/aoa_amp_building
mkdir -p ./logs/aoa_amp_building
mkdir -p ./data/building_training

echo "Starting training for AoA/Amplitude diffusion model with buildings..."

# First, generate training data (optional - use --generate_data flag)
python3 train_aoa_amp_building.py \
    --model_config configs/aoa_amp_building_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --checkpoint_dir ./checkpoints/aoa_amp_building \
    --log_dir ./logs/aoa_amp_building \
    --gpu 0 \
    "$@"

echo "Training completed!"

# Optional: Monitor training with tensorboard
echo "To monitor training, run:"
echo "tensorboard --logdir=./logs/aoa_amp_building"
