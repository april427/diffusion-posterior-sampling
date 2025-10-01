#!/bin/bash
# Training script for AoA/Amplitude diffusion model

GPU=${1:-0}
CHECKPOINT_DIR=${2:-./checkpoints/aoa_amp}
LOG_DIR=${3:-./logs/aoa_amp}

echo "Starting AoA/Amplitude diffusion model training..."
echo "GPU: $GPU"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Log directory: $LOG_DIR"

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR
mkdir -p ./data/aoa_amp_cache

# Run training
python train_dps.py \
  --model_config configs/aoa_amp_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --train_config configs/train_dps_config.yaml \
  --gpu 0


echo "Training completed!"