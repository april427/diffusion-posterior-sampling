#!/bin/bash
# Conditional sampling script for the AoA/Amplitude building model

set -e

python3 sample_condition_building.py \
  --model_config configs/aoa_amp_building_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/aoa_amp_building_inpainting.yaml \
  --gpu 0 \
  --save_dir ./results/aoa_amp_building
