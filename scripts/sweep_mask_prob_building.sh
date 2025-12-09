#!/bin/bash
# Sweep mask_prob_range and record per-channel NMSE

set -euo pipefail

START=0.2
END=0.8
STEP=0.1
SAMPLES=100
GPU=0
SAVE_DIR=./results/aoa_amp_building
OP_DIR=${SAVE_DIR}/inpainting
METRICS_DIR=${OP_DIR}/metrics

mkdir -p "${METRICS_DIR}"

# Write header for sweep summary CSV
SUMMARY_CSV="${METRICS_DIR}/nmse_sweep_summary.csv"
if [ ! -f "${SUMMARY_CSV}" ]; then
  echo "mask_prob,samples,avg_total_nmse,ch1,ch2,ch3,ch4,ch5,ch6" > "${SUMMARY_CSV}"
fi

for p in $(seq ${START} ${STEP} ${END}); do
  echo "==> Running sampling for mask_prob=${p} with ${SAMPLES} samples"
  conda run -n DPS python sample_condition_building.py \
    --model_config configs/aoa_amp_building_config.yaml \
    --diffusion_config configs/diffusion_config.yaml \
    --task_config configs/aoa_amp_building_inpainting.yaml \
    --gpu ${GPU} \
    --save_dir "${SAVE_DIR}" \
    --split all \
    --mask_prob ${p} \
    --num_samples ${SAMPLES}

  # Append summary line
  MASK_PROB=${p} conda run -n DPS python - <<'PY'
import os, json, sys
op_dir = './results/aoa_amp_building/inpainting/metrics'
summary_csv = os.path.join(op_dir, 'nmse_sweep_summary.csv')
mask_prob = float(os.environ.get('MASK_PROB', '0'))
summary_path = os.path.join(op_dir, f"nmse_summary_mask_{mask_prob:.2f}.json")
if not os.path.exists(summary_path):
    print(f"Summary not found: {summary_path}", file=sys.stderr)
    sys.exit(0)
with open(summary_path, 'r') as f:
    data = json.load(f)
row = [
    f"{mask_prob:.2f}",
    str(data.get('samples', 0)),
    f"{data.get('avg_total_nmse', 0.0):.6e}",
]
avg_ch = data.get('avg_channel_nmse', [])
row += [f"{v:.6e}" for v in avg_ch]
with open(summary_csv, 'a') as f:
    f.write(','.join(row) + '\n')
PY
done

echo "Sweep complete. Summary: ${SUMMARY_CSV}"
