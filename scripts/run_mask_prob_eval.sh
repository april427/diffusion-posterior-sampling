#!/bin/bash
# Automated evaluation of NMSE vs mask probability - Multi-GPU version
# Results are saved to separate directories per GPU, then aggregated

# Configuration
MODEL_CONFIG="configs/aoa_amp_building_config.yaml"
DIFFUSION_CONFIG="configs/diffusion_config.yaml"
TASK_CONFIG="configs/aoa_amp_building_inpainting.yaml"
NUM_TEST_SAMPLES=2
SEED=42
SAVE_DIR="./results/mask_prob_eval"

# Define GPUs and mask probabilities
GPUS=(0)  # Adjust to your available GPUs, e.g., GPUS=(0 1 2 3)
ALL_MASK_PROBS=(0.75 0.8 0.85 0.9 0.95)

# Optional: noise sigmas (comment out to use task_config default)
# NOISE_SIGMAS="0.01,0.025,0.05"

# Set to true to load full dataset with metadata (uses more memory)
# By default, tensor-only mode is used (lighter memory, avoids CUDA OOM)
FULL_DATASET=false

echo "=============================================="
echo "Multi-GPU Mask Probability Evaluation"
echo "=============================================="
echo "Using GPUs: ${GPUS[*]}"
echo "Mask probabilities: ${ALL_MASK_PROBS[*]}"
echo "Full dataset mode: ${FULL_DATASET}"
echo "Save directory: ${SAVE_DIR}"
echo ""

# Function to run evaluation on a specific GPU
run_on_gpu() {
    local gpu=$1
    local mask_prob=$2
    local save_subdir="${SAVE_DIR}/gpu${gpu}_mask${mask_prob}"
    
    mkdir -p "${save_subdir}"
    
    echo "[GPU ${gpu}] Starting mask_prob=${mask_prob}"
    
    # Build command
    CMD="python3 evaluate_mask_prob.py \
        --model_config ${MODEL_CONFIG} \
        --diffusion_config ${DIFFUSION_CONFIG} \
        --task_config ${TASK_CONFIG} \
        --gpu ${gpu} \
        --num_test_samples ${NUM_TEST_SAMPLES} \
        --mask_probs ${mask_prob} \
        --seed ${SEED} \
        --save_dir ${save_subdir}"
    
    # Add full_dataset flag if enabled
    if [ "$FULL_DATASET" = true ]; then
        CMD="${CMD} --full_dataset"
    fi
    
    # Add noise sigmas if specified
    if [ -n "$NOISE_SIGMAS" ]; then
        CMD="${CMD} --noise_sigmas ${NOISE_SIGMAS}"
    fi
    
    # Run and log
    eval ${CMD} 2>&1 | tee "${save_subdir}.log"
    
    echo "[GPU ${gpu}] Finished mask_prob=${mask_prob}"
}

# Create save directory
mkdir -p ${SAVE_DIR}

# Launch jobs in parallel across GPUs
job_idx=0
pids=()

for mask_prob in "${ALL_MASK_PROBS[@]}"; do
    gpu_idx=$((job_idx % ${#GPUS[@]}))
    gpu=${GPUS[$gpu_idx]}
    
    run_on_gpu $gpu $mask_prob &
    pids+=($!)
    
    job_idx=$((job_idx + 1))
    
    # If we've launched jobs on all GPUs, wait for them to finish
    if [ $((job_idx % ${#GPUS[@]})) -eq 0 ]; then
        echo "Waiting for batch to complete..."
        wait "${pids[@]}"
        pids=()
    fi
done

# Wait for remaining jobs
if [ ${#pids[@]} -gt 0 ]; then
    echo "Waiting for final batch to complete..."
    wait "${pids[@]}"
fi

echo ""
echo "=============================================="
echo "All GPU evaluations complete!"
echo "=============================================="
echo ""

# Aggregate results
echo "Aggregating results from all GPUs..."
python3 scripts/aggregate_results.py --results_dir ${SAVE_DIR}

echo ""
echo "=============================================="
echo "Done! Check ${SAVE_DIR}/combined_* for plots"
echo "=============================================="
