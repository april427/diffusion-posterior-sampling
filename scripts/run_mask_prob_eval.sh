#!/bin/bash
# Automated evaluation of NMSE vs noise sigma - Multi-GPU version
# Results are saved to separate directories per GPU, then aggregated

# Configuration
MODEL_CONFIG="configs/aoa_amp_building_config.yaml"
DIFFUSION_CONFIG="configs/diffusion_config.yaml"
TASK_CONFIG="configs/aoa_amp_building_inpainting.yaml"
NUM_TEST_SAMPLES=2
SEED=42
SAVE_DIR="./results/noise_sigma_eval"

# Define GPUs and noise sigmas
GPUS=(0)  # Adjust to your available GPUs, e.g., GPUS=(0 1 2 3)
ALL_NOISE_SIGMAS=(0.562 0.316 0.178 0.1 0.05)

# Fixed mask probability
MASK_PROB=0.8

# Set to true to load full dataset with metadata (uses more memory)
# By default, tensor-only mode is used (lighter memory, avoids CUDA OOM)
FULL_DATASET=false

echo "=============================================="
echo "Multi-GPU Noise Sigma Evaluation"
echo "=============================================="
echo "Using GPUs: ${GPUS[*]}"
echo "Noise sigmas: ${ALL_NOISE_SIGMAS[*]}"
echo "Fixed mask probability: ${MASK_PROB}"
echo "Full dataset mode: ${FULL_DATASET}"
echo "Save directory: ${SAVE_DIR}"
echo ""

# Function to run evaluation on a specific GPU
run_on_gpu() {
    local gpu=$1
    local noise_sigma=$2
    local save_subdir="${SAVE_DIR}/gpu${gpu}_sigma${noise_sigma}"
    
    mkdir -p "${save_subdir}"
    
    echo "[GPU ${gpu}] Starting noise_sigma=${noise_sigma}"
    
    # Build command
    CMD="python3 evaluate_mask_prob.py \
        --model_config ${MODEL_CONFIG} \
        --diffusion_config ${DIFFUSION_CONFIG} \
        --task_config ${TASK_CONFIG} \
        --gpu ${gpu} \
        --num_test_samples ${NUM_TEST_SAMPLES} \
        --mask_probs ${MASK_PROB} \
        --noise_sigmas ${noise_sigma} \
        --seed ${SEED} \
        --save_dir ${save_subdir}"
    
    # Add full_dataset flag if enabled
    if [ "$FULL_DATASET" = true ]; then
        CMD="${CMD} --full_dataset"
    fi
    
    # Run and log
    eval ${CMD} 2>&1 | tee "${save_subdir}.log"
    
    echo "[GPU ${gpu}] Finished noise_sigma=${noise_sigma}"
}

# Create save directory
mkdir -p ${SAVE_DIR}

# Launch jobs in parallel across GPUs
job_idx=0
pids=()

for noise_sigma in "${ALL_NOISE_SIGMAS[@]}"; do
    gpu_idx=$((job_idx % ${#GPUS[@]}))
    gpu=${GPUS[$gpu_idx]}
    
    run_on_gpu $gpu $noise_sigma &
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