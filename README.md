# Diffusion-Based Reconstruction of AoA / Amplitude Fields

This repository adapts the Diffusion Posterior Sampling (DPS) framework to a wireless-channel setting. For each user/base-station geometry we simulate a 2-channel tensor:
- Channel 1: Angle-of-Arrival (AoA) map
- Channel 2: Receive-amplitude map 

A denoising diffusion probabilistic model (DDPM) learns the joint distribution of these fields. During inference we solve inverse problems such as inpainting by sampling from the learned posterior conditioned on partial measurements.

---

## 1. Requirements & Environment

We recommend Python 3.8 and PyTorch 1.11 (CUDA 11.3). Create and activate a virtual environment:

```bash
# using conda
conda create -n DPS python=3.8
conda activate DPS

# or using venv
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

For CPU-only machines, install the CPU wheels instead of the CUDA packages.

---

## 2. Repository Structure

```
configs/
  aoa_amp_building_config.yaml      # UNet & training hyperparameters
  aoa_amp_building_inpainting.yaml  # inpainting task config
  diffusion_config.yaml             # beta schedule, sampler type, #steps
data/
  aoa_amp_building_dataset.py       # dataset class (AoAAmpBuildingDataset)
  building_training_randomized/     # ray-traced AoA/Amp data with buildings
guided_diffusion/                   # diffusion implementation (DDPM/DDIM)
scripts/
  train_aoa_amp_building.sh         # one-click training script
  run_sampling_building.sh          # conditional inpainting demo
sample_condition_building.py        # posterior sampling + utilities
train_aoa_amp_building.py           # training loop
```

---

## 3. Dataset

`AoAAmpBuildingDataset` (in `data/aoa_amp_building_dataset.py`) loads ray-traced AoA/amplitude maps from `data/building_training_randomized/`. Each sample contains 6 channels (3 AoA + 3 amplitude maps for the strongest paths) across randomized building configurations.

Dataset parameters are controlled in `configs/aoa_amp_building_config.yaml` under the `dataset` key:
- `map_size`: spatial resolution (default 128×128)
- `num_building_sets`: number of randomized building layouts (default 60)
- `building_distribution`: how many configs for 1, 2, 3 buildings (default `[20, 20, 20]`)
- `use_existing_data`: reuse pre-generated data if available

Key normalization:
- AoA: divided by π to map `[-π, π] → [-1, 1]`
- Amplitude: percentile scaling to `[-1, 1]`

### Data generation & ray-tracing files

There are several related files for data generation. Here is what each one does and how they relate:

| File | Role | Description |
|------|------|-------------|
| `aoa_amp_building.py` | **CPU ray tracer** | Core `RayTracingAoAMap` class — pure NumPy implementation of the 2-D ray tracing engine (LOS, reflections, building occlusion). |
| `aoa_amp_building_gpu.py` | **GPU ray tracer** | `RayTracingAoAMapGPU` — PyTorch reimplementation of the same ray tracer with CUDA / MPS acceleration. |
| `aoa_amp_building_data.py` | **CPU data generator** | Calls `RayTracingAoAMap` to sweep BS positions over a grid and produce per-sample pickle files. Original single-threaded version. |
| `aoa_amp_building_data_optimized.py` | **CPU data generator (parallel)** | Same logic as above but with `ProcessPoolExecutor` parallelism for faster generation. |
| `aoa_amp_building_data_gpu.py` | **GPU data generator** | Calls `RayTracingAoAMapGPU` to generate samples on GPU, with batch processing and optional HDF5 output. |
| `data/aoa_amp_building_dataset.py` | **PyTorch Dataset (used in training & sampling)** | `AoAAmpBuildingDataset` — the only file imported by the training and sampling scripts. Loads pre-generated data from `data/building_training_randomized/`. When data doesn't exist yet, it dispatches to one of the generators above with a fallback chain: GPU generator → optimized CPU → basic CPU. |

**In practice**, only `data/aoa_amp_building_dataset.py` is called directly. The data generation files (`aoa_amp_building_data*.py`) and ray tracer files (`aoa_amp_building*.py`) are invoked behind the scenes when `use_existing_data` is `False` or no cached data is found.

### Test dataset
`plot_test_dataset.py` Plot a dataset with 6 channels by set builings and one BS location

`visualize_dataset.py` Load the generated dataset and plot 3 datasets with 1, 2, and 3 buildings 

---

## 4. Training

Launch training with the shell script:

```bash
bash scripts/train_aoa_amp.sh
```

This runs:

```bash
python train_dps.py \
  --model_config configs/aoa_amp_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --train_config configs/train_dps_config.yaml \
  --gpu 0
```

Important settings:
- `train_dps_config.yaml`: set `train_dataset.num_samples` and `val_dataset.num_samples` (default: 10 000 / 1 024). Adjust here if you regenerate caches with different sizes.
- `aoa_amp_config.yaml`: U-Net architecture (base channels, residual blocks, attention levels). Input/output convolutions are patched automatically to handle the 2-channel data with learned variance.
- `diffusion_config.yaml`: `steps` (=1000) and `timestep_respacing`. Reduce `timestep_respacing` (e.g. `250` or `ddim100`) for faster sampling at the cost of some fidelity.

Checkpoints and TensorBoard logs are written under `./checkpoints/aoa_amp_dps` and `./logs/aoa_amp_dps`.

---

## 5. Conditional Sampling / Inpainting

Generate AoA/amplitude reconstructions conditioned on masked observations via:

```bash
bash scripts/run_sampling_building.sh
```

The script executes:

```bash
python3 sample_condition_building.py \
  --model_config configs/aoa_amp_building_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/aoa_amp_building_inpainting.yaml \
  --gpu 0 \
  --save_dir ./results/aoa_amp_building
```

Outputs:
- `input/`, `label/`, `recon/` directories (one subdir per operator, e.g. `inpainting/`).
- Channel visualisations (`*_channel1.pdf`, `*_channel2.pdf`, etc.) for all 6 channels.
- Raw tensors (`*.npy`) in training scale `[-1, 1]`.
- AoA values restored to radians (`*_aoa_rad.npy`).

You may edit `aoa_amp_building_inpainting.yaml` to change mask distribution (`mask_opt`) or conditioning strength (`conditioning.params.scale`).

---

## 6. Quick Sampling Without Conditioning

`quick_sample_debug.py` loads a checkpoint and generates unconditional samples:

```bash
python quick_sample_debug.py \
  --checkpoint checkpoints/aoa_amp_building/checkpoint_epoch_10.pt \
  --output checkpoints/aoa_amp_building/quick_sample.npy \
  --num_samples 4 \
  --gpu 0
```

This is useful for sanity-checking the prior.

---

## 7. Notes & Tips

- **Channel ranges**: Always rescale AoA back to radians via `θ = π * tensor[..., 0, :, :]` when evaluating results.
- **Training data volume**: Diffusion models benefit from large datasets; the simulator allows generating tens of thousands of samples. Increase `train_dataset.num_samples` as needed and regenerate caches.
- **Sampling steps**: Default DDPM uses 1000 steps. Adjust `diffusion_config.yaml:timestep_respacing` (e.g. `250`) for faster but potentially less accurate sampling. Switching to `sampler: ddim` with `timestep_respacing: ddim50` is also supported.
- **Ignoring results in git**: `.gitignore` includes `results/`, `checkpoints/`, `data/aoa_amp_cache/`. If you previously committed files there, run `git rm -r --cached <path>` once to remove them from the index.
- **NumPy / torch compatibility**: Use the curated environment above. Mixing PyTorch wheels compiled against NumPy 1.x with NumPy 2.x can raise `_ARRAY_API not found` errors.

---

## 8. Citation

If you build on DPS, please cite:

```
@inproceedings{
  chung2023diffusion,
  title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
  author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

---

## 9. Contact

For issues or contributions, please open a GitHub issue or submit a pull request.
