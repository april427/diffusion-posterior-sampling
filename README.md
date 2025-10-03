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
  aoa_amp_config.yaml       # UNet & training hyperparameters
  diffusion_config.yaml     # beta schedule, sampler type, #steps
  train_dps_config.yaml     # dataset, batch size, optimizer, logging
data/
  aoa_amp_cache/            # cached AoA/Amp tensors (auto generated)
guided_diffusion/           # diffusion implementation (DDPM/DDIM)
scripts/
  train_aoa_amp.sh          # one-click training script
  run_sampling.sh           # conditional inpainting demo
sample_condition.py         # posterior sampling + utilities
train_dps.py                # training loop (AoA adaptation)
```

---

## 3. Dataset Generation

`AoAAmpDataset` procedurally generates AoA/amplitude grids and caches them in `data/aoa_amp_cache`. The cache filename is determined by `num_samples`, `num_bs`, and `grid_resolution`, e.g.
```
aoa_amp_cache_<num_samples>_<num_bs>_<grid_resolution>.pkl
```
If the cache for a configuration already exists, it is reused; otherwise it is regenerated automatically.

Key normalization:
- AoA: divided by π to map `[-π, π] → [-1, 1]`
- Amplitude: percentile scaling to `[-1, 1]`

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
bash scripts/run_sampling.sh
```

The script executes:

```bash
python sample_condition.py \
  --model_config configs/aoa_amp_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/aoa_inpainting_config.yaml \
  --gpu 0 \
  --save_dir ./results/aoa_inpainting
```

Outputs:
- `input/`, `label/`, `recon/` directories (one subdir per operator, e.g. `inpainting/`).
- Channel visualisations (`*_channel1.png`, `*_channel2.png`) using HSV colormap.
- Raw tensors (`*.npy`) in training scale `[-1, 1]`.
- AoA values restored to radians (`*_aoa_rad.npy`).

You may edit `aoa_inpainting_config.yaml` to change mask distribution (`mask_opt`) or conditioning strength (`conditioning.params.scale`).

---

## 6. Quick Sampling Without Conditioning

`quick_sample_debug.py` loads a checkpoint and generates unconditional samples:

```bash
python quick_sample_debug.py \
  --checkpoint checkpoints/aoa_amp_dps/checkpoint_epoch_200.pt \
  --output checkpoints/aoa_amp_dps/quick_sample.npy \
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
