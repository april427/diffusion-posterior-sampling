#!/usr/bin/env python3
"""
Automated evaluation script to analyze NMSE vs mask probability and/or noise sigma.
Uses fixed test samples for fair comparison across different parameters.

Supports three evaluation modes:
1. Vary mask probability only (--eval_mode mask)
2. Vary noise sigma only (--eval_mode noise)
3. Vary both in a grid (--eval_mode grid)
"""

import os
import argparse
import yaml
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from itertools import product

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from torch.utils.data import DataLoader, Subset, Dataset
from util.img_utils import mask_generator
from util.logger import get_logger


class TensorOnlyDataset(Dataset):
    """
    Lightweight dataset that only loads pre-computed tensor data.
    Avoids loading the full raw data which can cause memory issues.
    """
    def __init__(self, root: str, seed: int = 42, return_index: bool = False):
        self.root = root
        self.seed = seed
        self.return_index = return_index
        
        tensor_cache_file = os.path.join(root, f'tensor_data_{seed}.h5')
        
        if not os.path.exists(tensor_cache_file):
            raise FileNotFoundError(
                f"Tensor cache file not found: {tensor_cache_file}\n"
                f"Please run training first to generate the tensor cache."
            )
        
        print(f"📖 Loading tensor-only data from: {tensor_cache_file}")
        self.tensor_data = self._load_tensor_cache(tensor_cache_file)
        print(f"✅ Loaded {len(self.tensor_data)} samples (tensor-only mode)")
    
    def _load_tensor_cache(self, cache_file):
        """Load pre-converted tensor data from HDF5"""
        import h5py
        
        with h5py.File(cache_file, 'r') as f:
            tensor_data = torch.from_numpy(f['tensor_data'][:]).float()
        
        return tensor_data
    
    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        sample = self.tensor_data[idx]
        if self.return_index:
            return sample, idx
        return sample


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def compute_nmse(reference: torch.Tensor, estimate: torch.Tensor):
    """Compute NMSE total and per-channel."""
    ref = reference.detach()
    est = estimate.detach()

    mse = torch.mean((est - ref) ** 2)
    denom = torch.mean(ref ** 2)
    total_nmse = (mse / (denom + 1e-8)).item()

    per_channel = []
    if ref.dim() == 4:
        for channel in range(ref.shape[1]):
            ref_c = ref[:, channel]
            est_c = est[:, channel]
            mse_c = torch.mean((est_c - ref_c) ** 2)
            denom_c = torch.mean(ref_c ** 2)
            per_channel.append((mse_c / (denom_c + 1e-8)).item())

    return total_nmse, per_channel


def get_fixed_test_indices(total_samples, num_test_samples, seed=42):
    """Get fixed test indices for reproducible evaluation."""
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)[:num_test_samples]
    return sorted(indices.tolist())


def evaluate_single_config(
    model, diffusion, operator, cond_cfg, mask_opt,
    test_loader, mask_prob, noise_sigma, device, logger, seed=42
):
    """Evaluate model with a specific mask probability and noise sigma."""
    
    # Create mask generator with fixed probability, using mask_opt from config
    mask_gen_kwargs = mask_opt.copy()
    mask_gen_kwargs['mask_prob_range'] = (mask_prob, mask_prob)  # Fixed probability
    mask_gen = mask_generator(**mask_gen_kwargs)
    
    # Create noiser with specified sigma
    noiser = get_noise(name='gaussian', sigma=noise_sigma)
    
    # Create conditioning method with the new noiser
    cond_method = get_conditioning_method(
        cond_cfg['method'], operator, noiser, **cond_cfg['params']
    )
    
    # Set seed for reproducible masks
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    nmse_totals = []
    channel_nmse_records = defaultdict(list)
    
    desc = f"mask={mask_prob:.2f}, σ={noise_sigma:.3f}"
    for batch_idx, batch in enumerate(tqdm(test_loader, desc=desc)):
        # Handle dataset that returns (tensor, index) or just tensor
        if isinstance(batch, (list, tuple)):
            ref_img = batch[0]
        else:
            ref_img = batch
        
        ref_img = ref_img.to(device)
        
        # Generate mask
        mask = mask_gen(ref_img)
        mask = mask[:, 0, :, :].unsqueeze(dim=0)
        
        # Setup conditioning
        measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
        sample_fn = partial(
            diffusion.p_sample_loop,
            model=model,
            measurement_cond_fn=measurement_cond_fn,
            record=False,
        )
        
        # Forward measurement
        y = operator.forward(ref_img, mask=mask)
        y_n = noiser(y)
        
        # Sampling - requires_grad for DPS conditioning
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=None)
        
        # Compute NMSE
        total_nmse, per_channel_nmse = compute_nmse(ref_img, sample)
        nmse_totals.append(total_nmse)
        for c_idx, value in enumerate(per_channel_nmse):
            channel_nmse_records[c_idx].append(value)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    # Calculate statistics
    results = {
        'mask_prob': mask_prob,
        'noise_sigma': noise_sigma,
        'total_nmse_mean': np.mean(nmse_totals),
        'total_nmse_std': np.std(nmse_totals),
        'total_nmse_all': nmse_totals,
        'channel_nmse_mean': {},
        'channel_nmse_std': {},
        'channel_nmse_all': {},
    }
    
    for c_idx, values in channel_nmse_records.items():
        results['channel_nmse_mean'][c_idx] = np.mean(values)
        results['channel_nmse_std'][c_idx] = np.std(values)
        results['channel_nmse_all'][c_idx] = values
    
    return results


def plot_results(all_results, save_dir, channel_names=None):
    """Plot NMSE vs mask probability for all channels."""
    
    if channel_names is None:
        channel_names = ['AoA_1', 'AoA_2', 'AoA_3', 'Amp_1', 'Amp_2', 'Amp_3']
    
    mask_probs = [r['mask_prob'] for r in all_results]
    
    # Plot 1: Total NMSE vs Mask Probability
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    total_means = [r['total_nmse_mean'] for r in all_results]
    total_stds = [r['total_nmse_std'] for r in all_results]
    
    ax1.errorbar(mask_probs, total_means, yerr=total_stds, 
                 marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('Mask Probability (% missing)', fontsize=12)
    ax1.set_ylabel('NMSE', fontsize=12)
    ax1.set_title('Total NMSE vs Mask Probability', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(mask_probs)
    ax1.set_xticklabels([f'{p*100:.0f}%' for p in mask_probs])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nmse_vs_mask_prob_total.png'), dpi=150)
    plt.close()
    
    # Plot 2: Per-channel NMSE vs Mask Probability
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    
    for c_idx in range(6):
        if c_idx in all_results[0]['channel_nmse_mean']:
            means = [r['channel_nmse_mean'][c_idx] for r in all_results]
            stds = [r['channel_nmse_std'][c_idx] for r in all_results]
            
            label = channel_names[c_idx] if c_idx < len(channel_names) else f'Channel {c_idx+1}'
            ax2.errorbar(mask_probs, means, yerr=stds, 
                        marker='o', capsize=3, linewidth=2, markersize=6,
                        label=label, color=colors[c_idx])
    
    ax2.set_xlabel('Mask Probability (% missing)', fontsize=12)
    ax2.set_ylabel('NMSE', fontsize=12)
    ax2.set_title('Per-Channel NMSE vs Mask Probability', fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(mask_probs)
    ax2.set_xticklabels([f'{p*100:.0f}%' for p in mask_probs])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nmse_vs_mask_prob_per_channel.png'), dpi=150)
    plt.close()
    
    # Plot 3: AoA vs Amplitude comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    aoa_means = []
    aoa_stds = []
    amp_means = []
    amp_stds = []
    
    for r in all_results:
        # Average over AoA channels (0, 1, 2)
        aoa_vals = [r['channel_nmse_mean'][i] for i in range(3) if i in r['channel_nmse_mean']]
        amp_vals = [r['channel_nmse_mean'][i] for i in range(3, 6) if i in r['channel_nmse_mean']]
        
        aoa_means.append(np.mean(aoa_vals) if aoa_vals else 0)
        amp_means.append(np.mean(amp_vals) if amp_vals else 0)
        
        aoa_stds.append(np.std(aoa_vals) if aoa_vals else 0)
        amp_stds.append(np.std(amp_vals) if amp_vals else 0)
    
    ax3.errorbar(mask_probs, aoa_means, yerr=aoa_stds, 
                 marker='s', capsize=5, linewidth=2, markersize=8, 
                 label='AoA (avg)', color='blue')
    ax3.errorbar(mask_probs, amp_means, yerr=amp_stds, 
                 marker='^', capsize=5, linewidth=2, markersize=8, 
                 label='Amplitude (avg)', color='red')
    
    ax3.set_xlabel('Mask Probability (% missing)', fontsize=12)
    ax3.set_ylabel('NMSE', fontsize=12)
    ax3.set_title('AoA vs Amplitude NMSE Comparison', fontsize=14)
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(mask_probs)
    ax3.set_xticklabels([f'{p*100:.0f}%' for p in mask_probs])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nmse_vs_mask_prob_aoa_vs_amp.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {save_dir}")


def plot_grid_results(all_results, save_dir, mask_probs, noise_sigmas, channel_names=None):
    """Plot NMSE results for grid evaluation (mask_prob x noise_sigma)."""
    
    if channel_names is None:
        channel_names = ['AoA_1', 'AoA_2', 'AoA_3', 'Amp_1', 'Amp_2', 'Amp_3']
    
    # Create heatmap of total NMSE
    nmse_matrix = np.zeros((len(noise_sigmas), len(mask_probs)))
    
    for r in all_results:
        mp_idx = mask_probs.index(r['mask_prob'])
        ns_idx = noise_sigmas.index(r['noise_sigma'])
        nmse_matrix[ns_idx, mp_idx] = r['total_nmse_mean']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(nmse_matrix, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(mask_probs)))
    ax.set_xticklabels([f'{p*100:.0f}%' for p in mask_probs])
    ax.set_yticks(range(len(noise_sigmas)))
    ax.set_yticklabels([f'{s:.3f}' for s in noise_sigmas])
    
    ax.set_xlabel('Mask Probability (% missing)', fontsize=12)
    ax.set_ylabel('Noise Sigma', fontsize=12)
    ax.set_title('Total NMSE Heatmap', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('NMSE', fontsize=12)
    
    # Add text annotations
    for i in range(len(noise_sigmas)):
        for j in range(len(mask_probs)):
            text = ax.text(j, i, f'{nmse_matrix[i, j]:.2e}',
                          ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nmse_heatmap.png'), dpi=150)
    plt.close()
    
    # Plot NMSE vs mask_prob for each noise_sigma
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.plasma(np.linspace(0, 1, len(noise_sigmas)))
    
    for ns_idx, ns in enumerate(noise_sigmas):
        mask_results = [r for r in all_results if r['noise_sigma'] == ns]
        mask_results.sort(key=lambda x: x['mask_prob'])
        
        mps = [r['mask_prob'] for r in mask_results]
        means = [r['total_nmse_mean'] for r in mask_results]
        stds = [r['total_nmse_std'] for r in mask_results]
        
        ax.errorbar(mps, means, yerr=stds, marker='o', capsize=3, 
                   linewidth=2, markersize=6, label=f'σ={ns:.3f}', color=colors[ns_idx])
    
    ax.set_xlabel('Mask Probability (% missing)', fontsize=12)
    ax.set_ylabel('NMSE', fontsize=12)
    ax.set_title('NMSE vs Mask Probability (by Noise Level)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(mask_probs)
    ax.set_xticklabels([f'{p*100:.0f}%' for p in mask_probs])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nmse_vs_mask_by_noise.png'), dpi=150)
    plt.close()
    
    # Plot NMSE vs noise_sigma for each mask_prob
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(mask_probs)))
    
    for mp_idx, mp in enumerate(mask_probs):
        noise_results = [r for r in all_results if r['mask_prob'] == mp]
        noise_results.sort(key=lambda x: x['noise_sigma'])
        
        nss = [r['noise_sigma'] for r in noise_results]
        means = [r['total_nmse_mean'] for r in noise_results]
        stds = [r['total_nmse_std'] for r in noise_results]
        
        ax.errorbar(nss, means, yerr=stds, marker='s', capsize=3, 
                   linewidth=2, markersize=6, label=f'mask={mp*100:.0f}%', color=colors[mp_idx])
    
    ax.set_xlabel('Noise Sigma', fontsize=12)
    ax.set_ylabel('NMSE', fontsize=12)
    ax.set_title('NMSE vs Noise Sigma (by Mask Probability)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nmse_vs_noise_by_mask.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate NMSE vs Mask Probability and/or Noise Sigma')
    parser.add_argument('--model_config', required=True, help='Path to model config')
    parser.add_argument('--diffusion_config', required=True, help='Path to diffusion config')
    parser.add_argument('--task_config', required=True, help='Path to task config')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save_dir', type=str, default='./results/mask_prob_eval', help='Output directory')
    parser.add_argument('--num_test_samples', type=int, default=10, help='Number of test samples')
    parser.add_argument('--mask_probs', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                        help='Comma-separated mask probabilities to evaluate')
    parser.add_argument('--noise_sigmas', type=str, default=None,
                        help='Comma-separated noise sigma values to evaluate (default: use task_config value)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--full_dataset', action='store_true',
                        help='Load full dataset with metadata (default: tensor-only for lighter memory)')
    # Keep --tensor_only for backward compatibility (now default behavior)
    parser.add_argument('--tensor_only', action='store_true',
                        help='[DEPRECATED] Tensor-only is now default. Use --full_dataset to load metadata.')
    args = parser.parse_args()
    
    # Setup
    logger = get_logger()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # Parse mask probabilities
    mask_probs = [float(p) for p in args.mask_probs.split(',')]
    logger.info(f"Evaluating mask probabilities: {mask_probs}")
    
    # Load configs
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    # Parse noise sigmas (from args or task_config)
    if args.noise_sigmas:
        noise_sigmas = [float(s) for s in args.noise_sigmas.split(',')]
    else:
        # Use single value from task_config
        default_sigma = task_config['measurement']['noise'].get('sigma', 0.05)
        noise_sigmas = [default_sigma]
    logger.info(f"Evaluating noise sigmas: {noise_sigmas}")
    
    # Determine evaluation mode
    grid_mode = len(noise_sigmas) > 1 and len(mask_probs) > 1
    
    # Save configs for reference
    with open(os.path.join(save_dir, 'eval_config.json'), 'w') as f:
        json.dump({
            'mask_probs': mask_probs,
            'noise_sigmas': noise_sigmas,
            'num_test_samples': args.num_test_samples,
            'seed': args.seed,
            'model_config': args.model_config,
            'task_config': args.task_config,
            'grid_mode': grid_mode,
            'full_dataset': args.full_dataset,
        }, f, indent=2)
    
    # Setup model
    extra_keys = {'batch_size', 'learning_rate', 'num_epochs', 'save_interval',
                  'epoch_save_interval', 'log_interval', 'dataset', 'dataloader', 'model_path'}
    model_params = {k: v for k, v in model_config.items() if k not in extra_keys}
    
    data_channels = int(model_config.get('data_channels', 6))
    model_params['data_channels'] = data_channels
    
    model = create_model(**model_params)
    model = model.to(device)
    
    # Load checkpoint
    model_path = model_config.get('model_path')
    if model_path:
        logger.info(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    model.eval()
    
    # Setup operator (noiser will be created per-config)
    measure_cfg = task_config['measurement']
    operator = get_operator(device=device, **measure_cfg['operator'])
    
    # Get mask options from config
    mask_opt = measure_cfg.get('mask_opt', {})
    
    # Setup conditioning config
    cond_cfg = task_config['conditioning']
    
    # Setup diffusion
    diffusion = create_sampler(**diffusion_config)
    
    # Load dataset
    data_cfg = task_config['data']
    root = data_cfg.get('root')
    
    # Default: tensor-only mode (lighter memory, avoids CUDA OOM)
    # Use --full_dataset to load with metadata
    use_tensor_only = not args.full_dataset
    
    if use_tensor_only:
        # Use lightweight tensor-only dataset (DEFAULT)
        logger.info("Using tensor-only mode (lighter memory footprint)")
        logger.info("  Use --full_dataset flag if you need metadata")
        full_dataset = TensorOnlyDataset(root=root, seed=args.seed, return_index=True)
    else:
        # Use full dataset with metadata
        logger.info("Using full dataset with metadata")
        from data.dataloader import get_dataset
        from data.aoa_amp_building_dataset import AoAAmpBuildingDataset  # noqa: F401
        
        data_kwargs = data_cfg.copy()
        dataset_name = data_kwargs.pop('name')
        root = data_kwargs.pop('root')
        data_kwargs.pop('num_samples', None)
        data_kwargs['return_index'] = True
        
        full_dataset = get_dataset(dataset_name, root, **data_kwargs)
    
    total_samples = len(full_dataset)
    logger.info(f"Full dataset size: {total_samples}")
    
    # Get fixed test indices
    test_indices = get_fixed_test_indices(total_samples, args.num_test_samples, args.seed)
    logger.info(f"Using {len(test_indices)} fixed test samples: {test_indices[:5]}...")
    
    # Create test subset and loader
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Evaluate all combinations
    all_results = []
    total_configs = len(mask_probs) * len(noise_sigmas)
    config_num = 0
    
    for noise_sigma in noise_sigmas:
        for mask_prob in mask_probs:
            config_num += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Config {config_num}/{total_configs}: mask={mask_prob:.2f} ({mask_prob*100:.0f}% missing), σ={noise_sigma:.4f}")
            logger.info(f"{'='*60}")
            
            results = evaluate_single_config(
                model=model,
                diffusion=diffusion,
                operator=operator,
                cond_cfg=cond_cfg,
                mask_opt=mask_opt,
                test_loader=test_loader,
                mask_prob=mask_prob,
                noise_sigma=noise_sigma,
                device=device,
                logger=logger,
                seed=args.seed,
            )
            
            all_results.append(results)
            
            # Log results
            logger.info(f"Result: Total NMSE = {results['total_nmse_mean']:.4e} ± {results['total_nmse_std']:.4e}")
            for c_idx in sorted(results['channel_nmse_mean'].keys()):
                logger.info(f"  Channel {c_idx+1}: {results['channel_nmse_mean'][c_idx]:.4e} ± {results['channel_nmse_std'][c_idx]:.4e}")
    
    # Save results
    results_path = os.path.join(save_dir, 'results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []
    for r in all_results:
        sr = {
            'mask_prob': r['mask_prob'],
            'noise_sigma': r['noise_sigma'],
            'total_nmse_mean': float(r['total_nmse_mean']),
            'total_nmse_std': float(r['total_nmse_std']),
            'total_nmse_all': [float(x) for x in r['total_nmse_all']],
            'channel_nmse_mean': {str(k): float(v) for k, v in r['channel_nmse_mean'].items()},
            'channel_nmse_std': {str(k): float(v) for k, v in r['channel_nmse_std'].items()},
        }
        serializable_results.append(sr)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Plot results
    channel_names = ['AoA_1', 'AoA_2', 'AoA_3', 'Amp_1', 'Amp_2', 'Amp_3']
    
    if grid_mode:
        plot_grid_results(all_results, save_dir, mask_probs, noise_sigmas, channel_names)
    else:
        plot_results(all_results, save_dir, channel_names)
    
    # Print summary table
    logger.info("\n" + "="*100)
    logger.info("SUMMARY TABLE")
    logger.info("="*100)
    
    if grid_mode:
        header = f"{'Noise σ':>10} | {'Mask Prob':>10} | {'Total NMSE':>12} | " + " | ".join([f'{n:>10}' for n in channel_names])
    else:
        header = f"{'Mask Prob':>10} | {'Noise σ':>10} | {'Total NMSE':>12} | " + " | ".join([f'{n:>10}' for n in channel_names])
    logger.info(header)
    logger.info("-" * len(header))
    
    for r in all_results:
        if grid_mode:
            row = f"{r['noise_sigma']:>10.4f} | {r['mask_prob']*100:>9.0f}% | {r['total_nmse_mean']:>12.4e} | "
        else:
            row = f"{r['mask_prob']*100:>9.0f}% | {r['noise_sigma']:>10.4f} | {r['total_nmse_mean']:>12.4e} | "
        row += " | ".join([f"{r['channel_nmse_mean'].get(i, 0):>10.4e}" for i in range(6)])
        logger.info(row)
    
    logger.info("="*100)
    logger.info(f"Evaluation complete! Results saved to {save_dir}")


if __name__ == '__main__':
    main()