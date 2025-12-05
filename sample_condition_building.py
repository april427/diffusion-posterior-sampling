"""Conditional sampling script tailored to the AoA/Amplitude building dataset."""

from functools import partial
from collections import defaultdict
import argparse
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import yaml

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from torch.utils.data import DataLoader, Subset
from data.dataloader import get_dataset, get_dataloader
from data.aoa_amp_building_dataset import AoAAmpBuildingDataset  # noqa: F401
from util.img_utils import clear_color, mask_generator, normalize_np
from util.logger import get_logger


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_tensor_channels(
    tensor,
    out_dir: str,
    base_name: str,
    cmap: str = 'viridis',
    normalize: bool = True,
    channel_multipliers: Optional[Sequence[float]] = None,
    channel_value_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    channel_cmaps: Optional[Sequence[str]] = None,
):
    data = tensor.detach().cpu().clone()
    if data.ndim == 4:
        data = data.squeeze(0)
    if data.ndim == 2:
        data = data.unsqueeze(0)

    os.makedirs(out_dir, exist_ok=True)
    for idx in range(data.shape[0]):
        channel = data[idx].numpy()
        if channel_multipliers is not None and idx < len(channel_multipliers):
            channel = channel * channel_multipliers[idx]

        channel_name = os.path.join(out_dir, f"{base_name}_channel{idx + 1}.png")
        current_cmap = cmap
        if channel_cmaps is not None and idx < len(channel_cmaps) and channel_cmaps[idx]:
            current_cmap = channel_cmaps[idx]

        if normalize:
            img = normalize_np(channel)
            plt.imsave(channel_name, img, cmap=current_cmap)
        else:
            kwargs = {}
            if channel_value_ranges is not None and idx < len(channel_value_ranges):
                vmin, vmax = channel_value_ranges[idx]
                kwargs['vmin'] = vmin
                kwargs['vmax'] = vmax
            plt.imsave(channel_name, channel, cmap=current_cmap, **kwargs)


def save_tensor_npy(tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, tensor.detach().cpu().numpy())


def save_aoa_radians(tensor, out_dir: str, base_name: str, aoa_channels: int):
    arr = tensor.detach().cpu().numpy().copy()
    if arr.ndim == 4:
        aoa = arr[:, :aoa_channels] * np.pi
    elif arr.ndim == 3:
        aoa = arr[:aoa_channels] * np.pi
        aoa = aoa[None, ...]
    else:
        return

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{base_name}_aoa_rad.npy"), aoa)


def plot_6channel_comparison(input_tensor, label_tensor, recon_tensor, save_path, 
                             metadata=None, denormalize=True):
    """
    Plot input, label, and reconstruction side-by-side with all 6 channels.
    Shows BS position and buildings if metadata is provided.
    
    Args:
        input_tensor: Input tensor (6, H, W)
        label_tensor: Ground truth tensor (6, H, W)
        recon_tensor: Reconstructed tensor (6, H, W)
        save_path: Path to save the figure
        metadata: Optional dict with 'bs_pos', 'buildings', etc.
        denormalize: Whether to denormalize from [-1, 1] range
    """
    # Convert tensors to numpy
    input_np = input_tensor.detach().cpu().numpy()
    label_np = label_tensor.detach().cpu().numpy()
    recon_np = recon_tensor.detach().cpu().numpy()
    
    # Handle batch dimension
    if input_np.ndim == 4:
        input_np = input_np[0]
    if label_np.ndim == 4:
        label_np = label_np[0]
    if recon_np.ndim == 4:
        recon_np = recon_np[0]
    
    # Denormalize if needed
    if denormalize:
        # AoA channels (0-2): [-1, 1] -> [-180, 180] degrees
        input_aoa = input_np[:3] * 180.0
        label_aoa = label_np[:3] * 180.0
        recon_aoa = recon_np[:3] * 180.0
        
        # Amplitude channels (3-5): [-1, 1] -> dB range
        input_amp = (input_np[3:] + 1) * 25.0 - 90.0
        label_amp = (label_np[3:] + 1) * 25.0 - 90.0
        recon_amp = (recon_np[3:] + 1) * 25.0 - 90.0
    else:
        input_aoa, label_aoa, recon_aoa = input_np[:3], label_np[:3], recon_np[:3]
        input_amp, label_amp, recon_amp = input_np[3:], label_np[3:], recon_np[3:]
    
    # Create figure: 6 rows (3 AoA + 3 Amp) x 3 columns (Input, Label, Recon)
    # Compact layout for paper: smaller figure, tighter spacing
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(6, 3, hspace=0.08, wspace=0.12,
                          left=0.06, right=0.92, top=0.94, bottom=0.04)
    
    # Custom colormap for AoA
    aoa_cmap = LinearSegmentedColormap.from_list(
        'aoa', 
        ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'red']
    )
    
    # Extract metadata if available
    bs_pos = metadata.get('bs_pos') if metadata else None
    buildings = metadata.get('buildings', []) if metadata else []
    grid_spacing = metadata.get('grid_spacing', 1.0) if metadata else 1.0
    
    column_titles = ['Input', 'Ground Truth', 'Reconstruction']
    row_labels = ['AoA 1', 'AoA 2', 'AoA 3', 'Amp 1', 'Amp 2', 'Amp 3']
    
    # Plot AoA channels (rows 0-2)
    for path_idx in range(3):
        data_list = [input_aoa[path_idx], label_aoa[path_idx], recon_aoa[path_idx]]
        
        for col_idx, data in enumerate(data_list):
            ax = fig.add_subplot(gs[path_idx, col_idx])
            im = ax.imshow(data, cmap=aoa_cmap, vmin=-180, vmax=180, origin='lower',
                          extent=[0, data.shape[1], 0, data.shape[0]])
            
            # Add title only on first row
            if path_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=9, fontweight='bold')
            
            # Add row label on first column
            if col_idx == 0:
                ax.set_ylabel(row_labels[path_idx], fontsize=8, fontweight='bold')
            
            # Remove tick labels for inner plots to save space
            if path_idx < 2:
                ax.set_xticklabels([])
            if col_idx > 0:
                ax.set_yticklabels([])
            ax.tick_params(axis='both', labelsize=6)
            
            # Overlay buildings and BS
            if buildings and col_idx == 1:  # Only on ground truth column
                for building in buildings:
                    x, y = building['x'], building['y']
                    w, h = building['width'], building['height']
                    x_grid = x / grid_spacing
                    y_grid = y / grid_spacing
                    w_grid = w / grid_spacing
                    h_grid = h / grid_spacing
                    rect = Rectangle((x_grid, y_grid), w_grid, h_grid,
                                    linewidth=2, edgecolor='white', 
                                    facecolor='gray', alpha=0.3)
                    ax.add_patch(rect)
                
                if bs_pos is not None:
                    bs_x_grid = bs_pos[0] / grid_spacing
                    bs_y_grid = bs_pos[1] / grid_spacing
                    ax.plot(bs_x_grid, bs_y_grid, 'w*', markersize=15, 
                           markeredgecolor='black', markeredgewidth=1.5)
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.ax.tick_params(labelsize=5)
            if col_idx == 2:  # Only label rightmost colorbar
                cbar.set_label('Angle (°)', fontsize=6)
            
            ax.grid(False)
    
    # Plot Amplitude channels (rows 3-5)
    for path_idx in range(3):
        data_list = [input_amp[path_idx], label_amp[path_idx], recon_amp[path_idx]]
        row_idx = path_idx + 3
        
        for col_idx, data in enumerate(data_list):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(data, cmap='hot', vmin=-90, vmax=-40, origin='lower',
                          extent=[0, data.shape[1], 0, data.shape[0]])
            
            # Add row label on first column
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=8, fontweight='bold')
            
            # Remove tick labels for inner plots to save space
            if path_idx < 2:
                ax.set_xticklabels([])
            if col_idx > 0:
                ax.set_yticklabels([])
            ax.tick_params(axis='both', labelsize=6)
            
            # Overlay buildings and BS
            if buildings and col_idx == 1:  # Only on ground truth column
                for building in buildings:
                    x, y = building['x'], building['y']
                    w, h = building['width'], building['height']
                    x_grid = x / grid_spacing
                    y_grid = y / grid_spacing
                    w_grid = w / grid_spacing
                    h_grid = h / grid_spacing
                    rect = Rectangle((x_grid, y_grid), w_grid, h_grid,
                                    linewidth=2, edgecolor='white', 
                                    facecolor='gray', alpha=0.3)
                    ax.add_patch(rect)
                
                if bs_pos is not None:
                    bs_x_grid = bs_pos[0] / grid_spacing
                    bs_y_grid = bs_pos[1] / grid_spacing
                    ax.plot(bs_x_grid, bs_y_grid, 'w*', markersize=15, 
                           markeredgecolor='black', markeredgewidth=1.5)
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.ax.tick_params(labelsize=5)
            if col_idx == 2:  # Only label rightmost colorbar
                cbar.set_label('Power (dB)', fontsize=6)
            
            ax.grid(False)
    
    # Overall title with metadata (compact)
    title = 'Input vs Ground Truth vs Reconstruction'
    if metadata and bs_pos is not None:
        title += f'  |  BS: ({bs_pos[0]:.0f}, {bs_pos[1]:.0f}), {len(buildings)} bldg'
    fig.suptitle(title, fontsize=10, fontweight='bold')
    
    # Save figure
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    print(f"✅ Comparison plot saved to {save_path}")


def plot_6channel_single(tensor, save_path, title_prefix='Sample', 
                         metadata=None, denormalize=True):
    """
    Plot a single 6-channel tensor (for input, label, or reconstruction alone).
    
    Args:
        tensor: Tensor of shape (6, H, W) or (1, 6, H, W)
        save_path: Path to save the figure
        title_prefix: Prefix for the title (e.g., 'Input', 'Label', 'Reconstruction')
        metadata: Optional dict with 'bs_pos', 'buildings', etc.
        denormalize: Whether to denormalize from [-1, 1] range
    """
    # Convert to numpy
    data = tensor.detach().cpu().numpy()
    if data.ndim == 4:
        data = data[0]
    
    # Denormalize if needed
    if denormalize:
        aoa_maps = data[:3] * 180.0
        amp_maps = (data[3:] + 1) * 25.0 - 90.0
    else:
        aoa_maps = data[:3]
        amp_maps = data[3:]
    
    # Create figure with 2 rows (AoA and Amplitude) and 3 columns (3 paths)
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # Custom colormap for AoA
    aoa_cmap = LinearSegmentedColormap.from_list(
        'aoa', 
        ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'red']
    )
    
    # Extract metadata if available
    bs_pos = metadata.get('bs_pos') if metadata else None
    buildings = metadata.get('buildings', []) if metadata else []
    map_size = metadata.get('map_size') if metadata else None
    grid_spacing = metadata.get('grid_spacing', 1.0) if metadata else 1.0
    
    # Plot AoA maps (top row)
    for i in range(3):
        ax = axes[0, i]
        im = ax.imshow(aoa_maps[i], cmap=aoa_cmap, vmin=-180, vmax=180, origin='lower',
                      extent=[0, aoa_maps[i].shape[1], 0, aoa_maps[i].shape[0]])
        ax.set_title(f'AoA Path {i+1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('X grid index')
        ax.set_ylabel('Y grid index')
        
        # Overlay buildings and BS position
        if buildings:
            for building in buildings:
                x, y = building['x'], building['y']
                w, h = building['width'], building['height']
                x_grid = x / grid_spacing
                y_grid = y / grid_spacing
                w_grid = w / grid_spacing
                h_grid = h / grid_spacing
                rect = Rectangle((x_grid, y_grid), w_grid, h_grid,
                                linewidth=2, edgecolor='white', facecolor='gray', alpha=0.3)
                ax.add_patch(rect)
        
        if bs_pos is not None:
            bs_x_grid = bs_pos[0] / grid_spacing
            bs_y_grid = bs_pos[1] / grid_spacing
            ax.plot(bs_x_grid, bs_y_grid, 'w*', markersize=15, 
                   markeredgecolor='black', markeredgewidth=1.5, label='BS')
            if i == 0:
                ax.legend(loc='upper right', fontsize=9)
        
        plt.colorbar(im, ax=ax, label='Angle (degrees)')
        
        # Add statistics
        stats_text = f'Range: [{aoa_maps[i].min():.1f}°, {aoa_maps[i].max():.1f}°]'
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    # Plot Amplitude maps (bottom row)
    for i in range(3):
        ax = axes[1, i]
        im = ax.imshow(amp_maps[i], cmap='hot', vmin=-90, vmax=-40, origin='lower',
                      extent=[0, amp_maps[i].shape[1], 0, amp_maps[i].shape[0]])
        ax.set_title(f'Amplitude Path {i+1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('X grid index')
        ax.set_ylabel('Y grid index')
        
        # Overlay buildings and BS position
        if buildings:
            for building in buildings:
                x, y = building['x'], building['y']
                w, h = building['width'], building['height']
                x_grid = x / grid_spacing
                y_grid = y / grid_spacing
                w_grid = w / grid_spacing
                h_grid = h / grid_spacing
                rect = Rectangle((x_grid, y_grid), w_grid, h_grid,
                                linewidth=2, edgecolor='white', facecolor='gray', alpha=0.3)
                ax.add_patch(rect)
        
        if bs_pos is not None:
            bs_x_grid = bs_pos[0] / grid_spacing
            bs_y_grid = bs_pos[1] / grid_spacing
            ax.plot(bs_x_grid, bs_y_grid, 'w*', markersize=15, 
                   markeredgecolor='black', markeredgewidth=1.5)
        
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # Add statistics
        stats_text = f'Range: [{amp_maps[i].min():.1f}, {amp_maps[i].max():.1f}] dB'
        if metadata and buildings:
            stats_text += f'\n{len(buildings)} building(s)'
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    # Overall title with metadata info
    title = f'{title_prefix} (6 Channels)'
    if metadata and bs_pos is not None:
        title += f'\nBS Position: ({bs_pos[0]:.1f}, {bs_pos[1]:.1f}) | '
        title += f'{len(buildings)} Building(s)'
        if map_size is not None:
            title += f' | Map: {map_size[0]}×{map_size[1]}m'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    # Save figure
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Single-view plot saved to {save_path}")


def compute_nmse(reference: torch.Tensor, estimate: torch.Tensor):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', required=True)
    parser.add_argument('--diffusion_config', required=True)
    parser.add_argument('--task_config', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()

    logger = get_logger()
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    logger.info(f"Device set to {device_str}.")

    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    extra_keys = {
        'batch_size', 'learning_rate', 'num_epochs', 'save_interval',
        'epoch_save_interval', 'log_interval', 'dataset', 'dataloader',
        'model_path'
    }
    model_params = {k: v for k, v in model_config.items() if k not in extra_keys}

    data_channels = int(model_config.get('data_channels', 6))
    
    # Ensure data_channels is passed to create_model
    model_params['data_channels'] = data_channels

    model = create_model(**model_params)

    model = model.to(device)

    model_path = model_config.get('model_path')
    if model_path:
        logger.info(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

    model.eval()

    measure_cfg = task_config['measurement']
    operator = get_operator(device=device, **measure_cfg['operator'])
    noiser = get_noise(**measure_cfg['noise'])
    logger.info(f"Operation: {measure_cfg['operator']['name']} / Noise: {measure_cfg['noise']['name']}")

    cond_cfg = task_config['conditioning']
    cond_method = get_conditioning_method(cond_cfg['method'], operator, noiser, **cond_cfg['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method: {cond_cfg['method']}")

    diffusion = create_sampler(**diffusion_config)
    sample_fn = partial(
        diffusion.p_sample_loop,
        model=model,
        measurement_cond_fn=measurement_cond_fn,
        record=False,
    )

    out_path = os.path.join(args.save_dir, measure_cfg['operator']['name'])
    for sub in ['input', 'recon', 'label']:
        os.makedirs(os.path.join(out_path, sub), exist_ok=True)

    data_cfg = task_config['data']
    data_kwargs = data_cfg.copy()
    dataset_name = data_kwargs.pop('name')
    root = data_kwargs.pop('root')
    data_kwargs.pop('num_samples', None)

    if dataset_name != 'aoa_amp_building':
        raise ValueError('sample_condition_building.py is intended for aoa_amp_building dataset only.')

    # Enable return_index to get actual dataset indices (important for metadata lookup)
    data_kwargs['return_index'] = True
    full_dataset = get_dataset(dataset_name, root, **data_kwargs)
    total_samples = len(full_dataset)
    logger.info(f"Full dataset size: {total_samples}")
    
    # ==========================================================================
    # USE TEST SPLIT BY BUILDING CONFIGURATION
    # Dataset structure: 60 building configs, 20 per group (1, 2, 3 buildings)
    # Each config has (map_size / bs_grid_spacing)^2 BS positions = samples
    # Training: first 18 configs per group (configs 0-17, 20-37, 40-57)
    # Testing:  last 2 configs per group  (configs 18-19, 38-39, 58-59)
    # ==========================================================================
    
    num_building_configs = 60  # Total building configurations
    configs_per_group = 20     # 20 configs each for 1, 2, 3 buildings
    train_configs_per_group = 18
    test_configs_per_group = 2
    
    # Calculate samples per building configuration
    samples_per_config = total_samples // num_building_configs
    logger.info(f"Samples per building config: {samples_per_config}")
    
    # Build test indices: configs 18-19, 38-39, 58-59
    # Organize by building count for interleaved sampling
    test_indices_by_group = {1: [], 2: [], 3: []}
    for group in range(3):  # 3 groups: 1-building, 2-building, 3-building
        num_buildings = group + 1
        group_start_config = group * configs_per_group  # 0, 20, 40
        for config_offset in range(train_configs_per_group, configs_per_group):  # 18-19
            config_id = group_start_config + config_offset
            sample_start = config_id * samples_per_config
            sample_end = sample_start + samples_per_config
            test_indices_by_group[num_buildings].extend(range(sample_start, sample_end))
    
    # Shuffle indices within each group for random BS locations
    import random
    random.seed(42)  # For reproducibility
    for num_buildings in test_indices_by_group:
        random.shuffle(test_indices_by_group[num_buildings])
    
    # Interleave samples: 1 bldg, 2 bldg, 3 bldg, 1 bldg, 2 bldg, 3 bldg, ...
    test_indices = []
    max_samples_per_group = max(len(indices) for indices in test_indices_by_group.values())
    
    for i in range(max_samples_per_group):
        for num_buildings in [1, 2, 3]:
            if i < len(test_indices_by_group[num_buildings]):
                test_indices.append(test_indices_by_group[num_buildings][i])
    
    logger.info(f"Test configs: 18-19 (1 bldg), 38-39 (2 bldg), 58-59 (3 bldg)")
    logger.info(f"Test samples: {len(test_indices)} (interleaved by building count, random BS locations)")
    logger.info(f"Sampling order: 1 bldg → 2 bldg → 3 bldg → 1 bldg → ...")
    
    # Create test subset - Note: we wrap indices so Subset returns (tensor, original_idx)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create dataloader (no shuffle for test)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if measure_cfg['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(**measure_cfg['mask_opt'])

    nmse_totals = []
    channel_nmse_records = defaultdict(list)

    aoa_channels = 3  # first three channels are AoA
    channel_ranges = [(-np.pi, np.pi)] * aoa_channels + [(-1.0, 1.0)] * (data_channels - aoa_channels)
    channel_scales = [np.pi] * aoa_channels + [1.0] * (data_channels - aoa_channels)
    channel_cmaps = ['hsv'] * aoa_channels + ['hsv'] * (data_channels - aoa_channels)

    for enum_idx, (ref_img, dataset_idx) in enumerate(loader):
        # dataset_idx is the original index in the full dataset (from return_index=True)
        # test_indices[enum_idx] gives us the same original index
        original_idx = dataset_idx.item()
        logger.info(f"Inference for test sample {enum_idx} (original dataset index: {original_idx})")
        fname_base = str(enum_idx).zfill(5)
        ref_img = ref_img.to(device)
        
        # Fetch metadata using the original dataset index
        metadata = None
        # Access the underlying full dataset (Subset wraps the original dataset)
        underlying_dataset = loader.dataset.dataset if hasattr(loader.dataset, 'dataset') else loader.dataset
        if hasattr(underlying_dataset, 'get_metadata'):
            try:
                metadata = underlying_dataset.get_metadata(original_idx)
                logger.info(f"Loaded metadata: BS @ {metadata.get('bs_pos')}, {len(metadata.get('buildings', []))} building(s)")
            except Exception as e:
                logger.warning(f"Could not load metadata for sample {enum_idx}: {e}")

        if measure_cfg['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else:
            y = operator.forward(ref_img)
            y_n = noiser(y)

        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        total_nmse, per_channel_nmse = compute_nmse(ref_img, sample)
        nmse_totals.append(total_nmse)
        for c_idx, value in enumerate(per_channel_nmse):
            channel_nmse_records[c_idx].append(value)

        nmse_msg = ", ".join(
            f"ch{c_idx + 1}: {value:.4e}" for c_idx, value in enumerate(per_channel_nmse)
        ) if per_channel_nmse else ""
        logger.info(f"NMSE (sample {enum_idx}): total {total_nmse:.4e} {nmse_msg}")

        # Save comprehensive comparison plot with metadata in comparison folder
        comparison_path = os.path.join(out_path, 'comparison', f'{fname_base}_comparison.png')
        plot_6channel_comparison(y_n, ref_img, sample, comparison_path, 
                                metadata=metadata, denormalize=True)
        
        # Save combined 6-channel plots in their respective directories
        plot_6channel_single(y_n, os.path.join(out_path, 'input', f'{fname_base}_combined.png'),
                            title_prefix='Input (Degraded)', metadata=metadata, denormalize=True)
        plot_6channel_single(ref_img, os.path.join(out_path, 'label', f'{fname_base}_combined.png'),
                            title_prefix='Ground Truth', metadata=metadata, denormalize=True)
        plot_6channel_single(sample, os.path.join(out_path, 'recon', f'{fname_base}_combined.png'),
                            title_prefix='Reconstruction', metadata=metadata, denormalize=True)

        # Legacy channel-by-channel saves (kept for compatibility)
        save_tensor_channels(
            y_n,
            os.path.join(out_path, 'input'),
            fname_base,
            cmap='viridis',
            normalize=False,
            channel_multipliers=channel_scales,
            channel_value_ranges=channel_ranges,
            channel_cmaps=channel_cmaps,
        )
        save_tensor_channels(
            ref_img,
            os.path.join(out_path, 'label'),
            fname_base,
            cmap='viridis',
            normalize=False,
            channel_multipliers=channel_scales,
            channel_value_ranges=channel_ranges,
            channel_cmaps=channel_cmaps,
        )
        save_tensor_channels(
            sample,
            os.path.join(out_path, 'recon'),
            fname_base,
            cmap='viridis',
            normalize=False,
            channel_multipliers=channel_scales,
            channel_value_ranges=channel_ranges,
            channel_cmaps=channel_cmaps,
        )

        save_tensor_npy(y_n, os.path.join(out_path, 'input', f'{fname_base}.npy'))
        save_tensor_npy(ref_img, os.path.join(out_path, 'label', f'{fname_base}.npy'))
        save_tensor_npy(sample, os.path.join(out_path, 'recon', f'{fname_base}.npy'))

        save_aoa_radians(y_n, os.path.join(out_path, 'input'), fname_base, aoa_channels)
        save_aoa_radians(ref_img, os.path.join(out_path, 'label'), fname_base, aoa_channels)
        save_aoa_radians(sample, os.path.join(out_path, 'recon'), fname_base, aoa_channels)

    if nmse_totals:
        avg_total_nmse = sum(nmse_totals) / len(nmse_totals)
        logger.info(f"Average NMSE over {len(nmse_totals)} samples: {avg_total_nmse:.4e}")
        for c_idx, values in channel_nmse_records.items():
            avg_channel_nmse = sum(values) / len(values)
            logger.info(f"Average NMSE channel {c_idx + 1}: {avg_channel_nmse:.4e}")


if __name__ == '__main__':
    main()
