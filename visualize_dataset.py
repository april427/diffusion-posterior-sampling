#!/usr/bin/env python3
"""
Standalone script to visualize samples from the AoA/Amplitude building dataset.
Now with support for displaying metadata (BS positions, building locations).

Usage: python3 visualize_dataset.py [--config path/to/config.yaml] [--num_samples N]
"""

import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import os

from data.dataloader import get_dataset
from data.aoa_amp_building_dataset import AoAAmpBuildingDataset


def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_dataset_sample(sample_tensor, sample_idx=0, save_path=None, denormalize=True, show_plot=True, metadata=None):
    """
    Plot all 6 channels of a dataset sample with optional metadata overlay.
    
    Args:
        sample_tensor: Tensor of shape (6, H, W) or batch (N, 6, H, W)
        sample_idx: Index of sample to plot if batch provided
        save_path: Path to save the figure (optional)
        denormalize: Whether to denormalize the data from [-1, 1] range
        show_plot: Whether to display the plot interactively
        metadata: Optional dict with 'bs_pos', 'buildings', 'map_size', 'grid_spacing'
    """
    # Handle batch dimension
    if sample_tensor.dim() == 4:
        sample = sample_tensor[sample_idx].cpu().numpy()
    else:
        sample = sample_tensor.cpu().numpy()
    
    # Denormalize if needed
    if denormalize:
        # Denormalize AoA maps (first 3 channels) from [-1, 1] to [-180, 180]
        aoa_maps = sample[:3] * 180.0
        
        # Denormalize amplitude maps (last 3 channels) from [-1, 1] to dB
        # Original normalization: 2 * (amp - (-90)) / ((-40) - (-90)) - 1
        # Inverse: amp = (norm + 1) * ((-40) - (-90)) / 2 + (-90)
        amp_maps = (sample[3:] + 1) * 25.0 - 90.0
    else:
        aoa_maps = sample[:3]
        amp_maps = sample[3:]
    
    # Create figure with 2 rows (AoA and Amplitude) and 3 columns (3 paths)
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # Custom colormap for AoA (circular: red -> yellow -> green -> cyan -> blue -> magenta -> red)
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
        ax.set_title(f'AoA Path {i+1} (Strongest → Weakest)', fontsize=11, fontweight='bold')
        ax.set_xlabel('X grid index')
        ax.set_ylabel('Y grid index')
        
        # Overlay buildings and BS position if metadata provided
        if buildings:
            for building in buildings:
                x, y = building['x'], building['y']
                w, h = building['width'], building['height']
                # Convert to grid coordinates
                x_grid = x / grid_spacing
                y_grid = y / grid_spacing
                w_grid = w / grid_spacing
                h_grid = h / grid_spacing
                rect = Rectangle((x_grid, y_grid), w_grid, h_grid,
                                linewidth=2, edgecolor='white', facecolor='gray', alpha=0.3)
                ax.add_patch(rect)
        
        if bs_pos is not None:
            # Convert BS position to grid coordinates
            bs_x_grid = bs_pos[0] / grid_spacing
            bs_y_grid = bs_pos[1] / grid_spacing
            ax.plot(bs_x_grid, bs_y_grid, 'w*', markersize=15, markeredgecolor='black', 
                   markeredgewidth=1.5, label='BS')
            if i == 0:  # Only show legend on first plot
                ax.legend(loc='upper right', fontsize=9)
        
        plt.colorbar(im, ax=ax, label='Angle (degrees)')
        
        # Add statistics
        stats_text = f'Range: [{aoa_maps[i].min():.1f}°, {aoa_maps[i].max():.1f}°]'
        if bs_pos is not None:
            stats_text += f'\nBS: ({bs_pos[0]:.1f}, {bs_pos[1]:.1f})'
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    # Plot Amplitude maps (bottom row)
    for i in range(3):
        ax = axes[1, i]
        im = ax.imshow(amp_maps[i], cmap='hot', vmin=-90, vmax=-40, origin='lower',
                      extent=[0, amp_maps[i].shape[1], 0, amp_maps[i].shape[0]])
        ax.set_title(f'Amplitude Path {i+1} (Strongest → Weakest)', fontsize=11, fontweight='bold')
        ax.set_xlabel('X grid index')
        ax.set_ylabel('Y grid index')
        
        # Overlay buildings and BS position if metadata provided
        if buildings:
            for building in buildings:
                x, y = building['x'], building['y']
                w, h = building['width'], building['height']
                # Convert to grid coordinates
                x_grid = x / grid_spacing
                y_grid = y / grid_spacing
                w_grid = w / grid_spacing
                h_grid = h / grid_spacing
                rect = Rectangle((x_grid, y_grid), w_grid, h_grid,
                                linewidth=2, edgecolor='white', facecolor='gray', alpha=0.3)
                ax.add_patch(rect)
        
        if bs_pos is not None:
            # Convert BS position to grid coordinates
            bs_x_grid = bs_pos[0] / grid_spacing
            bs_y_grid = bs_pos[1] / grid_spacing
            ax.plot(bs_x_grid, bs_y_grid, 'w*', markersize=15, markeredgecolor='black', 
                   markeredgewidth=1.5, label='BS')
        
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # Add statistics
        stats_text = f'Range: [{amp_maps[i].min():.1f}, {amp_maps[i].max():.1f}] dB'
        if metadata and buildings:
            stats_text += f'\n{len(buildings)} building(s)'
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    # Overall title with metadata info
    title = f'Dataset Sample #{sample_idx} Visualization (6 Channels)'
    if metadata:
        title += f'\nBS Position: ({bs_pos[0]:.1f}, {bs_pos[1]:.1f}) | '
        title += f'{len(buildings)} Building(s) | '
        title += f'Map: {map_size[0]}×{map_size[1]}m | Grid: {grid_spacing}m'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Sample plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print detailed statistics
    print("\n" + "="*70)
    print(f"Dataset Sample Statistics (Sample #{sample_idx})")
    print("="*70)
    print(f"Overall tensor shape: {sample.shape}")
    
    if metadata:
        print(f"\nMetadata:")
        print(f"  BS Position: ({bs_pos[0]:.2f}, {bs_pos[1]:.2f})")
        print(f"  Map Size: {map_size}")
        print(f"  Grid Spacing: {grid_spacing}m")
        print(f"  Number of Buildings: {len(buildings)}")
        if buildings:
            print(f"  Building Configurations:")
            for idx, b in enumerate(buildings):
                print(f"    {idx+1}. Position: ({b['x']}, {b['y']}), "
                      f"Size: {b['width']}×{b['height']}m")
    
    print(f"\nAoA Maps (Channels 0-2):")
    for i in range(3):
        print(f"  Path {i+1}: shape={aoa_maps[i].shape}, "
              f"range=[{aoa_maps[i].min():.2f}°, {aoa_maps[i].max():.2f}°], "
              f"mean={aoa_maps[i].mean():.2f}°, std={aoa_maps[i].std():.2f}°")
    
    print(f"\nAmplitude Maps (Channels 3-5):")
    for i in range(3):
        print(f"  Path {i+1}: shape={amp_maps[i].shape}, "
              f"range=[{amp_maps[i].min():.2f}, {amp_maps[i].max():.2f}] dB, "
              f"mean={amp_maps[i].mean():.2f} dB, std={amp_maps[i].std():.2f} dB")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize samples from AoA/Amplitude building dataset")
    parser.add_argument('--config', type=str, default='configs/aoa_amp_building_config.yaml',
                       help='Path to model configuration file containing dataset settings')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='./figures/dataset_samples',
                       help='Directory to save visualization plots')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display plots interactively (only save)')
    parser.add_argument('--show_metadata', action='store_true', default=True,
                       help='Show BS position and building locations on plots')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_yaml(args.config)
    
    # Get dataset configuration
    dataset_config = config.get('dataset', {})
    
    # Create dataset
    print(f"\n{'='*70}")
    print(f"Loading dataset...")
    print(f"{'='*70}")
    train_dataset = get_dataset(**dataset_config)
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ Dataset is empty!")
        return
    
    # Print dataset info
    sample = train_dataset[0]
    print(f"\n📊 Dataset Information:")
    print(f"   Sample shape: {sample.shape}")
    print(f"   Sample dtype: {sample.dtype}")
    print(f"   Sample device: {sample.device}")
    print(f"   Value range: [{sample.min():.3f}, {sample.max():.3f}]")
    
    # Check if dataset has metadata support
    has_metadata = hasattr(train_dataset, 'get_sample_with_metadata')
    if has_metadata:
        print(f"   ✅ Metadata available (BS positions, buildings)")
    else:
        print(f"   ⚠️  No metadata support in this dataset version")
    
    # Visualize samples
    num_samples = min(args.num_samples, len(train_dataset))
    print(f"\n{'='*70}")
    print(f"Visualizing {num_samples} samples...")
    print(f"{'='*70}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for i in range(num_samples):
        print(f"\n📈 Visualizing sample {i+1}/{num_samples}...")
        
        # Get sample with metadata if available
        if has_metadata and args.show_metadata:
            tensor, metadata = train_dataset.get_sample_with_metadata(i)
        else:
            tensor = train_dataset[i]
            metadata = None
        
        save_path = os.path.join(args.save_dir, f"sample_{i:03d}.png")
        
        plot_dataset_sample(
            tensor,
            sample_idx=i,
            save_path=None,
            denormalize=True,
            show_plot=not args.no_display,
            metadata=metadata
        )
    
    print(f"\n{'='*70}")
    print(f"✅ Visualization complete!")
    print(f"   Plots saved to: {args.save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
