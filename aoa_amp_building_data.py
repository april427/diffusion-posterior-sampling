"""
Generate training dataset using ray tracing with buildings for diffusion model.
Each datapoint is a map of AoA and amplitude for the three strongest paths when a BS is placed at different grid positions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from aoa_amp_building import RayTracingAoAMap

def generate_building_training_data(map_size=(100, 100), grid_spacing=1, bs_grid_spacing=1, 
                                  building_configs=None, save_dir='data/building_training'):
    """
    Generate training dataset using ray tracing with fixed buildings.
    
    Parameters:
    -----------
    map_size : tuple
        Size of the map in meters (width, height)
    grid_spacing : float
        Grid spacing for UE positions in meters
    bs_grid_spacing : float
        Grid spacing for BS positions in meters
    building_configs : list of dict
        List of building configurations [{'x': x, 'y': y, 'width': w, 'height': h}]
    save_dir : str
        Directory to save the training data
    
    Returns:
    --------
    dataset : list of dict
        Each dict contains 'bs_pos', 'aoa_maps', 'amplitude_maps', 'los_map'
    """
    
    # Default building configuration if none provided
    if building_configs is None:
        building_configs = [
            {'x': 20, 'y': 20, 'width': 30, 'height': 15},
            {'x': 75, 'y': 56, 'width': 25, 'height': 19}
        ]
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate BS positions on a grid
    bs_x_positions = np.arange(5, map_size[0]-5, bs_grid_spacing)  # Avoid edges
    bs_y_positions = np.arange(5, map_size[1]-5, bs_grid_spacing)
    
    dataset = []
    total_samples = len(bs_x_positions) * len(bs_y_positions)
    
    print(f"Generating {total_samples} training samples...")
    print(f"Map size: {map_size}, Grid spacing: {grid_spacing}")
    print(f"BS grid spacing: {bs_grid_spacing}")
    print(f"Buildings: {len(building_configs)} buildings")
    
    for i, bs_x in enumerate(bs_x_positions):
        for j, bs_y in enumerate(bs_y_positions):
            sample_idx = i * len(bs_y_positions) + j
            
            if sample_idx % 100 == 0:
                print(f"Processing sample {sample_idx}/{total_samples}")
            
            # Create ray tracing model
            rt = RayTracingAoAMap(map_size=map_size, grid_spacing=grid_spacing)
            
            # Set base station position
            rt.set_base_station(bs_x, bs_y)
            
            # Add buildings
            for building in building_configs:
                rt.add_building(building['x'], building['y'], 
                              building['width'], building['height'])
            
            # Generate AoA and amplitude maps for 3 strongest paths
            aoa_maps, los_map = rt.generate_aoa_map(num_paths=3)
            amplitude_maps = rt.generate_amplitude_map(num_paths=3)
            
            # Store sample
            sample = {
                'bs_pos': np.array([bs_x, bs_y]),
                'aoa_maps': aoa_maps,
                'amplitude_maps': amplitude_maps,
                'los_map': los_map,
                'buildings': building_configs,
                'map_size': map_size,
                'grid_spacing': grid_spacing
            }
            
            dataset.append(sample)
    
    # Save dataset
    dataset_file = os.path.join(save_dir, f'training_data_grid{bs_grid_spacing}.pkl')
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {dataset_file}")
    print(f"Total samples: {len(dataset)}")
    
    return dataset

def load_training_data(save_dir='data/building_training', grid_spacing=1):
    """Load training data from file."""
    dataset_file = os.path.join(save_dir, f'training_data_grid{grid_spacing}.pkl')
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Training data file not found: {dataset_file}")
    
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Loaded {len(dataset)} training samples from {dataset_file}")
    return dataset

def visualize_sample(dataset, sample_idx=0, save_plots=False, save_dir='figures'):
    """
    Visualize a sample from the dataset.
    
    Parameters:
    -----------
    dataset : list
        Training dataset
    sample_idx : int
        Index of sample to visualize
    save_plots : bool
        Whether to save plots to files
    save_dir : str
        Directory to save plots
    """
    if sample_idx >= len(dataset):
        raise ValueError(f"Sample index {sample_idx} out of range (max: {len(dataset)-1})")
    
    sample = dataset[sample_idx]
    bs_pos = sample['bs_pos']
    aoa_maps = sample['aoa_maps']
    amplitude_maps = sample['amplitude_maps']
    los_map = sample['los_map']
    buildings = sample['buildings']
    map_size = sample['map_size']
    grid_spacing = sample['grid_spacing']
    
    # Create coordinate grids for plotting
    x_grid = np.arange(0, map_size[0], grid_spacing)
    y_grid = np.arange(0, map_size[1], grid_spacing)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    path_names = ['Strongest Path', 'Second Strongest', 'Third Strongest']
    
    # Plot AoA maps
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    for idx, (aoa_map, name) in enumerate(zip(aoa_maps, path_names)):
        ax = axes[idx]
        im = ax.contourf(X, Y, aoa_map, levels=20, cmap='twilight')
        
        # Plot buildings
        for building in buildings:
            from matplotlib.patches import Rectangle
            rect = Rectangle((building['x'], building['y']), 
                           building['width'], building['height'],
                           linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
        
        # Plot BS
        ax.plot(bs_pos[0], bs_pos[1], 'r*', markersize=20, 
               label='Base Station', markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{name} - AoA')
        ax.set_aspect('equal')
        ax.legend()
        plt.colorbar(im, ax=ax, label='AoA (degrees)')
    
    # Plot LoS map
    ax = axes[3]
    im = ax.contourf(X, Y, los_map.astype(float), levels=[0, 0.5, 1], 
                    cmap='RdYlGn', alpha=0.6)
    
    for building in buildings:
        from matplotlib.patches import Rectangle
        rect = Rectangle((building['x'], building['y']), 
                       building['width'], building['height'],
                       linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
        ax.add_patch(rect)
    
    ax.plot(bs_pos[0], bs_pos[1], 'r*', markersize=20, 
           label='Base Station', markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('LoS Condition')
    ax.set_aspect('equal')
    ax.legend()
    
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_label('LoS')
    cbar.ax.set_yticklabels(['NLoS', 'LoS'])
    
    plt.suptitle(f'Sample {sample_idx}: BS at ({bs_pos[0]:.1f}, {bs_pos[1]:.1f})')
    plt.tight_layout()
    
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'aoa_sample_{sample_idx}.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Plot amplitude maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (amp_map, name) in enumerate(zip(amplitude_maps, path_names)):
        ax = axes[idx]
        im = ax.contourf(X, Y, amp_map, levels=20, cmap='hot_r', vmin=-90, vmax=-40)
        
        # Plot buildings
        for building in buildings:
            from matplotlib.patches import Rectangle
            rect = Rectangle((building['x'], building['y']), 
                           building['width'], building['height'],
                           linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
        
        # Plot BS
        ax.plot(bs_pos[0], bs_pos[1], 'r*', markersize=20, 
               label='Base Station', markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{name} - Amplitude')
        ax.set_aspect('equal')
        ax.legend()
        plt.colorbar(im, ax=ax, label='Amplitude (dB)')
    
    plt.suptitle(f'Sample {sample_idx}: BS at ({bs_pos[0]:.1f}, {bs_pos[1]:.1f})')
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(os.path.join(save_dir, f'amplitude_sample_{sample_idx}.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def get_dataset_statistics(dataset):
    """Get statistics about the dataset."""
    print(f"Dataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        map_shape = sample['aoa_maps'][0].shape
        print(f"  Map shape: {map_shape}")
        print(f"  Map size: {sample['map_size']}")
        print(f"  Grid spacing: {sample['grid_spacing']}")
        print(f"  Number of paths: {len(sample['aoa_maps'])}")
        print(f"  Number of buildings: {len(sample['buildings'])}")
        
        # AoA statistics
        all_aoa = np.concatenate([np.concatenate([aoa_map.flatten() for aoa_map in sample['aoa_maps']]) 
                                 for sample in dataset])
        print(f"  AoA range: [{np.min(all_aoa):.2f}, {np.max(all_aoa):.2f}] degrees")
        
        # Amplitude statistics
        all_amp = np.concatenate([np.concatenate([amp_map.flatten() for amp_map in sample['amplitude_maps']]) 
                                 for sample in dataset])
        print(f"  Amplitude range: [{np.min(all_amp):.2f}, {np.max(all_amp):.2f}] dB")

if __name__ == "__main__":
    # Generate training data
    print("Generating training data with buildings...")
    
    # Configuration
    map_size = (100, 100)
    grid_spacing = 1  # UE grid spacing
    bs_grid_spacing = 2  # BS grid spacing (every 2 meters)
    
    # Building configurations
    building_configs = [
        {'x': 20, 'y': 20, 'width': 30, 'height': 15},
        {'x': 75, 'y': 56, 'width': 25, 'height': 19}
    ]
    
    # Generate dataset
    dataset = generate_building_training_data(
        map_size=map_size,
        grid_spacing=grid_spacing,
        bs_grid_spacing=bs_grid_spacing,
        building_configs=building_configs
    )
    
    # Get statistics
    get_dataset_statistics(dataset)
    
    # Visualize a few samples
    print("\nVisualizing samples...")
    for i in [0, len(dataset)//4, len(dataset)//2, len(dataset)-1]:
        visualize_sample(dataset, sample_idx=i, save_plots=True)