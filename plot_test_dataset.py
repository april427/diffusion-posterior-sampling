#!/usr/bin/env python3
"""
Plot test dataset generated from aoa_amp_building_data_gpu.py using 
the plotting functions from aoa_amp_building_gpu.py
"""

import sys
import os
import numpy as np

# Add the repository root to path
sys.path.append('/Users/au759614/Documents/AU/DPS/diffusion-posterior-sampling')

from aoa_amp_building_data_gpu import generate_building_training_data_gpu_batch
from aoa_amp_building_gpu import RayTracingAoAMapGPU

def plot_test_dataset_samples():
    """Generate test dataset and plot selected samples"""
    
    print("Generating test dataset using GPU acceleration...")
    
    # Generate test dataset (same as in the original file)
    test_dataset = generate_building_training_data_gpu_batch(
        map_size=(50, 50),  # Slightly larger for better visualization
        grid_spacing=2,     # Coarser grid for faster generation
        bs_grid_spacing=25, # Fewer base station positions
        building_configs=[
            {'x': 15, 'y': 15, 'width': 12, 'height': 8},
            {'x': 30, 'y': 25, 'width': 10, 'height': 6}
        ],
        save_dir=None,
        device='auto',
        batch_size=2,
        num_workers=1
    )
    
    print(f"\n✅ Generated {len(test_dataset)} samples")
    
    if not test_dataset:
        print("❌ No test dataset generated!")
        return
    
    # Display dataset information
    sample = test_dataset[0]
    print(f"\nDataset Information:")
    print(f"  AoA maps shape: {[aoa_map.shape for aoa_map in sample['aoa_maps']]}")
    print(f"  Amplitude maps shape: {[amp_map.shape for amp_map in sample['amplitude_maps']]}")
    print(f"  LoS map shape: {sample['los_map'].shape}")
    print(f"  Base station position: {sample['bs_pos']}")
    print(f"  Map size: {sample['map_size']}")
    print(f"  Grid spacing: {sample['grid_spacing']}")
    
    # Create a RayTracingAoAMapGPU instance for plotting (matching the sample parameters)
    map_size = sample['map_size']
    grid_spacing = sample['grid_spacing']
    bs_pos = sample['bs_pos']
    
    # Create ray tracer for plotting
    rt = RayTracingAoAMapGPU(
        map_size=map_size, 
        grid_spacing=grid_spacing, 
        device='auto', 
        verbose=True
    )
    
    # Set base station position
    rt.set_base_station(bs_pos[0], bs_pos[1])
    
    # Add the same buildings used in dataset generation
    buildings = sample.get('building_configs', [
        {'x': 15, 'y': 15, 'width': 12, 'height': 8},
        {'x': 30, 'y': 25, 'width': 10, 'height': 6}
    ])
    
    for building in buildings:
        rt.add_building(building['x'], building['y'], 
                       building['width'], building['height'])
    
    print(f"\n📊 Plotting samples from the test dataset...")
    
    # Plot multiple samples if available
    num_samples_to_plot = min(3, len(test_dataset))
    
    for i in range(num_samples_to_plot):
        sample = test_dataset[i]
        bs_pos = sample['bs_pos']
        
        print(f"\n📈 Sample {i+1}: BS at ({bs_pos[0]:.1f}, {bs_pos[1]:.1f})")
        
        # Update base station position for this sample
        rt.set_base_station(bs_pos[0], bs_pos[1])
        
        # Extract data from sample
        aoa_maps = sample['aoa_maps']
        amplitude_maps = sample['amplitude_maps']
        los_map = sample['los_map']
        
        # Verify data types and shapes
        print(f"  AoA maps: {len(aoa_maps)} maps")
        for j, aoa_map in enumerate(aoa_maps):
            print(f"    Map {j}: shape={aoa_map.shape}, range=[{aoa_map.min():.1f}°, {aoa_map.max():.1f}°]")
        
        print(f"  Amplitude maps: {len(amplitude_maps)} maps")
        for j, amp_map in enumerate(amplitude_maps):
            print(f"    Map {j}: shape={amp_map.shape}, range=[{amp_map.min():.1f}, {amp_map.max():.1f}] dB")
        
        print(f"  LoS map: shape={los_map.shape}, LoS%={np.sum(los_map)/los_map.size*100:.1f}%")
        
        # Plot AoA maps
        path_names = ['Strongest Path', '2nd Strongest', '3rd Strongest']
        print(f"  🎨 Plotting AoA maps...")
        rt.plot_aoa_map(aoa_maps, los_map, path_names)
        
        # Plot amplitude maps  
        print(f"  🎨 Plotting amplitude maps...")
        rt.plot_amplitude_map(amplitude_maps, path_names)
        
        print(f"  ✅ Sample {i+1} plotted successfully!")
    
    print(f"\n🎉 All {num_samples_to_plot} samples plotted successfully!")
    
    # Display some statistics
    print(f"\n📊 Dataset Statistics:")
    all_bs_positions = [sample['bs_pos'] for sample in test_dataset]
    print(f"  Base station positions: {len(all_bs_positions)}")
    for i, pos in enumerate(all_bs_positions):
        print(f"    Sample {i+1}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # Calculate average LoS percentage across all samples
    avg_los_pct = np.mean([np.sum(sample['los_map'])/sample['los_map'].size*100 for sample in test_dataset])
    print(f"  Average LoS percentage: {avg_los_pct:.1f}%")


if __name__ == "__main__":
    print("🚀 Starting test dataset plotting...")
    plot_test_dataset_samples()