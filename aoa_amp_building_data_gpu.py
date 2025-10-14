"""
GPU-accelerated data generation for building training dataset.
This module integrates the GPU ray tracing with the dataset generation pipeline.
"""

import torch
import numpy as np
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import List, Dict, Tuple
import time

from aoa_amp_building_gpu import RayTracingAoAMapGPU

# Check CUDA availability
def get_best_device():
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def generate_sample_data_gpu(args):
    """Generate a single training sample using GPU acceleration."""
    bs_pos, map_size, grid_spacing, building_configs, device = args
    
    # Create ray tracer (without verbose output during worker execution)
    rt = RayTracingAoAMapGPU(map_size=map_size, grid_spacing=grid_spacing, device=device, verbose=False)
    
    # Extract BS position coordinates
    bs_x, bs_y = bs_pos
    
    # Set base station position
    rt.set_base_station(bs_x, bs_y)
    
    # Add buildings
    for building in building_configs:
        rt.add_building(building['x'], building['y'], 
                      building['width'], building['height'])
    
    # Generate AoA and amplitude maps for 3 strongest paths using GPU
    aoa_maps, los_map = rt.generate_aoa_map_gpu(num_paths=3)
    amplitude_maps = rt.generate_amplitude_map_gpu(num_paths=3)
    
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
    
    return sample


def generate_building_training_data_gpu_batch(map_size=(128, 128), grid_spacing=1, bs_grid_spacing=5, 
                                            building_configs=None, save_dir='data/building_training',
                                            device='auto', batch_size=8, num_workers=None,
                                            progress_callback=None, verbose=False):
    """
    Generate training dataset using GPU-accelerated ray tracing with batch processing.
    
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
    save_dir : str or None
        Directory to save the training data. If None, data won't be saved to disk.
    device : str
        Device to use ('cuda', 'cpu', or 'auto')
    batch_size : int
        Number of base station positions to process in each batch
    num_workers : int or None
        Number of worker processes. If None, uses 1 for GPU or CPU count for CPU.
    progress_callback : callable or None
        Function to call with progress updates
    verbose : bool
        Whether to show detailed progress information
    
    Returns:
    --------
    dataset : list of dict
        Each dict contains 'bs_pos', 'aoa_maps', 'amplitude_maps', 'los_map'
    """
    
    # Setup device
    if device == 'auto':
        device = get_best_device()
    
    # Default building configuration if none provided
    if building_configs is None:
        building_configs = [
            {'x': 20, 'y': 20, 'width': 30, 'height': 15},
            {'x': 75, 'y': 56, 'width': 25, 'height': 19}
        ]
    
    # Create save directory if it doesn't exist and save_dir is not None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Set number of workers
    if num_workers is None:
        if device == 'cuda':
            num_workers = 1  # GPU works best with fewer processes
        else:
            num_workers = min(mp.cpu_count() // 2, 4)
    
    # Generate BS positions on a grid
    # Handle both tuple and single integer map_size
    if isinstance(map_size, (tuple, list)):
        map_x, map_y = map_size[0], map_size[1]
    else:
        map_x, map_y = map_size, map_size
    
    bs_x_positions = np.arange(5, map_x-5, bs_grid_spacing)  # Avoid edges
    bs_y_positions = np.arange(5, map_y-5, bs_grid_spacing)
    
    # Create all BS position combinations
    bs_positions = [(x, y) for x in bs_x_positions for y in bs_y_positions]
    total_samples = len(bs_positions)
    
    # Only show details if verbose mode is enabled
    if verbose:
        print(f"Generating {total_samples} training samples using GPU acceleration...")
        print(f"Map size: {map_size}, Grid spacing: {grid_spacing}")
        print(f"BS grid spacing: {bs_grid_spacing}")
        print(f"Buildings: {len(building_configs)} buildings")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    dataset = []
    start_time = time.time()
    
    if device == 'cuda' and torch.cuda.is_available():
        # GPU processing - use smaller batches and fewer workers
        if verbose:
            print("Using GPU acceleration with batch processing...")
        
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_positions = bs_positions[i:batch_end]
            
            # Process batch sequentially on GPU (GPU doesn't benefit much from multiprocessing)
            for pos in batch_positions:
                args = (pos, map_size, grid_spacing, building_configs, device)
                sample = generate_sample_data_gpu(args)
                dataset.append(sample)
            
            # Only show progress if verbose or if callback is provided
            if progress_callback:
                progress_callback(batch_end, total_samples)
    
    else:
        # CPU processing - use multiprocessing
        if verbose:
            print("Using CPU with multiprocessing...")
        
        # Prepare arguments for workers
        worker_args = [(pos, map_size, grid_spacing, building_configs, device) for pos in bs_positions]
        
        # Use ThreadPoolExecutor for CPU (since we're using PyTorch which handles threading well)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Process in batches to provide progress updates
            for i in range(0, len(worker_args), batch_size):
                batch_end = min(i + batch_size, len(worker_args))
                batch_args = worker_args[i:batch_end]
                
                # Submit batch
                batch_results = list(executor.map(generate_sample_data_gpu, batch_args))
                dataset.extend(batch_results)
                
                # Only show progress if verbose or if callback is provided
                if progress_callback:
                    progress_callback(batch_end, total_samples)
    
    total_time = time.time() - start_time
    
    # Save dataset only if save_dir is provided
    if save_dir is not None:
        dataset_file = os.path.join(save_dir, f'training_data_gpu_grid{bs_grid_spacing}.pkl')
        print(f"Saving dataset to {dataset_file}...")
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {dataset_file}")
    
    print(f"Total samples: {len(dataset)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average speed: {len(dataset)/total_time:.2f} samples/sec")
    
    # GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleaned up")
    
    return dataset


def benchmark_gpu_vs_cpu(map_size=(50, 50), grid_spacing=5, bs_grid_spacing=15, 
                       building_configs=None, num_samples=9):
    """
    Benchmark GPU vs CPU performance for ray tracing
    """
    if building_configs is None:
        building_configs = [
            {'x': 10, 'y': 10, 'width': 15, 'height': 10},
            {'x': 25, 'y': 30, 'width': 12, 'height': 8}
        ]
    
    print("=" * 60)
    print("GPU vs CPU Ray Tracing Benchmark")
    print("=" * 60)
    
    # Generate limited BS positions for testing
    bs_x_positions = np.arange(5, map_size[0]-5, bs_grid_spacing)[:3]  # Limit to 3
    bs_y_positions = np.arange(5, map_size[1]-5, bs_grid_spacing)[:3]  # Limit to 3
    bs_positions = [(x, y) for x in bs_x_positions for y in bs_y_positions]
    
    results = {}
    
    # Test CPU
    print("\nTesting CPU performance...")
    start_time = time.time()
    cpu_dataset = []
    
    for pos in bs_positions:
        args = (pos, map_size, grid_spacing, building_configs, 'cpu')
        sample = generate_sample_data_gpu(args)
        cpu_dataset.append(sample)
    
    cpu_time = time.time() - start_time
    results['cpu'] = {'time': cpu_time, 'samples': len(cpu_dataset)}
    
    print(f"CPU: {len(cpu_dataset)} samples in {cpu_time:.3f}s ({len(cpu_dataset)/cpu_time:.2f} samples/s)")
    
    # Test GPU if available
    if torch.cuda.is_available():
        print("\nTesting GPU performance...")
        start_time = time.time()
        gpu_dataset = []
        
        for pos in bs_positions:
            args = (pos, map_size, grid_spacing, building_configs, 'cuda')
            sample = generate_sample_data_gpu(args)
            gpu_dataset.append(sample)
        
        gpu_time = time.time() - start_time
        results['gpu'] = {'time': gpu_time, 'samples': len(gpu_dataset)}
        
        print(f"GPU: {len(gpu_dataset)} samples in {gpu_time:.3f}s ({len(gpu_dataset)/gpu_time:.2f} samples/s)")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\nGPU Speedup: {speedup:.2f}x")
        
        # Memory usage
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
    else:
        print("\nGPU not available for testing")
    
    print("=" * 60)
    return results


if __name__ == "__main__":
    # Test the GPU-accelerated data generation
    print("Testing GPU-accelerated building training data generation...")
    
    # Small test case
    test_dataset = generate_building_training_data_gpu_batch(
        map_size=(20, 20),
        grid_spacing=1,
        bs_grid_spacing=15,
        building_configs=[
            {'x': 10, 'y': 10, 'width': 15, 'height': 10},
            {'x': 25, 'y': 30, 'width': 12, 'height': 8}
        ],
        save_dir=None,
        device='auto',
        batch_size=4,
        num_workers=2
    )
    
    print(f"\nGenerated {len(test_dataset)} samples")
    if test_dataset:
        sample = test_dataset[0]
        print(f"Sample shape - AoA: {sample['aoa_maps'][0].shape}")
        print(f"Sample shape - Amplitude: {sample['amplitude_maps'][0].shape}")
        print(f"LOS percentage: {np.sum(sample['los_map'])/sample['los_map'].size*100:.1f}%")
    
    # Run benchmark
    print("\n" + "="*60)
    benchmark_results = benchmark_gpu_vs_cpu()