"""
Dataset for AoA and Amplitude maps generated using ray tracing with buildings.
This dataset uses the strongest 3 paths approach with randomized building configurations.
GPU-optimized version for faster data processing.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing import Optional, Callable, List, Tuple
import os
import pickle
import sys
import random
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import time

from data.dataloader import register_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from aoa_amp_building_data import generate_building_training_data, load_training_data


@register_dataset(name='aoa_amp_building')
class AoAAmpBuildingDataset(VisionDataset):
    """
    GPU-optimized dataset for AoA and Amplitude maps generated using ray tracing with buildings.
    Each sample contains 6 channels: 3 AoA maps + 3 amplitude maps for the strongest paths.
    Buildings are randomized across different configurations.
    """
    
    def __init__(self, 
                 root: str,
                 transforms: Optional[Callable] = None,
                 map_size: Tuple[int, int] = (100, 100),
                 grid_spacing: float = 1.0,
                 bs_grid_spacing: float = 5.0,
                 num_building_sets: int = 120,
                 building_distribution: Tuple[int, int, int] = (40, 40, 40),
                 use_existing_data: bool = True,
                 normalize_data: bool = True,
                 building_size_range: Tuple[Tuple[int, int], Tuple[int, int]] = ((15, 35), (10, 25)),
                 min_building_distance: float = 5.0,
                 seed: int = 42,
                 device: str = 'auto',
                 use_gpu_processing: bool = True,
                 batch_size: int = 32,
                 num_workers: int = None,
                 pin_memory: bool = True,
                 prefetch_factor: int = 2):
        """
        Args:
            root: Root directory for caching data
            transforms: Transform pipeline for data augmentation
            map_size: Size of the map in meters (width, height)
            grid_spacing: Grid spacing for UE positions in meters
            bs_grid_spacing: Grid spacing for BS positions in meters
            num_building_sets: Total number of building configuration sets
            building_distribution: (num_1_building, num_2_buildings, num_3_buildings)
            use_existing_data: Whether to use existing data or generate new
            normalize_data: Whether to normalize data to [-1, 1] range
            building_size_range: ((min_width, max_width), (min_height, max_height))
            min_building_distance: Minimum distance between buildings
            seed: Random seed for reproducible building configurations
            device: Device to use ('cuda', 'cpu', or 'auto')
            use_gpu_processing: Whether to use GPU for data processing
            batch_size: Batch size for GPU processing
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            prefetch_factor: Number of samples loaded in advance by each worker
        """
        super().__init__(root, transforms)
        
        self.map_size = map_size
        self.grid_spacing = grid_spacing
        self.bs_grid_spacing = bs_grid_spacing
        self.use_existing_data = use_existing_data
        self.normalize_data = normalize_data
        self.num_building_sets = num_building_sets
        self.building_distribution = building_distribution
        self.building_size_range = building_size_range
        self.min_building_distance = min_building_distance
        self.seed = seed
        self.use_gpu_processing = use_gpu_processing
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Setup number of workers
        if num_workers is None:
            self.num_workers = min(8, mp.cpu_count())
        else:
            self.num_workers = num_workers
        
        print(f"Using device: {self.device}")
        print(f"GPU processing enabled: {self.use_gpu_processing}")
        print(f"Number of workers: {self.num_workers}")
        
        # Validate building distribution
        if sum(building_distribution) != num_building_sets:
            raise ValueError(f"Building distribution {building_distribution} doesn't sum to {num_building_sets}")
        
        # Generate randomized building configurations
        self.building_config_sets = self._generate_building_configurations()
        
        # Create cache directory
        os.makedirs(root, exist_ok=True)
        
        # Prepare data
        self._prepare_data()
    
    def _generate_building_configurations(self) -> List[List[dict]]:
        """Generate randomized building configurations using parallel processing"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Create tasks for parallel processing
        tasks = []
        for num_buildings, count in enumerate(self.building_distribution, 1):
            for i in range(count):
                # Use different seeds for each configuration to ensure uniqueness
                config_seed = self.seed + len(tasks)
                tasks.append((num_buildings, config_seed))
        
        # Generate configurations in parallel
        print(f"Generating {len(tasks)} building configurations using {self.num_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            configurations = list(executor.map(self._generate_single_configuration_worker, tasks))
        
        print(f"Generated {len(configurations)} building configuration sets:")
        for i in range(len(self.building_distribution)):
            print(f"  - {self.building_distribution[i]} sets with {i + 1} building(s)")

        return configurations
    
    def _generate_single_configuration_worker(self, task):
        """Worker function for parallel configuration generation"""
        num_buildings, config_seed = task
        
        # Set seed for this worker
        random.seed(config_seed)
        np.random.seed(config_seed)
        
        return self._generate_single_configuration(num_buildings)
    
    def _generate_single_configuration(self, num_buildings: int) -> List[dict]:
        """Generate a single building configuration with specified number of buildings"""
        buildings = []
        max_attempts = 1000
        
        for building_idx in range(num_buildings):
            attempts = 0
            while attempts < max_attempts:
                # Random building size
                width = random.randint(*self.building_size_range[0])
                height = random.randint(*self.building_size_range[1])
                
                # Random position (ensure building fits in map)
                margin = 5  # Keep buildings away from map edges
                x = random.uniform(margin, self.map_size[0] - width - margin)
                y = random.uniform(margin, self.map_size[1] - height - margin)
                
                new_building = {'x': x, 'y': y, 'width': width, 'height': height}
                
                # Check if this building overlaps with existing ones
                if self._is_valid_building_placement(new_building, buildings):
                    buildings.append(new_building)
                    break
                    
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not place building {building_idx + 1} after {max_attempts} attempts")
        
        return buildings
    
    def _is_valid_building_placement(self, new_building: dict, existing_buildings: List[dict]) -> bool:
        """Check if a new building placement is valid (doesn't overlap with existing ones)"""
        if not existing_buildings:
            return True
            
        for existing in existing_buildings:
            # Check if buildings overlap (with minimum distance buffer)
            if self._buildings_overlap(new_building, existing, self.min_building_distance):
                return False
        return True
    
    def _buildings_overlap(self, building1: dict, building2: dict, min_distance: float) -> bool:
        """Check if two buildings overlap considering minimum distance"""
        # Expand buildings by min_distance/2 on each side
        b1_left = building1['x'] - min_distance/2
        b1_right = building1['x'] + building1['width'] + min_distance/2
        b1_bottom = building1['y'] - min_distance/2
        b1_top = building1['y'] + building1['height'] + min_distance/2
        
        b2_left = building2['x'] - min_distance/2
        b2_right = building2['x'] + building2['width'] + min_distance/2
        b2_bottom = building2['y'] - min_distance/2
        b2_top = building2['y'] + building2['height'] + min_distance/2
        
        # Check for overlap
        return not (b1_right <= b2_left or b2_right <= b1_left or 
                   b1_top <= b2_bottom or b2_top <= b1_bottom)
    
    def _prepare_data(self):
        """Generate or load cached dataset for all building configurations"""
        cache_file = os.path.join(self.root, f'all_building_configs_data_{self.seed}.pkl')
        tensor_cache_file = os.path.join(self.root, f'tensor_data_{self.seed}.pt')
        
        try:
            if self.use_existing_data and os.path.exists(cache_file):
                print(f"Loading existing building configuration data from {cache_file}...")
                with open(cache_file, 'rb') as f:
                    self.data = pickle.load(f)
                
                # Try to load pre-converted tensor data directly to CUDA
                if os.path.exists(tensor_cache_file):
                    print(f"Loading pre-converted tensor data from {tensor_cache_file}...")
                    # Load directly to the target device
                    self.tensor_data = torch.load(tensor_cache_file, map_location=self.device)
                    # Convert to list of tensors on device
                    if isinstance(self.tensor_data, torch.Tensor):
                        self.tensor_data = [self.tensor_data[i] for i in range(self.tensor_data.shape[0])]
                else:
                    self._convert_to_tensors_gpu()
                    self._save_tensor_cache(tensor_cache_file)
            else:
                raise FileNotFoundError("Generating new data as requested")
                
        except FileNotFoundError:
            print("Generating new building training data for all configurations...")
            self._generate_all_data()
            
            # Save raw data
            print(f"Saving building configuration data to {cache_file}...")
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data, f)
            
            # Convert to tensors and save
            self._convert_to_tensors_gpu()
            self._save_tensor_cache(tensor_cache_file)
        
        print(f"Dataset prepared with {len(self.tensor_data)} samples across {len(self.building_config_sets)} building configurations")
        self._print_dataset_statistics()
    
    def _generate_all_data(self):
        """Generate data for all building configurations using parallel processing with progress bar"""
        self.data = []
        
        # Create tasks for parallel data generation
        tasks = [(i, config) for i, config in enumerate(self.building_config_sets)]
        
        print(f"\n🏗️  Generating training data for {len(tasks)} building configurations...")
        print(f"⚙️  Using {self.num_workers} workers on {self.device}")
        
        # Create a progress bar for configurations
        config_progress = tqdm(
            desc="Building configs", 
            total=len(tasks), 
            unit="config",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Track progress with a callback function
        completed_configs = 0
        config_lock = threading.Lock()
        
        def update_progress(future):
            nonlocal completed_configs
            with config_lock:
                completed_configs += 1
                config_progress.update(1)
            config_progress.set_postfix_str(f"Config {completed_configs}/{len(tasks)}")
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks and add progress callback
            futures = []
            for task in tasks:
                future = executor.submit(self._generate_config_data_worker, task)
                future.add_done_callback(update_progress)
                futures.append(future)
            
            # Collect results as they complete
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
        
        config_progress.close()
        
        # Flatten results with progress bar
        print("\n📊 Consolidating dataset...")
        total_samples = sum(len(config_data) for config_data in results)
        
        sample_progress = tqdm(
            desc="Samples", 
            total=total_samples, 
            unit="sample",
            ncols=100
        )
        
        for config_data in results:
            self.data.extend(config_data)
            sample_progress.update(len(config_data))
        
        sample_progress.close()
        print(f"✅ Generated {len(self.data)} total samples")
    
    def _generate_config_data_worker(self, task):
        """Worker function for parallel data generation"""
        config_idx, building_config = task
        
        # Reduce verbose output - only show for first few configs or periodically
        show_progress = (config_idx < 3) or (config_idx % 10 == 0)
        
        # Try to use GPU-accelerated version if available and requested
        if self.use_gpu_processing and self.device.type == 'cuda':
            try:
                from aoa_amp_building_data_gpu import generate_building_training_data_gpu_batch
                if show_progress:
                    print(f"🔥 Using GPU acceleration for config {config_idx + 1}")
                
                config_data = generate_building_training_data_gpu_batch(
                    map_size=self.map_size,
                    grid_spacing=self.grid_spacing,
                    bs_grid_spacing=self.bs_grid_spacing,
                    building_configs=building_config,
                    save_dir=None,  # Don't save individual configs to disk
                    device=str(self.device),
                    batch_size=min(8, self.batch_size),  # Smaller batches for GPU
                    num_workers=1,  # Single worker per config to avoid nested parallelism
                    progress_callback=None  # Disable inner progress for cleaner output
                )
            except ImportError:
                if show_progress:
                    print(f"⚠️  GPU acceleration not available, falling back to optimized CPU")
                # Fallback to optimized CPU version
                try:
                    from aoa_amp_building_data_optimized import generate_building_training_data_optimized
                    config_data = generate_building_training_data_optimized(
                        map_size=self.map_size,
                        grid_spacing=self.grid_spacing,
                        bs_grid_spacing=self.bs_grid_spacing,
                        building_configs=building_config,
                        save_dir=None,
                        num_workers=1,
                        use_multiprocessing=False
                    )
                except ImportError:
                    # Final fallback to original version
                    from aoa_amp_building_data import generate_building_training_data
                    config_data = generate_building_training_data(
                        map_size=self.map_size,
                        grid_spacing=self.grid_spacing,
                        bs_grid_spacing=self.bs_grid_spacing,
                        building_configs=building_config,
                        save_dir=None
                    )
        else:
            # Use CPU-optimized version
            try:
                from aoa_amp_building_data_optimized import generate_building_training_data_optimized
                if show_progress:
                    print(f"💻 Using optimized CPU for config {config_idx + 1}")
                config_data = generate_building_training_data_optimized(
                    map_size=self.map_size,
                    grid_spacing=self.grid_spacing,
                    bs_grid_spacing=self.bs_grid_spacing,
                    building_configs=building_config,
                    save_dir=None,
                    num_workers=1,
                    use_multiprocessing=False
                )
            except ImportError:
                # Fallback to original version
                if show_progress:
                    print(f" Using original CPU version for config {config_idx + 1}")
                from aoa_amp_building_data import generate_building_training_data
                config_data = generate_building_training_data(
                    map_size=self.map_size,
                    grid_spacing=self.grid_spacing,
                    bs_grid_spacing=self.bs_grid_spacing,
                    building_configs=building_config,
                    save_dir=None
                )
        
        # Add configuration index to each sample
        for sample in config_data:
            sample['config_idx'] = config_idx
            sample['num_buildings'] = len(building_config)
        
        return config_data
    
    def _convert_to_tensors_gpu(self):
        """Convert data to tensors using GPU acceleration"""
        print("\n🔄 Converting data to tensor format...")
        
        if self.use_gpu_processing and self.device.type == 'cuda':
            self._convert_with_gpu_batching()
        else:
            self._convert_with_cpu_batching()
    
    def _convert_with_gpu_batching(self):
        """Convert data using GPU batch processing"""
        self.tensor_data = []
        num_samples = len(self.data)
        
        print(f"🔥 Converting {num_samples} samples using GPU batching...")
        
        # Use tqdm for progress tracking
        with tqdm(total=num_samples, desc="GPU Conversion", unit="sample", ncols=100) as pbar:
            for i in range(0, num_samples, self.batch_size):
                batch_end = min(i + self.batch_size, num_samples)
                batch_samples = self.data[i:batch_end]
                
                # Process batch on GPU
                batch_tensors = []
                for sample in batch_samples:
                    tensor = self._convert_to_tensor_gpu(sample)
                    batch_tensors.append(tensor)
                
                # Stack tensors and keep on GPU
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors)
                    
                    # Split back to individual tensors (still on GPU)
                    for j in range(batch_tensor.shape[0]):
                        self.tensor_data.append(batch_tensor[j])
                
                pbar.update(batch_end - i)
    
    def _convert_with_cpu_batching(self):
        """Convert data using CPU with parallel processing"""
        print(f"💻 Converting {len(self.data)} samples using CPU parallel processing...")
        
        # Use ThreadPoolExecutor for CPU-bound tensor conversion
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Use tqdm to track progress
            self.tensor_data = list(tqdm(
                executor.map(self._convert_to_tensor, self.data),
                total=len(self.data),
                desc="CPU Conversion",
                unit="sample",
                ncols=100
            ))
        
        # Pin memory if requested
        if self.pin_memory and self.device.type == 'cuda':
            print("📌 Pinning memory for faster GPU transfer...")
            with tqdm(total=len(self.tensor_data), desc="Pinning memory", unit="sample", ncols=100) as pbar:
                pinned_data = []
                for tensor in self.tensor_data:
                    pinned_data.append(tensor.pin_memory())
                    pbar.update(1)
                self.tensor_data = pinned_data
    
    def _convert_to_tensor_gpu(self, sample_data):
        """Convert single sample to tensor using GPU operations"""
        aoa_maps = sample_data['aoa_maps']  # List of 3 2D arrays
        amplitude_maps = sample_data['amplitude_maps']  # List of 3 2D arrays
        
        # First, normalize all maps to 2D and get expected shape
        processed_aoa_maps = []
        for i, aoa_map in enumerate(aoa_maps):
            if aoa_map.ndim > 2:
                aoa_map = np.squeeze(aoa_map)
            processed_aoa_maps.append(aoa_map)
        
        processed_amp_maps = []
        for i, amp_map in enumerate(amplitude_maps):
            if amp_map.ndim > 2:
                amp_map = np.squeeze(amp_map)
            processed_amp_maps.append(amp_map)
        
        # Use the first processed map to determine expected shape
        expected_shape = processed_aoa_maps[0].shape
        all_maps_list = processed_aoa_maps + processed_amp_maps
        
        # Check for shape mismatches and fix them
        fixed_maps = []
        for i, map_data in enumerate(all_maps_list):
            if map_data.shape != expected_shape:
                try:
                    from scipy.ndimage import zoom
                    zoom_factors = [expected_shape[j] / map_data.shape[j] for j in range(len(map_data.shape))]
                    if len(zoom_factors) != len(expected_shape):
                        if map_data.ndim > len(expected_shape):
                            map_data = np.squeeze(map_data)
                            zoom_factors = [expected_shape[j] / map_data.shape[j] for j in range(len(expected_shape))]
                    
                    map_data = zoom(map_data, zoom_factors, order=1)
                except ImportError:
                    if len(expected_shape) == 2 and map_data.ndim == 2:
                        old_h, old_w = map_data.shape
                        new_h, new_w = expected_shape
                        y_coords = np.linspace(0, old_h-1, new_h).astype(int)
                        x_coords = np.linspace(0, old_w-1, new_w).astype(int)
                        map_data = map_data[np.ix_(y_coords, x_coords)]
                    else:
                        map_data = np.squeeze(map_data)
                        if map_data.ndim == 2 and map_data.shape != expected_shape:
                            old_h, old_w = map_data.shape
                            new_h, new_w = expected_shape
                            y_coords = np.linspace(0, old_h-1, new_h).astype(int)
                            x_coords = np.linspace(0, old_w-1, new_w).astype(int)
                            map_data = map_data[np.ix_(y_coords, x_coords)]
                except Exception as e:
                    map_data = np.squeeze(map_data)
                        
            fixed_maps.append(map_data)
        
        # Final verification that all shapes match
        final_shapes = [map_data.shape for map_data in fixed_maps]
        if len(set(final_shapes)) > 1:
            from collections import Counter
            shape_counts = Counter(final_shapes)
            most_common_shape = shape_counts.most_common(1)[0][0]
            
            final_fixed_maps = []
            for i, map_data in enumerate(fixed_maps):
                if map_data.shape != most_common_shape:
                    try:
                        from scipy.ndimage import zoom
                        zoom_factors = [most_common_shape[j] / map_data.shape[j] for j in range(len(most_common_shape))]
                        map_data = zoom(map_data, zoom_factors, order=1)
                    except:
                        if len(most_common_shape) == 2 and map_data.ndim == 2:
                            old_h, old_w = map_data.shape
                            new_h, new_w = most_common_shape
                            y_coords = np.linspace(0, old_h-1, new_h).astype(int)
                            x_coords = np.linspace(0, old_w-1, new_w).astype(int)
                            map_data = map_data[np.ix_(y_coords, x_coords)]
                final_fixed_maps.append(map_data)
            fixed_maps = final_fixed_maps
            
            final_shapes_check = [map_data.shape for map_data in fixed_maps]
            if len(set(final_shapes_check)) > 1:
                raise ValueError(f"Could not resolve shape mismatch. Final shapes: {final_shapes_check}")
        
        # Stack all maps: shape = (6, H, W) - 3 AoA + 3 Amplitude
        all_maps = np.stack(fixed_maps, axis=0)
        
        # Convert to tensor but keep on CPU for DataLoader compatibility
        tensor = torch.from_numpy(all_maps).float()
        
        if self.normalize_data:
            # Normalize AoA maps (first 3 channels) from [-180, 180] to [-1, 1]
            tensor[:3] = tensor[:3] / 180.0
            
            # Normalize amplitude maps (last 3 channels) 
            # Amplitude is in dB, typically ranging from -90 to -40 dB
            amp_data = tensor[3:]
            tensor[3:] = 2 * (amp_data - (-90)) / ((-40) - (-90)) - 1
            tensor[3:] = torch.clamp(tensor[3:], -1, 1)
        
        return tensor  # Return on CPU for DataLoader compatibility
    
    def _convert_to_tensor(self, sample_data):
        """Convert AoA/amplitude maps to tensor format (CPU version)"""
        aoa_maps = sample_data['aoa_maps']  # List of 3 2D arrays
        amplitude_maps = sample_data['amplitude_maps']  # List of 3 2D arrays
        
        # Stack all maps: shape = (6, H, W) - 3 AoA + 3 Amplitude
        all_maps = np.stack(aoa_maps + amplitude_maps, axis=0)
        
        if self.normalize_data:
            # Normalize AoA maps (first 3 channels) from [-180, 180] to [-1, 1]
            all_maps[:3] = all_maps[:3] / 180.0
            
            # Normalize amplitude maps (last 3 channels) 
            # Amplitude is in dB, typically ranging from -90 to -40 dB
            amp_data = all_maps[3:]
            # Normalize from typical range [-90, -40] to [-1, 1]
            all_maps[3:] = 2 * (amp_data - (-90)) / ((-40) - (-90)) - 1
            all_maps[3:] = np.clip(all_maps[3:], -1, 1)
        
        return torch.FloatTensor(all_maps)
    
    def _save_tensor_cache(self, cache_file):
        """Save tensor data to cache"""
        print(f"Saving tensor cache to {cache_file}...")
        
        # Move to CPU temporarily for saving, then back to GPU
        if self.tensor_data:
            # Move all tensors to CPU for saving
            cpu_tensors = [tensor.cpu() for tensor in self.tensor_data]
            stacked_tensors = torch.stack(cpu_tensors)
            torch.save(stacked_tensors, cache_file)
            
            # Keep tensors on GPU for usage
            # self.tensor_data is already on GPU, no need to change
    
    def _print_dataset_statistics(self):
        """Print dataset statistics"""
        config_counts = {}
        for sample in self.data:
            num_buildings = sample['num_buildings']
            config_counts[num_buildings] = config_counts.get(num_buildings, 0) + 1
        
        print("Dataset statistics:")
        for num_buildings in sorted(config_counts.keys()):
            print(f"  - {config_counts[num_buildings]} samples with {num_buildings} building(s)")
        
        if self.tensor_data is not None and len(self.tensor_data) > 0:
            sample_device = self.tensor_data[0].device if hasattr(self.tensor_data[0], 'device') else 'CPU'
            print(f"Tensor data device: {sample_device}")
    
    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: Shape (6, H, W) where first 3 channels are AoA maps
                        and last 3 channels are amplitude maps for strongest paths
        """
        sample = self.tensor_data[idx]
        
        # Tensor is on CPU, DataLoader will handle GPU transfer with pin_memory
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample
    
    def get_sample_info(self, idx):
        """Get additional information about a sample"""
        original_sample = self.data[idx]
        return {
            'bs_pos': original_sample['bs_pos'],
            'map_size': original_sample['map_size'],
            'grid_spacing': original_sample['grid_spacing'],
            'buildings': original_sample['buildings'],
            'config_idx': original_sample['config_idx'],
            'num_buildings': original_sample['num_buildings']
        }
    
    def get_building_configurations(self):
        """Get all building configurations used in the dataset"""
        return self.building_config_sets
    
    def get_dataloader(self, batch_size=None, shuffle=True, **kwargs):
        """Get optimized DataLoader for this dataset"""
        if batch_size is None:
            batch_size = self.batch_size
        
        # Check if tensors are on GPU
        tensors_on_gpu = False
        if self.tensor_data and len(self.tensor_data) > 0:
            sample_tensor = self.tensor_data[0]
            if hasattr(sample_tensor, 'device') and sample_tensor.device.type == 'cuda':
                tensors_on_gpu = True
        
        if tensors_on_gpu:
            # Tensors are on GPU - must use single-threaded DataLoader
            dataloader_kwargs = {
                'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': 0,  # Must be 0 when tensors are on GPU
                'pin_memory': False,  # Not needed when data is already on GPU
                'persistent_workers': False,
                # Don't set prefetch_factor when num_workers=0
            }
            print(f"DataLoader config: workers=0 (GPU tensors), pin_memory=False, device={self.device}")
        else:
            # Tensors are on CPU - can use multiprocessing
            dataloader_kwargs = {
                'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': min(self.num_workers or 4, 4),
                'pin_memory': True,  # Pin memory for fast GPU transfer
                'prefetch_factor': 2,
                'persistent_workers': True,
            }
            print(f"DataLoader config: workers={dataloader_kwargs['num_workers']}, pin_memory=True, device={self.device}")
        
        # Allow overrides from kwargs
        dataloader_kwargs.update(kwargs)
        
        # Ensure prefetch_factor is not set when num_workers=0
        if dataloader_kwargs.get('num_workers', 0) == 0:
            dataloader_kwargs.pop('prefetch_factor', None)
        
        return torch.utils.data.DataLoader(self, **dataloader_kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test the GPU-optimized dataset
    dataset = AoAAmpBuildingDataset(
        root="./data/building_training_gpu",
        map_size=(100, 100),
        grid_spacing=1.0,
        bs_grid_spacing=15.0,  # Larger spacing for testing
        num_building_sets=12,  # Smaller number for testing
        building_distribution=(4, 4, 4),  # 4 configs each
        use_existing_data=False,  # Generate new data for testing
        device='auto',
        use_gpu_processing=True,
        batch_size=16,
        num_workers=4,
        seed=42
    )
    
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample device: {sample.device}")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        # Test optimized dataloader
        dataloader = dataset.get_dataloader(batch_size=8, shuffle=True)
        print(f"Created optimized DataLoader with {len(dataloader)} batches")
        
        # Test a batch
        batch = next(iter(dataloader))
        print(f"Batch shape: {batch.shape}")
        print(f"Batch device: {batch.device}")