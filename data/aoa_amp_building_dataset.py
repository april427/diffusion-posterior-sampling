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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

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
        print(f"  - {self.building_distribution[0]} sets with 1 building")
        print(f"  - {self.building_distribution[1]} sets with 2 buildings") 
        print(f"  - {self.building_distribution[2]} sets with 3 buildings")
        
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
                
                # Try to load pre-converted tensor data
                if os.path.exists(tensor_cache_file):
                    print(f"Loading pre-converted tensor data from {tensor_cache_file}...")
                    if self.device.type == 'cuda':
                        self.tensor_data = torch.load(tensor_cache_file, map_location=self.device)
                    else:
                        self.tensor_data = torch.load(tensor_cache_file, map_location='cpu')
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
        """Generate data for all building configurations using parallel processing"""
        self.data = []
        
        # Create tasks for parallel data generation
        tasks = [(i, config) for i, config in enumerate(self.building_config_sets)]
        
        print(f"Generating training data for {len(tasks)} configurations using {self.num_workers} workers...")
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._generate_config_data_worker, tasks))
        
        # Flatten results
        for config_data in results:
            self.data.extend(config_data)
    
    def _generate_config_data_worker(self, task):
        """Worker function for parallel data generation"""
        config_idx, building_config = task
        
        # Import the function here to avoid circular import
        from aoa_amp_building_data import generate_building_training_data
        
        print(f"Generating data for building configuration {config_idx + 1}/{len(self.building_config_sets)} "
              f"({len(building_config)} buildings)...")
        
        config_data = generate_building_training_data(
            map_size=self.map_size,
            grid_spacing=self.grid_spacing,
            bs_grid_spacing=self.bs_grid_spacing,
            building_configs=building_config,
            save_dir=None  # Don't save individual configs to disk
        )
        
        # Add configuration index to each sample
        for sample in config_data:
            sample['config_idx'] = config_idx
            sample['num_buildings'] = len(building_config)
        
        return config_data
    
    def _convert_to_tensors_gpu(self):
        """Convert data to tensors using GPU acceleration"""
        print("Converting data to tensor format using GPU...")
        
        if self.use_gpu_processing and self.device.type == 'cuda':
            self._convert_with_gpu_batching()
        else:
            self._convert_with_cpu_batching()
    
    def _convert_with_gpu_batching(self):
        """Convert data using GPU batch processing"""
        self.tensor_data = []
        num_samples = len(self.data)
        
        for i in range(0, num_samples, self.batch_size):
            batch_end = min(i + self.batch_size, num_samples)
            batch_samples = self.data[i:batch_end]
            
            # Process batch on GPU
            batch_tensors = []
            for sample in batch_samples:
                tensor = self._convert_to_tensor_gpu(sample)
                batch_tensors.append(tensor)
            
            # Stack and move to GPU
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Move back to CPU for storage if needed
                if self.pin_memory:
                    batch_tensor = batch_tensor.pin_memory()
                
                # Split back to individual tensors
                for j in range(batch_tensor.shape[0]):
                    self.tensor_data.append(batch_tensor[j])
            
            if (batch_end) % 500 == 0:
                print(f"Converted {batch_end}/{num_samples} samples")
    
    def _convert_with_cpu_batching(self):
        """Convert data using CPU with parallel processing"""
        print("Converting data using CPU parallel processing...")
        
        # Use ThreadPoolExecutor for CPU-bound tensor conversion
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            self.tensor_data = list(executor.map(self._convert_to_tensor, self.data))
        
        # Pin memory if requested
        if self.pin_memory and self.device.type == 'cuda':
            print("Pinning memory for faster GPU transfer...")
            self.tensor_data = [tensor.pin_memory() for tensor in self.tensor_data]
    
    def _convert_to_tensor_gpu(self, sample_data):
        """Convert single sample to tensor using GPU operations"""
        aoa_maps = sample_data['aoa_maps']  # List of 3 2D arrays
        amplitude_maps = sample_data['amplitude_maps']  # List of 3 2D arrays
        
        # Stack all maps: shape = (6, H, W) - 3 AoA + 3 Amplitude
        all_maps = np.stack(aoa_maps + amplitude_maps, axis=0)
        
        # Convert to tensor and move to GPU
        tensor = torch.from_numpy(all_maps).float().to(self.device)
        
        if self.normalize_data:
            # Normalize AoA maps (first 3 channels) from [-180, 180] to [-1, 1]
            tensor[:3] = tensor[:3] / 180.0
            
            # Normalize amplitude maps (last 3 channels) 
            # Amplitude is in dB, typically ranging from -90 to -40 dB
            amp_data = tensor[3:]
            tensor[3:] = 2 * (amp_data - (-90)) / ((-40) - (-90)) - 1
            tensor[3:] = torch.clamp(tensor[3:], -1, 1)
        
        return tensor.cpu()  # Move back to CPU for storage
    
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
        
        # Stack all tensors for efficient saving
        if self.tensor_data:
            stacked_tensors = torch.stack(self.tensor_data)
            torch.save(stacked_tensors, cache_file)
            
            # Convert back to list
            self.tensor_data = [stacked_tensors[i] for i in range(stacked_tensors.shape[0])]
    
    def _print_dataset_statistics(self):
        """Print dataset statistics"""
        config_counts = {}
        for sample in self.data:
            num_buildings = sample['num_buildings']
            config_counts[num_buildings] = config_counts.get(num_buildings, 0) + 1
        
        print("Dataset statistics:")
        for num_buildings in sorted(config_counts.keys()):
            print(f"  - {config_counts[num_buildings]} samples with {num_buildings} building(s)")
        
        if self.tensor_data:
            print(f"Tensor data device: {self.tensor_data[0].device if hasattr(self.tensor_data[0], 'device') else 'CPU'}")
    
    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: Shape (6, H, W) where first 3 channels are AoA maps
                         and last 3 channels are amplitude maps for strongest paths
        """
        sample = self.tensor_data[idx]
        
        # Move to GPU if needed
        if self.device.type == 'cuda' and sample.device.type == 'cpu':
            sample = sample.to(self.device, non_blocking=True)
        
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
        
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory and self.device.type == 'cuda',
            'prefetch_factor': self.prefetch_factor,
            'persistent_workers': True if self.num_workers > 0 else False,
        }
        dataloader_kwargs.update(kwargs)
        
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