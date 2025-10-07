"""
Dataset for AoA and Amplitude maps generated using ray tracing with buildings.
This dataset uses the strongest 3 paths approach with fixed building configurations.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing import Optional, Callable, List, Tuple
import os
import pickle
import sys

from data.dataloader import register_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from aoa_amp_building_data import generate_building_training_data, load_training_data


@register_dataset(name='aoa_amp_building')
class AoAAmpBuildingDataset(VisionDataset):
    """
    Dataset for AoA and Amplitude maps generated using ray tracing with buildings.
    Each sample contains 6 channels: 3 AoA maps + 3 amplitude maps for the strongest paths.
    """
    
    def __init__(self, 
                 root: str,
                 transforms: Optional[Callable] = None,
                 map_size: Tuple[int, int] = (100, 100),
                 grid_spacing: float = 1.0,
                 bs_grid_spacing: float = 5.0,
                 building_configs: Optional[List[dict]] = None,
                 use_existing_data: bool = True,
                 normalize_data: bool = True):
        """
        Args:
            root: Root directory for caching data
            transforms: Transform pipeline for data augmentation
            map_size: Size of the map in meters (width, height)
            grid_spacing: Grid spacing for UE positions in meters
            bs_grid_spacing: Grid spacing for BS positions in meters
            building_configs: List of building configurations
            use_existing_data: Whether to use existing data or generate new
            normalize_data: Whether to normalize data to [-1, 1] range
        """
        super().__init__(root, transforms)
        
        self.map_size = map_size
        self.grid_spacing = grid_spacing
        self.bs_grid_spacing = bs_grid_spacing
        self.use_existing_data = use_existing_data
        self.normalize_data = normalize_data
        
        # Default building configuration
        if building_configs is None:
            self.building_configs = [
                {'x': 20, 'y': 20, 'width': 30, 'height': 15},
                {'x': 75, 'y': 56, 'width': 25, 'height': 19}
            ]
        else:
            self.building_configs = building_configs
        
        # Create cache directory
        os.makedirs(root, exist_ok=True)
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Generate or load cached dataset"""
        try:
            if self.use_existing_data:
                print("Loading existing building training data...")
                self.data = load_training_data(
                    save_dir=self.root, 
                    grid_spacing=self.bs_grid_spacing
                )
            else:
                raise FileNotFoundError("Generating new data as requested")
                
        except FileNotFoundError:
            print("Generating new building training data...")
            self.data = generate_building_training_data(
                map_size=self.map_size,
                grid_spacing=self.grid_spacing,
                bs_grid_spacing=self.bs_grid_spacing,
                building_configs=self.building_configs,
                save_dir=self.root
            )
        
        # Convert to tensor format
        print("Converting data to tensor format...")
        self.tensor_data = []
        for i, sample in enumerate(self.data):
            tensor_sample = self._convert_to_tensor(sample)
            self.tensor_data.append(tensor_sample)
            
            if (i + 1) % 100 == 0:
                print(f"Converted {i + 1}/{len(self.data)} samples")
        
        print(f"Dataset prepared with {len(self.tensor_data)} samples")
    
    def _convert_to_tensor(self, sample_data):
        """Convert AoA/amplitude maps to tensor format"""
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
    
    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: Shape (6, H, W) where first 3 channels are AoA maps
                         and last 3 channels are amplitude maps for strongest paths
        """
        sample = self.tensor_data[idx]
        
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
            'buildings': original_sample['buildings']
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    dataset = AoAAmpBuildingDataset(
        root="./data/building_training",
        map_size=(100, 100),
        grid_spacing=1.0,
        bs_grid_spacing=10.0,  # Larger spacing for testing
        use_existing_data=False  # Generate new data for testing
    )
    
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        # Check sample info
        info = dataset.get_sample_info(0)
        print(f"BS position: {info['bs_pos']}")
        print(f"Map size: {info['map_size']}")
        print(f"Grid spacing: {info['grid_spacing']}")
        print(f"Number of buildings: {len(info['buildings'])}")
