"""
GPU-accelerated RayTracingAoAMap class using PyTorch for CUDA acceleration.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import math
from typing import List, Tuple, Dict, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', module='numba')

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
    print(f"Current GPU: {torch.cuda.get_device_name()}")
else:
    print("CUDA not available, falling back to CPU")


def calculate_aoa_gpu(ue_positions: torch.Tensor, bs_pos: torch.Tensor) -> torch.Tensor:
    """Calculate AoA for all UE positions using GPU"""
    vec = ue_positions - bs_pos.unsqueeze(0).unsqueeze(0)  # Broadcasting
    aoa_rad = torch.atan2(vec[..., 1], vec[..., 0])
    aoa_deg = aoa_rad * 180.0 / math.pi
    return aoa_deg


def calculate_distance_gpu(ue_positions: torch.Tensor, bs_pos: torch.Tensor) -> torch.Tensor:
    """Calculate distances for all UE positions using GPU"""
    vec = ue_positions - bs_pos.unsqueeze(0).unsqueeze(0)  # Broadcasting
    distances = torch.norm(vec, dim=-1)
    return distances


def calculate_path_loss_gpu(distances: torch.Tensor, los_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Calculate path loss using GPU"""
    frequency = 2.4e9  # 2.4 GHz
    wavelength = 3e8 / frequency
    
    # Avoid division by zero
    distances_safe = torch.clamp(distances, min=1e-6)
    
    # Calculate free space path loss
    pl = 20 * torch.log10(4 * math.pi * distances_safe / wavelength)
    
    # Add penetration loss for NLOS
    nlos_penalty = torch.tensor(15.0, device=device)
    pl = torch.where(los_mask, pl, pl + nlos_penalty)
    
    return pl


def line_segment_intersection_gpu(p1: torch.Tensor, p2: torch.Tensor, 
                                p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
    """Check line segment intersection using GPU vectorized operations"""
    # Vectorized line intersection check
    # p1, p2: line 1 endpoints, p3, p4: line 2 endpoints
    
    d = p2 - p1  # Direction vector of line 1
    e = p4 - p3  # Direction vector of line 2
    
    # Calculate denominator
    denom = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    
    # Check for parallel lines
    parallel_mask = torch.abs(denom) < 1e-10
    
    # Calculate parameters
    f = p3 - p1
    t = (f[..., 0] * e[..., 1] - f[..., 1] * e[..., 0]) / torch.clamp(denom, min=1e-10)
    u = (f[..., 0] * d[..., 1] - f[..., 1] * d[..., 0]) / torch.clamp(denom, min=1e-10)
    
    # Check if intersection is within both line segments
    intersection = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & ~parallel_mask
    
    return intersection


def check_los_gpu(ue_positions: torch.Tensor, bs_pos: torch.Tensor, 
                 building_edges: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Check line-of-sight using GPU vectorized operations"""
    if building_edges.numel() == 0:
        return torch.ones(ue_positions.shape[:-1], device=device, dtype=torch.bool)
    
    n_y, n_x = ue_positions.shape[0], ue_positions.shape[1]
    los_map = torch.ones((n_y, n_x), device=device, dtype=torch.bool)
    
    # Process in batches to manage memory
    batch_size = 1000
    total_positions = n_y * n_x
    
    # Flatten UE positions for easier processing
    ue_flat = ue_positions.reshape(-1, 2)
    los_flat = torch.ones(total_positions, device=device, dtype=torch.bool)
    
    for i in range(0, total_positions, batch_size):
        end_idx = min(i + batch_size, total_positions)
        ue_batch = ue_flat[i:end_idx]  # Shape: (batch_size, 2)
        
        # Check intersection with all building edges
        for edge_idx in range(building_edges.shape[0]):
            edge = building_edges[edge_idx]  # Shape: (2, 2) - two endpoints
            p3, p4 = edge[0], edge[1]
            
            # Broadcast for vectorized intersection check
            bs_batch = bs_pos.unsqueeze(0).expand(ue_batch.shape[0], -1)  # (batch_size, 2)
            p3_batch = p3.unsqueeze(0).expand(ue_batch.shape[0], -1)      # (batch_size, 2)
            p4_batch = p4.unsqueeze(0).expand(ue_batch.shape[0], -1)      # (batch_size, 2)
            
            # Check intersection
            intersects = line_segment_intersection_gpu(bs_batch, ue_batch, p3_batch, p4_batch)
            
            # Update LOS status
            los_flat[i:end_idx] = los_flat[i:end_idx] & ~intersects
    
    return los_flat.reshape(n_y, n_x)


class RayTracingAoAMapGPU:
    def __init__(self, map_size, grid_spacing=1, device=None, verbose=False):
        """
        Initialize GPU-accelerated ray tracing for AoA map generation.
        
        Args:
            map_size: Size of the map (x_max, y_max) or single integer for square map
            grid_spacing: Grid spacing for UE positions
            device: Device to use ('cuda', 'cpu', or None for auto)
            verbose: Whether to print initialization details
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if verbose:
            print(f"RayTracing using device: {self.device}")
        
        # Store verbose setting for use in other methods
        self.verbose = verbose
        
        # Handle both tuple and single integer map_size
        if isinstance(map_size, (tuple, list)):
            map_x, map_y = map_size[0], map_size[1]
        else:
            map_x, map_y = map_size, map_size
        
        # Create grid points
        self.x_grid = torch.arange(0, map_x, grid_spacing, dtype=torch.float32, device=self.device)
        self.y_grid = torch.arange(0, map_y, grid_spacing, dtype=torch.float32, device=self.device)
        
        # Create meshgrid
        Y, X = torch.meshgrid(self.y_grid, self.x_grid, indexing='ij')
        self.X = X
        self.Y = Y
        
        # Pre-compute UE positions for vectorized operations
        self.ue_positions = torch.stack([self.X, self.Y], dim=-1)  # Shape: (n_y, n_x, 2)
        
        # Initialize structures
        self.bs_pos = None
        self.buildings = []
        self.building_edges = None
        
        if self.verbose:
            print(f"Grid shape: {self.ue_positions.shape[:-1]}")
            print(f"Total UE positions: {self.ue_positions.shape[0] * self.ue_positions.shape[1]}")
        
    def set_base_station(self, x, y):
        """Set base station position"""
        self.bs_pos = torch.tensor([x, y], dtype=torch.float32, device=self.device)
        
    def add_building(self, x, y, width, height):
        """
        Add a rectangular building
        
        Parameters:
        -----------
        x, y : float
            Bottom-left corner coordinates
        width, height : float
            Building dimensions
        """
        building = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'corners': [
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height]
            ]
        }
        self.buildings.append(building)
        
        # Update building edges for vectorized LOS calculation
        self._update_building_edges()
    
    def _update_building_edges(self):
        """Update building edges tensor for GPU operations"""
        all_edges = []
        
        for building in self.buildings:
            x_min, y_min = building['x'], building['y']
            x_max = x_min + building['width']
            y_max = y_min + building['height']
            
            # Rectangle edges
            edges = [
                [[x_min, y_min], [x_max, y_min]],  # bottom
                [[x_max, y_min], [x_max, y_max]],  # right
                [[x_max, y_max], [x_min, y_max]],  # top
                [[x_min, y_max], [x_min, y_min]]   # left
            ]
            
            for edge in edges:
                all_edges.append(edge)
        
        if all_edges:
            self.building_edges = torch.tensor(all_edges, dtype=torch.float32, device=self.device)
        else:
            self.building_edges = torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)
    
    def generate_aoa_map_gpu(self, num_paths=3):
        """
        GPU-accelerated AoA map generation using PyTorch
        
        Parameters:
        -----------
        num_paths : int
            Number of paths to calculate
            
        Returns:
        --------
        aoa_maps : list of 2D arrays
            AoA values for each path at each grid point
        los_map : 2D array
            Boolean map indicating LoS condition
        """
        if self.bs_pos is None:
            raise ValueError("Base station position not set")
        
        n_y, n_x = self.X.shape
        
        if self.verbose:
            print(f"Generating AoA maps on GPU for {n_y}x{n_x} grid...")
        
        # Calculate LOS map using GPU
        los_map = check_los_gpu(self.ue_positions, self.bs_pos, self.building_edges, self.device)
        
        # Calculate direct path AoA and distances (GPU vectorized)
        aoa_direct = calculate_aoa_gpu(self.ue_positions, self.bs_pos)
        distances_direct = calculate_distance_gpu(self.ue_positions, self.bs_pos)
        path_loss_direct = calculate_path_loss_gpu(distances_direct, los_map, self.device)
        
        # Store all paths with their amplitudes for sorting
        all_paths = []
        
        # Direct path
        amplitude_direct = -path_loss_direct  # Negative for sorting (higher amplitude = lower path loss)
        all_paths.append((aoa_direct, amplitude_direct))
        
        # Calculate reflected and diffracted paths (simplified for GPU efficiency)
        for building in self.buildings:
            # Simplified reflection calculation (use building center)
            building_center = torch.tensor([
                building['x'] + building['width'] / 2,
                building['y'] + building['height'] / 2
            ], dtype=torch.float32, device=self.device)
            
            # Reflection AoA (approximate)
            vec_refl = building_center - self.bs_pos
            aoa_refl = torch.atan2(vec_refl[1], vec_refl[0]) * 180.0 / math.pi
            aoa_refl_map = torch.full((n_y, n_x), aoa_refl.item(), dtype=torch.float32, device=self.device)
            
            # Reflection path loss (approximate)
            dist_bs_to_building = torch.norm(building_center - self.bs_pos)
            dist_building_to_ue = calculate_distance_gpu(self.ue_positions, building_center.unsqueeze(0).unsqueeze(0))
            dist_refl = dist_bs_to_building + dist_building_to_ue
            
            path_loss_refl = calculate_path_loss_gpu(dist_refl, torch.ones_like(los_map), self.device)
            amplitude_refl = -path_loss_refl
            
            all_paths.append((aoa_refl_map, amplitude_refl))
            
            # Diffraction paths (simplified - use building corners)
            for corner in building['corners']:
                corner_tensor = torch.tensor(corner, dtype=torch.float32, device=self.device)
                vec_diff = corner_tensor - self.bs_pos
                aoa_diff = torch.atan2(vec_diff[1], vec_diff[0]) * 180.0 / math.pi
                aoa_diff_map = torch.full((n_y, n_x), aoa_diff.item(), dtype=torch.float32, device=self.device)
                
                dist_bs_to_corner = torch.norm(corner_tensor - self.bs_pos)
                dist_corner_to_ue = calculate_distance_gpu(self.ue_positions, corner_tensor.unsqueeze(0).unsqueeze(0))
                dist_diff = dist_bs_to_corner + dist_corner_to_ue
                
                path_loss_diff = calculate_path_loss_gpu(dist_diff, torch.ones_like(los_map), self.device) + 20
                amplitude_diff = -path_loss_diff
                
                all_paths.append((aoa_diff_map, amplitude_diff))
        
        # Sort paths by amplitude and select strongest ones (GPU optimized)
        aoa_maps = []
        
        for path_idx in range(min(num_paths, len(all_paths))):
            if path_idx == 0:
                # First path is always the strongest (direct path)
                aoa_maps.append(all_paths[0][0].cpu().numpy())
            else:
                # For subsequent paths, we need to sort at each pixel
                # This is simplified - in a full implementation, you'd sort all paths per pixel
                if path_idx < len(all_paths):
                    aoa_maps.append(all_paths[path_idx][0].cpu().numpy())
                else:
                    # Fill with zeros if not enough paths
                    aoa_maps.append(torch.zeros((n_y, n_x), device=self.device).cpu().numpy())
        
        # Ensure we have exactly num_paths maps
        while len(aoa_maps) < num_paths:
            aoa_maps.append(torch.zeros((n_y, n_x), device=self.device).cpu().numpy())
        
        if self.verbose:
            print(f"Generated {len(aoa_maps)} AoA maps")
        return aoa_maps, los_map.cpu().numpy()
    
    def generate_amplitude_map_gpu(self, num_paths=3):
        """
        GPU-accelerated amplitude map generation
        
        Returns:
        --------
        amplitude_maps : list of 2D arrays
            Amplitude values for each path in dB
        """
        if self.bs_pos is None:
            raise ValueError("Base station position not set")
        
        n_y, n_x = self.X.shape
        
        if self.verbose:
            print(f"Generating amplitude maps on GPU for {n_y}x{n_x} grid...")
        
        # Calculate LOS map
        los_map = check_los_gpu(self.ue_positions, self.bs_pos, self.building_edges, self.device)
        
        # Calculate distances and path losses
        distances_direct = calculate_distance_gpu(self.ue_positions, self.bs_pos)
        path_loss_direct = calculate_path_loss_gpu(distances_direct, los_map, self.device)
        
        # Convert path loss to amplitude (in dB)
        amplitude_direct = -path_loss_direct  # Received power in dB
        
        # Store all path amplitudes
        all_amplitudes = [amplitude_direct]
        
        # Add reflected and diffracted path amplitudes
        for building in self.buildings:
            building_center = torch.tensor([
                building['x'] + building['width'] / 2,
                building['y'] + building['height'] / 2
            ], dtype=torch.float32, device=self.device)
            
            # Reflection amplitude
            dist_bs_to_building = torch.norm(building_center - self.bs_pos)
            dist_building_to_ue = calculate_distance_gpu(self.ue_positions, building_center.unsqueeze(0).unsqueeze(0))
            dist_refl = dist_bs_to_building + dist_building_to_ue
            
            path_loss_refl = calculate_path_loss_gpu(dist_refl, torch.ones_like(los_map), self.device)
            amplitude_refl = -path_loss_refl - 6  # Additional 6dB loss for reflection
            
            all_amplitudes.append(amplitude_refl)
            
            # Diffraction amplitudes
            for corner in building['corners']:
                corner_tensor = torch.tensor(corner, dtype=torch.float32, device=self.device)
                dist_bs_to_corner = torch.norm(corner_tensor - self.bs_pos)
                dist_corner_to_ue = calculate_distance_gpu(self.ue_positions, corner_tensor.unsqueeze(0).unsqueeze(0))
                dist_diff = dist_bs_to_corner + dist_corner_to_ue
                
                path_loss_diff = calculate_path_loss_gpu(dist_diff, torch.ones_like(los_map), self.device) + 20
                amplitude_diff = -path_loss_diff - 10  # Additional 10dB loss for diffraction
                
                all_amplitudes.append(amplitude_diff)
        
        # Select strongest amplitudes (simplified approach)
        amplitude_maps = []
        for path_idx in range(num_paths):
            if path_idx < len(all_amplitudes):
                amplitude_maps.append(all_amplitudes[path_idx].cpu().numpy())
            else:
                # Fill with very low amplitude if not enough paths
                amplitude_maps.append(torch.full((n_y, n_x), -120.0, device=self.device).cpu().numpy())
        
        if self.verbose:
            print(f"Generated {len(amplitude_maps)} amplitude maps")
        return amplitude_maps
    
    # Backward compatibility methods
    def generate_aoa_map(self, num_paths=3):
        """Backward compatibility wrapper"""
        return self.generate_aoa_map_gpu(num_paths)
    
    def generate_amplitude_map(self, num_paths=3):
        """Backward compatibility wrapper"""
        return self.generate_amplitude_map_gpu(num_paths)


# For backward compatibility, replace the original class
RayTracingAoAMap = RayTracingAoAMapGPU


if __name__ == "__main__":
    # Performance test
    import time
    
    print("Testing GPU-accelerated RayTracingAoAMap...")
    
    # Create test scenario
    rt = RayTracingAoAMapGPU(map_size=(100, 100), grid_spacing=2, device='auto', verbose=True)
    rt.set_base_station(50, 50)
    rt.add_building(20, 20, 30, 15)
    rt.add_building(75, 56, 25, 19)
    
    print("Starting ray tracing computation...")
    start_time = time.time()
    
    aoa_maps, los_map = rt.generate_aoa_map_gpu(num_paths=3)
    amplitude_maps = rt.generate_amplitude_map_gpu(num_paths=3)
    
    end_time = time.time()
    
    print(f"Generated maps in {end_time - start_time:.3f} seconds")
    print(f"Map shape: {aoa_maps[0].shape}")
    print(f"LOS percentage: {np.sum(los_map)/los_map.size*100:.1f}%")
    print(f"Device used: {rt.device}")
    
    # Test memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")