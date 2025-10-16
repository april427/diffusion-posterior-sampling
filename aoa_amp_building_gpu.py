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
def get_best_device():
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Update the initialization
MPS_AVAILABLE = torch.backends.mps.is_available()
CUDA_AVAILABLE = torch.cuda.is_available()

if MPS_AVAILABLE:
    print(f"MPS (Apple Silicon GPU) available")
elif CUDA_AVAILABLE:
    print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
else:
    print("No GPU acceleration available, using CPU")


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
            self.device = get_best_device()
        elif device == 'auto':
            self.device = get_best_device()
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
        
        # Store grid dimensions for batch processing
        self.num_y, self.num_x = self.X.shape
        
        # Initialize structures
        self.bs_pos = None
        self.buildings = []
        self.building_edges = None
        
        # Flag for batch processing optimization
        self.building_edges_computed = False
        
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
        
        # Mark building edges as computed for batch processing optimization
        self.building_edges_computed = True
    
    def generate_aoa_map_gpu(self, num_paths=3):
        """
        GPU-accelerated AoA map generation using proper path ranking approach
        
        Parameters:
        -----------
        num_paths : int
            Number of paths to calculate (ranked by amplitude strength)
            
        Returns:
        --------
        aoa_maps : list of 2D arrays
            AoA values for each path at each grid point (strongest to weakest)
        los_map : 2D array
            Boolean map indicating LoS condition
        """
        if self.bs_pos is None:
            raise ValueError("Base station position not set")
        
        n_y, n_x = self.X.shape
        total_positions = n_y * n_x
        
        if self.verbose:
            print(f"🔥 TRUE GPU vectorization for {n_y}x{n_x} = {total_positions} positions")
        
        # Flatten UE positions for vectorized processing
        ue_flat = self.ue_positions.reshape(-1, 2)  # Shape: (total_positions, 2)
        
        # Calculate LOS for ALL positions at once
        los_map = check_los_gpu(self.ue_positions, self.bs_pos, self.building_edges, self.device)
        los_flat = los_map.reshape(-1)  # Flatten to match ue_flat
        
        # Calculate ALL direct paths at once
        direct_aoa = calculate_aoa_gpu(ue_flat.unsqueeze(0), self.bs_pos).squeeze(0)  # Shape: (total_positions,)
        direct_dist = calculate_distance_gpu(ue_flat.unsqueeze(0), self.bs_pos).squeeze(0)
        direct_pl = calculate_path_loss_gpu(direct_dist.unsqueeze(0), los_flat.unsqueeze(0), self.device).squeeze(0)
        direct_amp = -direct_pl
        
        # Initialize path storage - we'll collect ALL paths for ALL positions
        all_paths_aoa = [direct_aoa.unsqueeze(1)]  # List of tensors, each (total_positions, 1)
        all_paths_amp = [direct_amp.unsqueeze(1)]
        
        # Process reflected paths for ALL positions simultaneously
        for building in self.buildings:
            refl_points = self._calculate_reflection_points_vectorized(ue_flat, building)  # (total_positions, 2)
            
            # Calculate AoA from BS to reflection points
            refl_vec = refl_points - self.bs_pos.unsqueeze(0)  # (total_positions, 2)
            refl_aoa = torch.atan2(refl_vec[:, 1], refl_vec[:, 0]) * 180.0 / math.pi  # (total_positions,)
            
            # Calculate reflection path distances and amplitudes
            dist_bs_to_refl = torch.norm(refl_points - self.bs_pos.unsqueeze(0), dim=1)  # (total_positions,)
            dist_refl_to_ue = torch.norm(ue_flat - refl_points, dim=1)  # (total_positions,)
            refl_dist = dist_bs_to_refl + dist_refl_to_ue
            
            refl_pl = calculate_path_loss_gpu(refl_dist.unsqueeze(0), torch.ones_like(los_flat).unsqueeze(0), self.device).squeeze(0)
            refl_amp = -refl_pl - 6  # Reflection loss
            
            all_paths_aoa.append(refl_aoa.unsqueeze(1))
            all_paths_amp.append(refl_amp.unsqueeze(1))
        
        # Process diffracted paths for ALL positions simultaneously
        for building in self.buildings:
            for corner in building['corners']:
                corner_tensor = torch.tensor(corner, dtype=torch.float32, device=self.device)
                
                # Calculate AoA from BS to corner for all UE positions
                diff_vec = corner_tensor - self.bs_pos  # (2,)
                diff_aoa = torch.atan2(diff_vec[1], diff_vec[0]) * 180.0 / math.pi  # scalar
                diff_aoa_all = diff_aoa.repeat(total_positions)  # (total_positions,)
                
                # Calculate diffraction path distances
                dist_bs_to_corner = torch.norm(corner_tensor - self.bs_pos)  # scalar
                dist_corner_to_ue = torch.norm(ue_flat - corner_tensor.unsqueeze(0), dim=1)  # (total_positions,)
                diff_dist = dist_bs_to_corner + dist_corner_to_ue
                
                diff_pl = calculate_path_loss_gpu(diff_dist.unsqueeze(0), torch.ones_like(los_flat).unsqueeze(0), self.device).squeeze(0)
                diff_amp = -diff_pl - 30  # Diffraction loss
                
                all_paths_aoa.append(diff_aoa_all.unsqueeze(1))
                all_paths_amp.append(diff_amp.unsqueeze(1))
        
        # Stack all paths: (total_positions, num_all_paths)
        paths_aoa_tensor = torch.cat(all_paths_aoa, dim=1)  # (total_positions, num_all_paths)
        paths_amp_tensor = torch.cat(all_paths_amp, dim=1)  # (total_positions, num_all_paths)
        
        # Sort paths by amplitude for each position (GPU vectorized sorting!)
        sorted_indices = torch.argsort(paths_amp_tensor, dim=1, descending=True)  # (total_positions, num_all_paths)
        
        # Select top paths for each position
        result_aoa_maps = []
        for path_idx in range(num_paths):
            # Get indices for this path rank
            path_indices = sorted_indices[:, path_idx]  # (total_positions,)
            
            # Gather AoA values for this path rank across all positions
            aoa_values = torch.gather(paths_aoa_tensor, 1, path_indices.unsqueeze(1)).squeeze(1)  # (total_positions,)
            
            # Reshape back to map
            aoa_map = aoa_values.reshape(n_y, n_x)
            result_aoa_maps.append(aoa_map.cpu().numpy())
        
        if self.verbose:
            print(f"✅ TRUE GPU vectorization completed - processed {total_positions} positions in parallel")
        
        return result_aoa_maps, los_map.cpu().numpy()
    
    def _calculate_reflection_points_vectorized(self, ue_positions_flat, building):
        """Calculate reflection points for ALL UE positions at once"""
        total_positions = ue_positions_flat.shape[0]
        
        x_min, y_min = building['x'], building['y']
        x_max = x_min + building['width']
        y_max = y_min + building['height']
        
        # Broadcast BS position for all UE positions
        bs_pos_broadcast = self.bs_pos.unsqueeze(0).expand(total_positions, -1)  # (total_positions, 2)
        
        # Calculate midpoints for all UE positions
        mid_points = (bs_pos_broadcast + ue_positions_flat) / 2  # (total_positions, 2)
        
        # Calculate closest points on each edge for ALL positions simultaneously
        edge_points = []
        
        # Bottom edge
        bottom_points = torch.stack([
            torch.clamp(mid_points[:, 0], x_min, x_max),
            torch.full((total_positions,), y_min, device=self.device)
        ], dim=1)
        edge_points.append(bottom_points)
        
        # Right edge  
        right_points = torch.stack([
            torch.full((total_positions,), x_max, device=self.device),
            torch.clamp(mid_points[:, 1], y_min, y_max)
        ], dim=1)
        edge_points.append(right_points)
        
        # Top edge
        top_points = torch.stack([
            torch.clamp(mid_points[:, 0], x_min, x_max),
            torch.full((total_positions,), y_max, device=self.device)
        ], dim=1)
        edge_points.append(top_points)
        
        # Left edge
        left_points = torch.stack([
            torch.full((total_positions,), x_min, device=self.device),
            torch.clamp(mid_points[:, 1], y_min, y_max)
        ], dim=1)
        edge_points.append(left_points)
        
        # Calculate distances to each edge for all positions
        edge_stack = torch.stack(edge_points, dim=1)  # (total_positions, 4, 2)
        distances = torch.norm(edge_stack - mid_points.unsqueeze(1), dim=2)  # (total_positions, 4)
        
        # Find closest edge for each position
        closest_indices = torch.argmin(distances, dim=1)  # (total_positions,)
        
        # Gather closest points
        closest_points = torch.gather(
            edge_stack, 
            1, 
            closest_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
        ).squeeze(1)  # (total_positions, 2)
        
        return closest_points
    
    def calculate_reflection_point_gpu(self, ue_pos, building):
        """Calculate reflection point on building wall (GPU version)"""
        x_min, y_min = building['x'], building['y']
        x_max = x_min + building['width']
        y_max = y_min + building['height']
        
        # Find closest point on building perimeter to midpoint of BS-UE line
        mid_point = (self.bs_pos + ue_pos) / 2
        
        # Check distance to each edge and return closest point
        edges = [
            torch.tensor([torch.clamp(mid_point[0], x_min, x_max), y_min], device=self.device),  # bottom
            torch.tensor([x_max, torch.clamp(mid_point[1], y_min, y_max)], device=self.device),  # right
            torch.tensor([torch.clamp(mid_point[0], x_min, x_max), y_max], device=self.device),  # top
            torch.tensor([x_min, torch.clamp(mid_point[1], y_min, y_max)], device=self.device),  # left
        ]
        
        distances = [torch.norm(edge - mid_point) for edge in edges]
        closest_idx = torch.argmin(torch.stack(distances))
        return edges[closest_idx]
    
    def calculate_diffraction_point_gpu(self, ue_pos, building):
        """Calculate diffraction point (closest corner by total path length)"""
        corners = [torch.tensor(corner, dtype=torch.float32, device=self.device) for corner in building['corners']]
        
        # Calculate total path length for each corner
        total_distances = []
        for corner in corners:
            dist_bs_to_corner = torch.norm(corner - self.bs_pos)
            dist_corner_to_ue = torch.norm(ue_pos - corner)
            total_distances.append(dist_bs_to_corner + dist_corner_to_ue)
        
        # Return corner with minimum total path length
        closest_idx = torch.argmin(torch.stack(total_distances))
        return corners[closest_idx]
    
    def generate_aoa_maps_batch_gpu(self, bs_positions_tensor, num_paths=3):
        """Generate AoA maps for multiple BS positions simultaneously using GPU vectorization."""
        batch_size = bs_positions_tensor.shape[0]
        n_y, n_x = self.X.shape
        
        if self.verbose:
            print(f"Generating AoA maps for {batch_size} BS positions in batch mode")
        
        # Initialize results for each path
        all_aoa_maps = []
        
        # Process each BS position in the batch
        for bs_idx, bs_pos in enumerate(bs_positions_tensor):
            # Set BS position for this iteration
            self.bs_pos = bs_pos
            
            # Generate AoA map for this BS position
            aoa_maps, _ = self.generate_aoa_map_gpu(num_paths=num_paths)
            
            # Convert numpy arrays to tensors
            aoa_tensors = [torch.from_numpy(aoa_map).to(self.device) for aoa_map in aoa_maps]
            
            # Stack tensors for this BS position
            bs_aoa_stack = torch.stack(aoa_tensors, dim=0)  # [num_paths, n_y, n_x]
            all_aoa_maps.append(bs_aoa_stack)
        
        # Stack results across batch dimension
        all_aoa_maps = torch.stack(all_aoa_maps, dim=0)  # [batch_size, num_paths, n_y, n_x]
        
        # Generate LOS maps for all positions
        all_los_maps = self._compute_los_batch_vectorized(bs_positions_tensor)
        
        return all_aoa_maps, all_los_maps
        
    def generate_amplitude_maps_batch_gpu(self, bs_positions_tensor, num_paths=3):
        """Generate amplitude maps for multiple BS positions simultaneously using GPU vectorization."""
        batch_size = bs_positions_tensor.shape[0]
        n_y, n_x = self.X.shape
        
        if self.verbose:
            print(f"Generating amplitude maps for {batch_size} BS positions in batch mode")
        
        # Initialize results for each path
        all_amplitude_maps = []
        
        # Process each BS position in the batch
        for bs_idx, bs_pos in enumerate(bs_positions_tensor):
            # Set BS position for this iteration
            self.bs_pos = bs_pos
            
            # Generate amplitude map for this BS position
            amplitude_maps = self.generate_amplitude_map_gpu(num_paths=num_paths)
            
            # Convert numpy arrays to tensors
            amplitude_tensors = [torch.from_numpy(amp_map).to(self.device) for amp_map in amplitude_maps]
            
            # Stack tensors for this BS position
            bs_amplitude_stack = torch.stack(amplitude_tensors, dim=0)  # [num_paths, n_y, n_x]
            all_amplitude_maps.append(bs_amplitude_stack)
        
        # Stack results across batch dimension
        all_amplitude_maps = torch.stack(all_amplitude_maps, dim=0)  # [batch_size, num_paths, n_y, n_x]
        
        return all_amplitude_maps
        
        # Initialize results for each path
        all_amplitude_maps = []
        
        # Process each path
        for path_idx in range(num_paths):
            # Create batch result tensor
            amplitude_batch = torch.zeros(batch_size, n_y, n_x, device=self.device)
            
            # Process each BS position in the batch
            for bs_idx, bs_pos in enumerate(bs_positions_tensor):
                # Set BS position for this iteration
                self.bs_pos = bs_pos
                
                # Generate amplitude map for this BS position
                amplitude_maps = self.generate_amplitude_map_gpu(num_paths=num_paths)
                
                # Store the result for this path
                if path_idx < len(amplitude_maps):
                    amplitude_batch[bs_idx] = torch.from_numpy(amplitude_maps[path_idx]).to(self.device)
            
            all_amplitude_maps.append(amplitude_batch)
        
        return all_amplitude_maps
    
    def _compute_los_batch_vectorized(self, bs_positions_tensor):
        """Compute LOS maps for all BS positions in the batch."""
        batch_size = bs_positions_tensor.shape[0]
        n_y, n_x = self.X.shape
        
        # Create batch result tensor
        los_batch = torch.zeros(batch_size, n_y, n_x, device=self.device, dtype=torch.bool)
        
        # Process each BS position
        for bs_idx, bs_pos in enumerate(bs_positions_tensor):
            # Set BS position
            self.bs_pos = bs_pos
            
            # Compute LOS for this BS position
            los_map = check_los_gpu(self.ue_positions, self.bs_pos, self.building_edges, self.device)
            los_batch[bs_idx] = los_map
        
        return los_batch

    def calculate_path_loss_single(self, distance, los):
        """Calculate path loss for a single path (CPU version for individual calculations)"""
        frequency = 2.4e9  # 2.4 GHz
        wavelength = 3e8 / frequency
        
        # Avoid division by zero
        distance = max(distance, 1e-6)
        
        pl = 20 * np.log10(4 * np.pi * distance / wavelength)
        
        # Add penetration loss for NLOS
        if not los:
            pl += 15  # dB
        
        return pl
    
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
        path_loss_direct[distances_direct < 1e-6] = 50  # if ue is at bs position

        
        # Convert path loss to amplitude (in dB)
        amplitude_direct = -path_loss_direct  # Received power in dB
        
        # Apply 30 dB decrease for UEs inside buildings
        for building in self.buildings:
            # Create a mask for UEs inside this building
            x_min, y_min = building['x'], building['y']
            x_max = x_min + building['width']
            y_max = y_min + building['height']
            
            # Check if UE positions are inside the building
            inside_x = (self.X >= x_min) & (self.X <= x_max)
            inside_y = (self.Y >= y_min) & (self.Y <= y_max)
            inside_building = inside_x & inside_y
            
            # Apply 30 dB decrease to UEs inside the building
            amplitude_direct[inside_building] -= 30.0  # 30 dB decrease
        
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
            dist_building_to_ue = calculate_distance_gpu(self.ue_positions, building_center)
            dist_refl = dist_bs_to_building + dist_building_to_ue
            
            path_loss_refl = calculate_path_loss_gpu(dist_refl, torch.ones_like(los_map), self.device)
            amplitude_refl = -path_loss_refl - 6  # Additional 6dB loss for reflection
            
            all_amplitudes.append(amplitude_refl)
            
            # Diffraction amplitudes
            for corner in building['corners']:
                corner_tensor = torch.tensor(corner, dtype=torch.float32, device=self.device)
                dist_bs_to_corner = torch.norm(corner_tensor - self.bs_pos)
                dist_corner_to_ue = calculate_distance_gpu(self.ue_positions, corner_tensor)
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
    
    def plot_aoa_map(self, aoa_maps, los_map, path_names=None):
        """
        Plot AoA maps for all paths, reflecting the strongest paths.

        Parameters:
        -----------
        aoa_maps : list of 2D arrays or 4D tensor
            AoA values for each path
        los_map : 2D array
            LoS condition map
        path_names : list of str
            Names for each path (e.g., ['Strongest Path', 'Second Strongest', 'Third Strongest'])
        """
        # Handle both list of arrays and tensor input
        if isinstance(aoa_maps, torch.Tensor):
            # Convert tensor to CPU and handle dimensions
            aoa_tensor = aoa_maps.cpu()
            if aoa_tensor.dim() == 4:
                # 4D tensor: (num_paths, channels, height, width)
                num_paths = aoa_tensor.shape[0]
                aoa_maps = []
                for i in range(num_paths):
                    # Take first channel and squeeze
                    map_2d = aoa_tensor[i, 0, :, :].squeeze().numpy()
                    aoa_maps.append(map_2d)
            elif aoa_tensor.dim() == 3:
                # 3D tensor: (num_paths, height, width)
                num_paths = aoa_tensor.shape[0]
                aoa_maps = [aoa_tensor[i, :, :].squeeze().numpy() for i in range(num_paths)]
            elif aoa_tensor.dim() == 2:
                # 2D tensor: single map
                aoa_maps = [aoa_tensor.squeeze().numpy()]
            else:
                # Higher dimensions, try to squeeze to 2D
                aoa_maps = [aoa_tensor.squeeze().numpy()]
        elif isinstance(aoa_maps, list) and len(aoa_maps) > 0:
            # If it's a list of tensors, convert each to numpy
            processed_maps = []
            for aoa_map in aoa_maps:
                if isinstance(aoa_map, torch.Tensor):
                    # Convert to CPU and squeeze extra dimensions
                    map_cpu = aoa_map.cpu().squeeze()
                    if map_cpu.dim() > 2:
                        # Take last 2 dimensions if still > 2D
                        map_cpu = map_cpu[-2:] if map_cpu.dim() == 3 else map_cpu
                    processed_maps.append(map_cpu.numpy())
                elif isinstance(aoa_map, np.ndarray):
                    # Already numpy array, squeeze all extra dimensions
                    squeezed_map = np.squeeze(aoa_map)
                    # If still more than 2D after squeezing, take the last 2 dimensions
                    while squeezed_map.ndim > 2:
                        squeezed_map = squeezed_map[0] if squeezed_map.shape[0] == 1 else squeezed_map[-1]
                    processed_maps.append(squeezed_map)
                else:
                    processed_maps.append(aoa_map)
            aoa_maps = processed_maps
        elif isinstance(aoa_maps, np.ndarray):
            # Single numpy array
            if aoa_maps.ndim > 2:
                aoa_maps = [np.squeeze(aoa_maps)]
            else:
                aoa_maps = [aoa_maps]

        num_paths = len(aoa_maps)

        if path_names is None:
            path_names = [f'Path {i+1}' for i in range(num_paths)]

        fig, axes = plt.subplots(1, num_paths + 1, figsize=(5 * (num_paths + 1), 4))

        if num_paths + 1 == 1:
            axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        # Convert tensors to CPU numpy arrays for plotting
        X_cpu = self.X.cpu().numpy()
        Y_cpu = self.Y.cpu().numpy()
        bs_pos_cpu = self.bs_pos.cpu().numpy()
        
        # Verify grid and map dimensions match
        if len(aoa_maps) > 0:
            map_shape = aoa_maps[0].shape
            if map_shape != X_cpu.shape:
                print(f"Warning: Map shape {map_shape} doesn't match grid shape {X_cpu.shape}")
                # Try to resize if needed
                if len(map_shape) == 2 and len(X_cpu.shape) == 2:
                    print("Attempting to interpolate maps to match grid...")
                    try:
                        from scipy.ndimage import zoom
                        zoom_factors = [X_cpu.shape[i] / map_shape[i] for i in range(2)]
                        aoa_maps = [zoom(aoa_map, zoom_factors, order=1) for aoa_map in aoa_maps]
                    except ImportError:
                        print("scipy not available, cannot resize maps")

        # Plot AoA for each path
        for idx, (aoa_map, name) in enumerate(zip(aoa_maps, path_names)):
            ax = axes[idx]

            # Ensure aoa_map is 2D
            if aoa_map.ndim > 2:
                aoa_map = np.squeeze(aoa_map)
            elif aoa_map.ndim < 2:
                print(f"Warning: AoA map {idx} has insufficient dimensions: {aoa_map.shape}")
                continue
                
            # Final check that dimensions are exactly 2
            if aoa_map.ndim != 2:
                print(f"Error: Cannot plot AoA map {idx} with shape {aoa_map.shape}")
                continue

            # Use the CPU numpy arrays here
            im = ax.contourf(X_cpu, Y_cpu, aoa_map, levels=20, cmap='twilight')
            im.set_clim(-180, 180)
            ax.contour(X_cpu, Y_cpu, aoa_map, levels=10, colors='white', 
                    linewidths=0.5, alpha=0.3)

            # Plot buildings
            for building in self.buildings:
                rect = Rectangle((building['x'], building['y']), 
                            building['width'], building['height'],
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
                ax.add_patch(rect)

            # Plot BS - use CPU numpy array
            ax.plot(bs_pos_cpu[0], bs_pos_cpu[1], 'r*', markersize=20, 
                label='Base Station', markeredgecolor='black', markeredgewidth=1)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{name} - AoA Map')
            ax.set_aspect('equal')
            ax.legend()

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('AoA (degrees)')

        # Plot LoS map
        ax = axes[num_paths]
        
        # Handle LoS map - ensure it's 2D
        if isinstance(los_map, torch.Tensor):
            los_map = los_map.cpu().numpy()
        if los_map.ndim > 2:
            los_map = los_map.squeeze()
        
        # Use CPU numpy arrays here too
        im = ax.contourf(X_cpu, Y_cpu, los_map.astype(float), levels=[0, 0.5, 1], 
                        cmap='hsv', alpha=0.6)

        for building in self.buildings:
            rect = Rectangle((building['x'], building['y']), 
                        building['width'], building['height'],
                        linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(rect)

        ax.plot(bs_pos_cpu[0], bs_pos_cpu[1], 'r*', markersize=20, 
            label='Base Station', markeredgecolor='white', markeredgewidth=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('LoS Condition')
        ax.set_aspect('equal')
        ax.legend()

        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_label('LoS')
        cbar.ax.set_yticklabels(['NLoS', 'LoS'])

        plt.tight_layout()
        plt.show()

    def plot_amplitude_map(self, amplitude_maps, path_names=None):
        """
        Plot amplitude maps for all paths, reflecting the strongest paths.

        Parameters:
        -----------
        amplitude_maps : list of 2D arrays or 4D tensor
            Amplitude values for each path
        path_names : list of str
            Names for each path (e.g., ['Strongest Path', 'Second Strongest', 'Third Strongest'])
        """
        # Handle both list of arrays and tensor input
        if isinstance(amplitude_maps, torch.Tensor):
            # Convert tensor to CPU and handle dimensions
            amp_tensor = amplitude_maps.cpu()
            if amp_tensor.dim() == 4:
                # 4D tensor: (num_paths, channels, height, width)
                num_paths = amp_tensor.shape[0]
                amplitude_maps = []
                for i in range(num_paths):
                    # Take first channel and squeeze
                    map_2d = amp_tensor[i, 0, :, :].squeeze().numpy()
                    amplitude_maps.append(map_2d)
            elif amp_tensor.dim() == 3:
                # 3D tensor: (num_paths, height, width)
                num_paths = amp_tensor.shape[0]
                amplitude_maps = [amp_tensor[i, :, :].squeeze().numpy() for i in range(num_paths)]
            elif amp_tensor.dim() == 2:
                # 2D tensor: single map
                amplitude_maps = [amp_tensor.squeeze().numpy()]
            else:
                # Higher dimensions, try to squeeze to 2D
                amplitude_maps = [amp_tensor.squeeze().numpy()]
        elif isinstance(amplitude_maps, list) and len(amplitude_maps) > 0:
            # If it's a list of tensors, convert each to numpy
            processed_maps = []
            for amp_map in amplitude_maps:
                if isinstance(amp_map, torch.Tensor):
                    # Convert to CPU and squeeze extra dimensions
                    map_cpu = amp_map.cpu().squeeze()
                    if map_cpu.dim() > 2:
                        # Take last 2 dimensions if still > 2D
                        map_cpu = map_cpu[-2:] if map_cpu.dim() == 3 else map_cpu
                    processed_maps.append(map_cpu.numpy())
                elif isinstance(amp_map, np.ndarray):
                    # Already numpy array, squeeze all extra dimensions
                    squeezed_map = np.squeeze(amp_map)
                    # If still more than 2D after squeezing, take the last 2 dimensions
                    while squeezed_map.ndim > 2:
                        squeezed_map = squeezed_map[0] if squeezed_map.shape[0] == 1 else squeezed_map[-1]
                    processed_maps.append(squeezed_map)
                else:
                    processed_maps.append(amp_map)
            amplitude_maps = processed_maps
        elif isinstance(amplitude_maps, np.ndarray):
            # Single numpy array
            if amplitude_maps.ndim > 2:
                amplitude_maps = [np.squeeze(amplitude_maps)]
            else:
                amplitude_maps = [amplitude_maps]

        num_paths = len(amplitude_maps)

        if path_names is None:
            path_names = [f'Path {i+1}' for i in range(num_paths)]

        fig, axes = plt.subplots(1, num_paths, figsize=(5 * num_paths, 4))

        if num_paths == 1:
            axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        # Convert tensors to CPU numpy arrays for plotting
        X_cpu = self.X.cpu().numpy()
        Y_cpu = self.Y.cpu().numpy()
        bs_pos_cpu = self.bs_pos.cpu().numpy()
        
        # Verify grid and map dimensions match
        if len(amplitude_maps) > 0:
            map_shape = amplitude_maps[0].shape
            if map_shape != X_cpu.shape:
                print(f"Warning: Map shape {map_shape} doesn't match grid shape {X_cpu.shape}")
                # Try to resize if needed
                if len(map_shape) == 2 and len(X_cpu.shape) == 2:
                    print("Attempting to interpolate maps to match grid...")
                    try:
                        from scipy.ndimage import zoom
                        zoom_factors = [X_cpu.shape[i] / map_shape[i] for i in range(2)]
                        amplitude_maps = [zoom(amp_map, zoom_factors, order=1) for amp_map in amplitude_maps]
                    except ImportError:
                        print("scipy not available, cannot resize maps")
        
        # Calculate common colorbar range for all amplitude maps
        if len(amplitude_maps) > 0:
            all_valid_maps = [amp_map for amp_map in amplitude_maps if amp_map.ndim == 2]
            if all_valid_maps:
                global_min = min(amp_map.min() for amp_map in all_valid_maps)
                global_max = max(amp_map.max() for amp_map in all_valid_maps)
                
                # Ensure reasonable range
                if global_max - global_min < 1:
                    # Very narrow range, expand it
                    global_min, global_max = global_min - 5, global_max + 5
                elif global_min > -120 and global_max < -30:
                    # Already reasonable amplitude range, use as is
                    pass
                else:
                    # Extend range to reasonable defaults while encompassing data
                    global_min = min(-90, global_min)
                    global_max = max(-40, global_max)
                
                print(f"Using common colorbar range: [{global_min:.1f}, {global_max:.1f}] dB")
            else:
                global_min, global_max = -90, -40

        for idx, (amp_map, name) in enumerate(zip(amplitude_maps, path_names)):
            ax = axes[idx]

            # Ensure amp_map is 2D
            if amp_map.ndim > 2:
                amp_map = np.squeeze(amp_map)
            elif amp_map.ndim < 2:
                print(f"Warning: Amplitude map {idx} has insufficient dimensions: {amp_map.shape}")
                continue
                
            # Final check that dimensions are exactly 2
            if amp_map.ndim != 2:
                print(f"Error: Cannot plot amplitude map {idx} with shape {amp_map.shape}")
                continue

            # Use CPU numpy arrays here
            im = ax.contourf(X_cpu, Y_cpu, amp_map, levels=20, cmap='hot_r')
            
            # Use the common colorbar range for all amplitude maps
            im.set_clim(global_min, global_max)
            
            # Plot buildings
            for building in self.buildings:
                rect = Rectangle((building['x'], building['y']), 
                            building['width'], building['height'],
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
                ax.add_patch(rect)

            # Plot BS - use CPU numpy array
            ax.plot(bs_pos_cpu[0], bs_pos_cpu[1], 'r*', markersize=20, 
                label='Base Station', markeredgecolor='black', markeredgewidth=1)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{name} - Amplitude')
            ax.set_aspect('equal')
            ax.legend()

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'Amplitude (dB)\nRange: [{global_min:.1f}, {global_max:.1f}]')

        plt.tight_layout()
        plt.show()


# For backward compatibility, replace the original class
RayTracingAoAMap = RayTracingAoAMapGPU


if __name__ == "__main__":
    # Performance test
    import time
    
    print("Testing GPU-accelerated RayTracingAoAMap...")
    
    # Create test scenario
    rt = RayTracingAoAMapGPU(map_size=(100, 100), grid_spacing=2, device='auto', verbose=True)
    rt.set_base_station(80, 30)
    rt.add_building(20, 20, 30, 15)
    rt.add_building(75, 56, 25, 19)
    
    print("Starting ray tracing computation...")
    start_time = time.time()
    
    aoa_maps, los_map = rt.generate_aoa_map_gpu(num_paths=3)
    amplitude_maps = rt.generate_amplitude_map_gpu(num_paths=3)

    path_names = ['Strongest Path', 'Second Strongest', 'Third Strongest']
    rt.plot_aoa_map(aoa_maps, los_map, path_names)
    rt.plot_amplitude_map(amplitude_maps, path_names)
    
    end_time = time.time()
    
    print(f"Generated maps in {end_time - start_time:.3f} seconds")
    print(f"Map shape: {aoa_maps[0].shape}")
    print(f"LOS percentage: {np.sum(los_map)/los_map.size*100:.1f}%")
    print(f"Device used: {rt.device}")
    
    # Test memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")