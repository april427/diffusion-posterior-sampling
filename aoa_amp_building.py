import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

class RayTracingAoAMap:
    def __init__(self, map_size=(100, 100), grid_spacing=10):
        """
        Initialize the ray tracing AoA map generator
        
        Parameters:
        -----------
        map_size : tuple
            Size of the map in meters (width, height)
        grid_spacing : float
            Grid spacing in meters
        """
        self.map_size = map_size
        self.grid_spacing = grid_spacing
        
        # Create grid points
        self.x_grid = np.arange(0, map_size[0], grid_spacing)
        self.y_grid = np.arange(0, map_size[1], grid_spacing)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Initialize structures
        self.bs_pos = None
        self.buildings = []
        
    def set_base_station(self, x, y):
        """Set base station position"""
        self.bs_pos = np.array([x, y])
        
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
        self.buildings.append({
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
        })
    
    def line_intersects_rectangle(self, p1, p2, rect):
        """
        Check if line segment from p1 to p2 intersects with rectangle
        
        Parameters:
        -----------
        p1, p2 : array-like
            Line segment endpoints
        rect : dict
            Rectangle definition with x, y, width, height
        """
        x_min, y_min = rect['x'], rect['y']
        x_max = x_min + rect['width']
        y_max = y_min + rect['height']
        
        # Get rectangle edges
        edges = [
            ([x_min, y_min], [x_max, y_min]),  # bottom
            ([x_max, y_min], [x_max, y_max]),  # right
            ([x_max, y_max], [x_min, y_max]),  # top
            ([x_min, y_max], [x_min, y_min])   # left
        ]
        
        for edge in edges:
            if self.line_segments_intersect(p1, p2, edge[0], edge[1]):
                return True
        return False
    
    def line_segments_intersect(self, p1, p2, p3, p4):
        """Check if two line segments intersect"""
        p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
        
        d = (p2 - p1)
        e = (p4 - p3)
        
        denom = d[0] * e[1] - d[1] * e[0]
        
        if abs(denom) < 1e-10:
            return False
        
        t = ((p3[0] - p1[0]) * e[1] - (p3[1] - p1[1]) * e[0]) / denom
        u = ((p3[0] - p1[0]) * d[1] - (p3[1] - p1[1]) * d[0]) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def is_los(self, ue_pos):
        """Check if there is line-of-sight between BS and UE"""
        for building in self.buildings:
            if self.line_intersects_rectangle(self.bs_pos, ue_pos, building):
                return False
        return True
    
    def calculate_aoa(self, ue_pos):
        """
        Calculate angle of arrival at UE position
        
        Returns AoA in degrees (-180 to 180)
        """
        # Vector from BS to UE
        vec = ue_pos - self.bs_pos
        
        # Calculate angle in degrees
        aoa = np.arctan2(vec[1], vec[0]) * 180 / np.pi
        
        return aoa
    
    def calculate_reflection_point(self, ue_pos, building):
        """
        Calculate reflection point on building wall (simplified)
        Returns the closest point on building perimeter
        """
        x_min, y_min = building['x'], building['y']
        x_max = x_min + building['width']
        y_max = y_min + building['height']
        
        # Find closest point on building perimeter to midpoint of BS-UE line
        mid_point = (self.bs_pos + ue_pos) / 2
        
        # Check distance to each edge
        edges = [
            {'pos': np.array([np.clip(mid_point[0], x_min, x_max), y_min]), 'normal': np.array([0, -1])},
            {'pos': np.array([x_max, np.clip(mid_point[1], y_min, y_max)]), 'normal': np.array([1, 0])},
            {'pos': np.array([np.clip(mid_point[0], x_min, x_max), y_max]), 'normal': np.array([0, 1])},
            {'pos': np.array([x_min, np.clip(mid_point[1], y_min, y_max)]), 'normal': np.array([-1, 0])}
        ]
        
        closest_edge = min(edges, key=lambda e: np.linalg.norm(e['pos'] - mid_point))
        return closest_edge['pos']
    
    def calculate_diffraction_point(self, ue_pos, building):
        """Calculate diffraction point (closest corner)"""
        corners = building['corners']
        distances = [np.linalg.norm(self.bs_pos - corner) + np.linalg.norm(ue_pos - corner) 
                    for corner in corners]
        closest_idx = np.argmin(distances)
        return np.array(corners[closest_idx])
    
    def generate_aoa_map(self, num_paths=3):
        """
        Generate AoA map for all grid points, ranking paths by amplitude.

        Parameters:
        -----------
        num_paths : int
            Number of paths to calculate (1=LoS, 2=Second Strongest, 3=Third Strongest)

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

        # Initialize maps
        aoa_maps = [np.zeros((n_y, n_x)) for _ in range(num_paths)]
        los_map = np.zeros((n_y, n_x), dtype=bool)

        # Calculate for each grid point
        for i in range(n_y):
            for j in range(n_x):
                ue_pos = np.array([self.X[i, j], self.Y[i, j]])

                # Check LoS
                los = self.is_los(ue_pos)
                los_map[i, j] = los

                # Store all paths with their amplitudes and AoAs
                paths = []

                # Path 1: Direct/LoS path
                aoa_los = self.calculate_aoa(ue_pos)
                paths.append((aoa_los, self.calculate_path_loss(np.linalg.norm(ue_pos - self.bs_pos), los)))

                # Reflected paths
                for building in self.buildings:
                    refl_point = self.calculate_reflection_point(ue_pos, building)
                    vec_refl =  refl_point -  self.bs_pos
                    aoa_refl = np.arctan2(vec_refl[1], vec_refl[0]) * 180 / np.pi
                    dist_refl = np.linalg.norm(refl_point - self.bs_pos) + np.linalg.norm(ue_pos - refl_point)
                    paths.append((aoa_refl, self.calculate_path_loss(dist_refl, True)))

                # Diffracted paths
                for building in self.buildings:
                    diff_point = self.calculate_diffraction_point(ue_pos, building)
                    vec_diff =   diff_point - self.bs_pos
                    aoa_diff = np.arctan2(vec_diff[1], vec_diff[0]) * 180 / np.pi
                    dist_diff = np.linalg.norm(diff_point - self.bs_pos) + np.linalg.norm(ue_pos - diff_point)
                    paths.append((aoa_diff, self.calculate_path_loss(dist_diff, True) + 20))

                # Sort paths by amplitude (strongest first)
                paths = sorted(paths, key=lambda x: x[1])

                # Assign strongest paths to maps
                for k in range(min(num_paths, len(paths))):
                    aoa_maps[k][i, j] = paths[k][0]

        return aoa_maps, los_map
    
    def calculate_path_loss(self, distance, los):
        """Calculate free space path loss"""
        frequency = 2.4e9  # 2.4 GHz
        wavelength = 3e8 / frequency
        
        # Avoid division by zero
        distance = max(distance, 1e-6)
        
        pl = 20 * np.log10(4 * np.pi * distance / wavelength)
        
        # Add penetration loss for NLOS
        if not los:
            pl += 15  # dB
        
        return pl
    
    def generate_amplitude_map(self, num_paths=3):
        """
        Generate amplitude map for all paths, ranking paths by amplitude.

        Returns:
        --------
        amplitude_maps : list of 2D arrays
            Amplitude values for each path in linear scale (V/m)
        """
        n_y, n_x = self.X.shape
        wavelength = 0.1  # Wavelength in meters

        amplitude_maps = [np.zeros((n_y, n_x)) for _ in range(num_paths)]

        for i in range(n_y):
            for j in range(n_x):
                ue_pos = np.array([self.X[i, j], self.Y[i, j]])

                if (ue_pos == self.bs_pos).all():
                    # Directly at BS position
                    for k in range(num_paths):
                        amplitude_maps[k][i, j] = -75  # UE cannot be at BS position
                    continue

                # Store all paths with their amplitudes
                paths = []

                # Path 1: Direct path
                distance = np.linalg.norm(ue_pos - self.bs_pos)
                distance = max(distance, 1e-6)  # Avoid division by zero
                los = self.is_los(ue_pos)
                amplitude = 20 * np.log10(wavelength / (4 * np.pi * distance))
                paths.append((amplitude, distance))

                # Reflected paths
                for building in self.buildings:
                    refl_point = self.calculate_reflection_point(ue_pos, building)
                    dist1 = np.linalg.norm(refl_point - self.bs_pos)
                    dist2 = np.linalg.norm(ue_pos - refl_point)
                    total_dist = dist1 + dist2
                    total_dist = max(total_dist, 1e-6)  # Avoid division by zero
                    amplitude = 20 * np.log10(wavelength / (4 * np.pi * total_dist))
                    paths.append((amplitude, total_dist))

                # Diffracted paths
                for building in self.buildings:
                    diff_point = self.calculate_diffraction_point(ue_pos, building)
                    dist1 = np.linalg.norm(diff_point - self.bs_pos)
                    dist2 = np.linalg.norm(ue_pos - diff_point)
                    total_dist = dist1 + dist2
                    total_dist = max(total_dist, 1e-6)  # Avoid division by zero
                    amplitude = 20 * np.log10(wavelength / (4 * np.pi * total_dist))
                    paths.append((amplitude, total_dist))

                # Sort paths by amplitude (strongest first)
                paths = sorted(paths, key=lambda x: -x[0])
                # Assign strongest paths to maps
                for k in range(min(num_paths, len(paths))):
                    amplitude_maps[k][i, j] = paths[k][0]

        # Create a mask for points inside any building
        inside_building_mask = np.zeros((n_y, n_x), dtype=bool)
        for building in self.buildings:
            x_min, y_min = building['x'], building['y']
            x_max = x_min + building['width']
            y_max = y_min + building['height']
            inside = (self.X >= x_min) & (self.X <= x_max) & (self.Y >= y_min) & (self.Y <= y_max)
            inside_building_mask |= inside

        # Apply 20 dB penalty to all amplitudes at points inside buildings
        for k in range(num_paths):
            amplitude_maps[k][inside_building_mask] -= 20

        return amplitude_maps
    
    def plot_aoa_map(self, aoa_maps, los_map, path_names=None):
        """
        Plot AoA maps for all paths, reflecting the strongest paths.

        Parameters:
        -----------
        aoa_maps : list of 2D arrays
            AoA values for each path
        los_map : 2D array
            LoS condition map
        path_names : list of str
            Names for each path (e.g., ['Strongest Path', 'Second Strongest', 'Third Strongest'])
        """
        num_paths = len(aoa_maps)

        if path_names is None:
            path_names = [f'Path {i+1}' for i in range(num_paths)]

        fig, axes = plt.subplots(1, num_paths + 1, figsize=(5 * (num_paths + 1), 4))

        if num_paths == 0:
            axes = [axes]

        # Plot AoA for each path
        for idx, (aoa_map, name) in enumerate(zip(aoa_maps, path_names)):
            ax = axes[idx]

            im = ax.contourf(self.X, self.Y, aoa_map, levels=20, cmap='twilight')
            ax.contour(self.X, self.Y, aoa_map, levels=10, colors='white', 
                      linewidths=0.5, alpha=0.3)

            # Plot buildings
            for building in self.buildings:
                rect = Rectangle((building['x'], building['y']), 
                               building['width'], building['height'],
                               linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
                ax.add_patch(rect)

            # Plot BS
            ax.plot(self.bs_pos[0], self.bs_pos[1], 'r*', markersize=20, 
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
        im = ax.contourf(self.X, self.Y, los_map.astype(float), levels=[0, 0.5, 1], 
                        cmap='hsv', alpha=0.6)

        for building in self.buildings:
            rect = Rectangle((building['x'], building['y']), 
                           building['width'], building['height'],
                           linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(rect)

        ax.plot(self.bs_pos[0], self.bs_pos[1], 'r*', markersize=20, 
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
        amplitude_maps : list of 2D arrays
            Amplitude values for each path
        path_names : list of str
            Names for each path (e.g., ['Strongest Path', 'Second Strongest', 'Third Strongest'])
        """
        num_paths = len(amplitude_maps)

        if path_names is None:
            path_names = [f'Path {i+1}' for i in range(num_paths)]

        fig, axes = plt.subplots(1, num_paths, figsize=(5 * num_paths, 4))

        if num_paths == 1:
            axes = [axes]

        for idx, (amp_map, name) in enumerate(zip(amplitude_maps, path_names)):
            ax = axes[idx]

            # Already in dB
            # amp_db = 10 * np.log10(amp_map + 1e-12)

            im = ax.contourf(self.X, self.Y, amp_map, levels=20, cmap='hot_r')
            im.set_clim(-90, -40)  # Set colorbar range from -90 dB to -40 dB
            # Plot buildings
            for building in self.buildings:
                rect = Rectangle((building['x'], building['y']), 
                               building['width'], building['height'],
                               linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
                ax.add_patch(rect)

            # Plot BS
            ax.plot(self.bs_pos[0], self.bs_pos[1], 'r*', markersize=20, 
                   label='Base Station', markeredgecolor='black', markeredgewidth=1)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{name} - Amplitude')
            ax.set_aspect('equal')
            ax.legend()

            cbar = plt.colorbar(im, ax=ax)
            
            cbar.set_label('Amplitude (dB relative to 1 V/m)')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create ray tracing model
    rt = RayTracingAoAMap(map_size=(100, 100), grid_spacing=1)
    
    # Set base station position
    rt.set_base_station(20, 80)
    
    # Add building(s)
    rt.add_building(20, 20, 30, 15)
    # You can add more buildings:
    rt.add_building(75, 56, 25, 19)

    # Generate AoA maps for 3 paths
    print("Generating AoA maps...")
    aoa_maps, los_map = rt.generate_aoa_map(num_paths=3)
    
    # Generate amplitude maps
    print("Generating amplitude maps...")
    amplitude_maps = rt.generate_amplitude_map(num_paths=3)
    
    # Plot results
    path_names = ['Strongest Path', 'Second Strongest', 'Third Strongest']
    rt.plot_aoa_map(aoa_maps, los_map, path_names)
    rt.plot_amplitude_map(amplitude_maps, path_names)
    
    # Access data at specific grid point
    grid_i, grid_j = 25, 25  # Example grid indices
    print(f"\nAt grid point ({rt.X[grid_i, grid_j]:.1f}, {rt.Y[grid_i, grid_j]:.1f}):")
    print(f"  LoS: {los_map[grid_i, grid_j]}")
    for i, name in enumerate(path_names):
        print(f"  {name} - AoA: {aoa_maps[i][grid_i, grid_j]:.2f}°, "
              f"Amplitude: {amplitude_maps[i][grid_i, grid_j]:.2f} dB")