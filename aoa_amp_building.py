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
        self._ranked_maps_cache = None
        
    def set_base_station(self, x, y):
        """Set base station position"""
        self.bs_pos = np.array([x, y])
        self._ranked_maps_cache = None
        
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
        self._ranked_maps_cache = None
    
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

    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """Return Euclidean distance between a point and a line segment."""
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq < 1e-12:
            return np.linalg.norm(point - seg_start)

        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        projection = seg_start + t * seg_vec
        return np.linalg.norm(point - projection)

    def _get_building_walls(self, building):
        """Return rectangle wall segments as (name, start, end)."""
        x_min, y_min = building['x'], building['y']
        x_max = x_min + building['width']
        y_max = y_min + building['height']

        return [
            ('bottom', np.array([x_min, y_min]), np.array([x_max, y_min])),
            ('right', np.array([x_max, y_min]), np.array([x_max, y_max])),
            ('top', np.array([x_max, y_max]), np.array([x_min, y_max])),
            ('left', np.array([x_min, y_max]), np.array([x_min, y_min])),
        ]

    def _get_two_closest_walls(self, ue_pos, building):
        """Pick the two wall facets closest to the UE point."""
        walls = self._get_building_walls(building)
        distances = []
        for idx, (_, wall_start, wall_end) in enumerate(walls):
            dist = self._point_to_segment_distance(ue_pos, wall_start, wall_end)
            distances.append((dist, idx))

        distances.sort(key=lambda item: item[0])
        return [walls[idx] for _, idx in distances[:2]]

    def _reflect_point_across_wall(self, point, wall_start, wall_end):
        """Reflect a point across an axis-aligned wall segment."""
        if abs(wall_start[0] - wall_end[0]) < 1e-8:  # vertical wall x = c
            wall_x = wall_start[0]
            return np.array([2.0 * wall_x - point[0], point[1]])

        if abs(wall_start[1] - wall_end[1]) < 1e-8:  # horizontal wall y = c
            wall_y = wall_start[1]
            return np.array([point[0], 2.0 * wall_y - point[1]])

        return None

    def _line_wall_intersection(self, line_start, line_end, wall_start, wall_end):
        """Return intersection between a line segment and a wall segment if valid."""
        d = line_end - line_start
        e = wall_end - wall_start
        denom = d[0] * e[1] - d[1] * e[0]

        if abs(denom) < 1e-10:
            return None

        delta = wall_start - line_start
        t = (delta[0] * e[1] - delta[1] * e[0]) / denom
        u = (delta[0] * d[1] - delta[1] * d[0]) / denom

        # Reflection point must be on BS-image segment and on the finite wall length.
        if not (0.0 < t < 1.0 and 0.0 <= u <= 1.0):
            return None

        return line_start + t * d
    
    def calculate_reflection_point(self, ue_pos, building):
        """Calculate a valid image-theory reflection point for one building."""
        two_closest_walls = self._get_two_closest_walls(ue_pos, building)

        for _, wall_start, wall_end in two_closest_walls:
            ue_image = self._reflect_point_across_wall(ue_pos, wall_start, wall_end)
            if ue_image is None:
                continue

            reflection_point = self._line_wall_intersection(
                self.bs_pos, ue_image, wall_start, wall_end
            )
            if reflection_point is not None:
                return reflection_point

        return None
    
    def calculate_diffraction_point(self, ue_pos, building):
        """Calculate diffraction point (closest corner)"""
        corners = building['corners']
        distances = [np.linalg.norm(self.bs_pos - corner) + np.linalg.norm(ue_pos - corner) 
                    for corner in corners]
        closest_idx = np.argmin(distances)
        return np.array(corners[closest_idx])

    def _generate_reflection_candidates(self, ue_pos):
        """Generate reflection candidates from two closest walls per building."""
        candidates = []

        for building in self.buildings:
            two_closest_walls = self._get_two_closest_walls(ue_pos, building)
            for _, wall_start, wall_end in two_closest_walls:
                ue_image = self._reflect_point_across_wall(ue_pos, wall_start, wall_end)
                if ue_image is None:
                    continue

                reflection_point = self._line_wall_intersection(
                    self.bs_pos, ue_image, wall_start, wall_end
                )
                if reflection_point is None:
                    continue

                dist_bs_to_refl = np.linalg.norm(reflection_point - self.bs_pos)
                dist_refl_to_ue = np.linalg.norm(ue_pos - reflection_point)
                total_dist = dist_bs_to_refl + dist_refl_to_ue
                if total_dist < 1e-6:
                    continue

                path_loss = self.calculate_path_loss(total_dist, True) + 6.0
                amplitude_db = -path_loss
                vec = reflection_point - self.bs_pos
                aoa = np.arctan2(vec[1], vec[0]) * 180 / np.pi

                candidates.append({
                    'aoa': aoa,
                    'amplitude_db': amplitude_db,
                    'distance': total_dist,
                })

        candidates.sort(key=lambda item: item['amplitude_db'], reverse=True)
        return candidates

    def _generate_diffraction_candidates(self, ue_pos):
        """Generate diffraction candidates from all building corners."""
        candidates = []
        for building in self.buildings:
            for corner in building['corners']:
                corner = np.array(corner, dtype=np.float64)
                dist_bs_to_corner = np.linalg.norm(corner - self.bs_pos)
                dist_corner_to_ue = np.linalg.norm(ue_pos - corner)
                total_dist = dist_bs_to_corner + dist_corner_to_ue

                path_loss = self.calculate_path_loss(total_dist, True) + 30.0
                amplitude_db = -path_loss
                vec = corner - self.bs_pos
                aoa = np.arctan2(vec[1], vec[0]) * 180 / np.pi

                candidates.append({
                    'aoa': aoa,
                    'amplitude_db': amplitude_db,
                    'distance': total_dist,
                })

        # Closest-corner ordering per requirement.
        candidates.sort(key=lambda item: item['distance'])
        return candidates

    def _build_ranked_paths_for_ue(self, ue_pos, los, num_paths):
        """Build path list: direct + reflections, then diffraction fallback."""
        direct_dist = np.linalg.norm(ue_pos - self.bs_pos)
        direct_dist = max(direct_dist, 1e-6)
        direct_amp = -self.calculate_path_loss(direct_dist, los)
        direct_aoa = self.calculate_aoa(ue_pos)

        ranked_paths = [
            {
                'aoa': direct_aoa,
                'amplitude_db': direct_amp,
                'distance': direct_dist,
            }
        ]

        reflection_candidates = self._generate_reflection_candidates(ue_pos)
        for candidate in reflection_candidates:
            if len(ranked_paths) >= num_paths:
                break
            ranked_paths.append(candidate)

        if len(ranked_paths) < num_paths:
            diffraction_candidates = self._generate_diffraction_candidates(ue_pos)
            for candidate in diffraction_candidates:
                if len(ranked_paths) >= num_paths:
                    break
                ranked_paths.append(candidate)

        while len(ranked_paths) < num_paths:
            ranked_paths.append(
                {
                    'aoa': direct_aoa,
                    'amplitude_db': -120.0,
                    'distance': np.inf,
                }
            )

        return ranked_paths

    def _generate_ranked_path_maps(self, num_paths=3):
        """Generate AoA/amplitude/LoS maps using the same ranked path assignment."""
        if self.bs_pos is None:
            raise ValueError("Base station position not set")

        if (
            self._ranked_maps_cache is not None
            and self._ranked_maps_cache['num_paths'] == num_paths
        ):
            cached = self._ranked_maps_cache
            return (
                [m.copy() for m in cached['aoa_maps']],
                [m.copy() for m in cached['amplitude_maps']],
                cached['los_map'].copy(),
            )

        n_y, n_x = self.X.shape
        aoa_maps = [np.zeros((n_y, n_x), dtype=np.float32) for _ in range(num_paths)]
        amplitude_maps = [np.full((n_y, n_x), -120.0, dtype=np.float32) for _ in range(num_paths)]
        los_map = np.zeros((n_y, n_x), dtype=bool)

        for i in range(n_y):
            for j in range(n_x):
                ue_pos = np.array([self.X[i, j], self.Y[i, j]], dtype=np.float64)
                los = self.is_los(ue_pos)
                los_map[i, j] = los

                ranked_paths = self._build_ranked_paths_for_ue(ue_pos, los, num_paths)
                for k, path in enumerate(ranked_paths[:num_paths]):
                    aoa_maps[k][i, j] = path['aoa']
                    amplitude_maps[k][i, j] = path['amplitude_db']

        inside_building_mask = np.zeros((n_y, n_x), dtype=bool)
        for building in self.buildings:
            x_min, y_min = building['x'], building['y']
            x_max = x_min + building['width']
            y_max = y_min + building['height']
            inside = (self.X >= x_min) & (self.X <= x_max) & (self.Y >= y_min) & (self.Y <= y_max)
            inside_building_mask |= inside

        for k in range(num_paths):
            amplitude_maps[k][inside_building_mask] -= 20.0

        self._ranked_maps_cache = {
            'num_paths': num_paths,
            'aoa_maps': [m.copy() for m in aoa_maps],
            'amplitude_maps': [m.copy() for m in amplitude_maps],
            'los_map': los_map.copy(),
        }

        return aoa_maps, amplitude_maps, los_map
    
    def generate_aoa_map(self, num_paths=3):
        """
        Generate AoA maps for ranked paths.

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
        aoa_maps, _, los_map = self._generate_ranked_path_maps(num_paths=num_paths)
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
        Generate amplitude maps for ranked paths.

        Returns:
        --------
        amplitude_maps : list of 2D arrays
            Amplitude values for each path in linear scale (V/m)
        """
        _, amplitude_maps, _ = self._generate_ranked_path_maps(num_paths=num_paths)
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