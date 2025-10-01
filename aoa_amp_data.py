"generate training dataset"
"Each datapoint is a map of AoA and amplitude when a BS is randomly placed"


import numpy as np
import matplotlib.pyplot as plt

def generate_training_data(grid_resolution, bs_positions, wavelength=0.1):
    """
    Generate training dataset of AoA and amplitude maps for given BS positions.

    Parameters:
    num_samples (int): Number of samples to generate.
    bs_positions (list of tuples): List of (x, y) positions for the base stations.

    Returns:
    list of dicts: Each dict contains 'aoa_map' and 'amplitude_map'.
    """
    data = []

    x_lim = [-50, 50]
    y_lim = [-50, 50]
    num_samples_x = int((x_lim[1] - x_lim[0]) / grid_resolution)
    num_samples_y = int((y_lim[1] - y_lim[0]) / grid_resolution)
    num_samples = int(num_samples_x * num_samples_y)

    user_position = np.array(np.meshgrid(np.linspace(x_lim[0], x_lim[1], num_samples_x), \
                        np.linspace(y_lim[0], y_lim[1], num_samples_y))).T.reshape(-1, 2)

    aoa_map = {}
    amplitude_map = {}
    
    for idx, (bs_x, bs_y) in enumerate(bs_positions):
        # Calculate AoA (angle in degrees)
        delta_x = user_position[:,0] - bs_x
        delta_y = user_position[:,1] - bs_y
        aoa = np.arctan2(delta_y, delta_x) # [-pi, pi]
        
        # Calculate amplitude (inverse distance)
        distance = np.sqrt(delta_x**2 + delta_y**2)
        amplitude = (4*np.pi * (distance + 1e-6))**2/wavelength  # Avoid division by zero
        
        aoa_map[f'bs_{idx}'] = aoa
        amplitude_map[f'bs_{idx}'] = 10*np.log10(amplitude)
    
    data.append({'aoa_map': aoa_map, 'amplitude_map': amplitude_map})
    
    return data


if __name__ == "__main__":
    grid_resolution = 1  # Example grid resolution
    bs_positions = [(-20, -20)]  # Example BS positions
    dataset = generate_training_data(grid_resolution, bs_positions, wavelength=3e8/2.4e9)
    
    fig, axs = plt.subplots(2, len(bs_positions), figsize=(15, 6))
    for idx, bs in enumerate(bs_positions):
        aoa_map = dataset[0]['aoa_map'][f'bs_{idx}']
        amplitude_map = dataset[0]['amplitude_map'][f'bs_{idx}']

        num_samples_x = int((50 - (-50)) / grid_resolution)
        num_samples_y = int((50 - (-50)) / grid_resolution)

        aoa_map_reshaped = aoa_map.reshape(num_samples_x, num_samples_y)
        amplitude_map_reshaped = amplitude_map.reshape(num_samples_x, num_samples_y)

        im0 = axs[0].imshow(aoa_map_reshaped, extent=[-50, 50, -50, 50], origin='lower', cmap='hsv')
        axs[0].set_title(f'AoA Heatmap BS {idx}')
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(amplitude_map_reshaped, extent=[-50, 50, -50, 50], origin='lower', cmap='viridis')
        axs[1].set_title(f'Amplitude Heatmap BS {idx}')
        plt.colorbar(im1, ax=axs[1])
        plt.show()
