# generate_sample_map.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.aoa_amp_dataset import AoAAmpDataset

def generate_and_visualize_sample():
    """Generate and visualize one AoA and amplitude map"""
    
    # Create dataset with small size for quick generation
    dataset = AoAAmpDataset(
        root="./data/aoa_amp_cache",
        grid_resolution=2.0,     # 50x50 grid
        num_bs=1,                # Single base station
        num_samples=1,           # Just one sample
        bs_range=(-40, 40),      # BS position range
        wavelength=0.125,        # 2.4GHz wavelength: 3e8/2.4e9 ≈ 0.125m
        cache_data=False         # Don't cache for this single sample
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Get the first (and only) sample
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
    
    # Extract AoA and amplitude maps
    # Sample shape is (2*num_bs, H, W) = (2, 50, 50) for num_bs=1
    aoa_map = sample[0].numpy()      # First channel: AoA
    amp_map = sample[1].numpy()      # Second channel: Amplitude
    
    print(f"AoA map shape: {aoa_map.shape}")
    print(f"AoA range: [{aoa_map.min():.3f}, {aoa_map.max():.3f}]")
    print(f"Amplitude map shape: {amp_map.shape}")
    print(f"Amplitude range: [{amp_map.min():.3f}, {amp_map.max():.3f}]")
    
    # Convert normalized values back to original ranges for visualization
    aoa_original = aoa_map * np.pi  # Convert from [-1,1] back to [-π,π]
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot AoA map
    im1 = axes[0].imshow(aoa_original, extent=[-50, 50, -50, 50], 
                         origin='lower', cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title('Angle of Arrival (AoA) Map')
    axes[0].set_xlabel('X position (m)')
    axes[0].set_ylabel('Y position (m)')
    plt.colorbar(im1, ax=axes[0], label='AoA (radians)')
    
    # Plot amplitude map (normalized)
    im2 = axes[1].imshow(amp_map, extent=[-50, 50, -50, 50], 
                         origin='lower', cmap='viridis', vmin=-1, vmax=1)
    axes[1].set_title('Amplitude Map (Normalized)')
    axes[1].set_xlabel('X position (m)')
    axes[1].set_ylabel('Y position (m)')
    plt.colorbar(im2, ax=axes[1], label='Normalized Amplitude')
    
    # Plot combined view (AoA as hue, amplitude as brightness)
    # Create HSV image where H=AoA, S=1, V=amplitude
    aoa_normalized = (aoa_original + np.pi) / (2 * np.pi)  # [0, 1]
    amp_normalized = (amp_map + 1) / 2  # [-1, 1] -> [0, 1]
    
    hsv_img = np.zeros((*aoa_map.shape, 3))
    hsv_img[:, :, 0] = aoa_normalized  # Hue = AoA
    hsv_img[:, :, 1] = 1.0             # Full saturation
    hsv_img[:, :, 2] = amp_normalized  # Value = amplitude
    
    from matplotlib.colors import hsv_to_rgb
    rgb_img = hsv_to_rgb(hsv_img)
    
    axes[2].imshow(rgb_img, extent=[-50, 50, -50, 50], origin='lower')
    axes[2].set_title('Combined View\n(Color=AoA, Brightness=Amplitude)')
    axes[2].set_xlabel('X position (m)')
    axes[2].set_ylabel('Y position (m)')
    
    plt.tight_layout()
    
    plt.show()
    
    # Print some statistics
    print("\n" + "="*50)
    print("SAMPLE STATISTICS")
    print("="*50)
    print(f"Grid resolution: {dataset.grid_resolution}m")
    print(f"Grid size: {dataset.num_samples_x} x {dataset.num_samples_y}")
    print(f"Number of base stations: {dataset.num_bs}")
    print(f"BS range: {dataset.bs_range}")
    print(f"Wavelength: {dataset.wavelength}m")
    
    # Show the tensor that would be fed to the diffusion model
    print(f"\nTensor shape for diffusion model: {sample.shape}")
    print(f"Tensor dtype: {sample.dtype}")
    print(f"Tensor device: {sample.device}")
    
    return sample, aoa_map, amp_map

if __name__ == "__main__":
    sample, aoa_map, amp_map = generate_and_visualize_sample()