"""
Test script to verify AoA/Amplitude dataset and training setup
"""

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

from data.dataloader import get_dataset, get_dataloader
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.unet import create_model


def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_dataset():
    """Test the dataset creation and loading"""
    print("Testing AoA/Amplitude dataset...")
    
    # Load config
    config = load_yaml('configs/aoa_amp_config.yaml')
    dataset_config = config['dataset']
    
    # Create dataset with small number of samples for testing
    dataset_config['num_samples'] = 10
    dataset = get_dataset(**dataset_config)
    
    print(f"Dataset created successfully!")
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")
    print(f"Sample range: [{dataset[0].min():.3f}, {dataset[0].max():.3f}]")
    
    # Visualize a sample
    sample = dataset[0].numpy()
    aoa_map = sample[0]  # First channel is AoA
    amp_map = sample[1]  # Second channel is amplitude
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(aoa_map, cmap='hsv', extent=[-50, 50, -50, 50])
    ax1.set_title('AoA Map')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(amp_map, cmap='viridis', extent=[-50, 50, -50, 50])
    ax2.set_title('Amplitude Map')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('test_aoa_amp_sample.png')
    print("Sample visualization saved as 'test_aoa_amp_sample.png'")
    
    return dataset


def test_dataloader(dataset):
    """Test dataloader functionality"""
    print("\nTesting dataloader...")
    
    config = load_yaml('configs/aoa_amp_config.yaml')
    dataloader_config = config['dataloader']
    dataloader_config['batch_size'] = 4  # Small batch for testing
    
    dataloader = get_dataloader(dataset, train=True, **dataloader_config)
    
    # Test loading a batch
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    print(f"Batch range: [{batch.min():.3f}, {batch.max():.3f}]")
    
    return dataloader


def test_model():
    """Test model creation"""
    print("\nTesting model creation...")
    
    # Load configs
    model_config = load_yaml('configs/aoa_amp_config.yaml')
    diffusion_config = load_yaml('configs/diffusion_config.yaml')
    
    # Extract model parameters
    model_params = {k: v for k, v in model_config.items() 
                   if k not in ['batch_size', 'learning_rate', 'num_epochs', 'save_interval', 'log_interval', 'dataset', 'dataloader']}
    
    # Create model
    model = create_model(**model_params)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, model_config['in_channels'], 
                            model_config['image_size'], model_config['image_size'])
    test_timesteps = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output = model(test_input, test_timesteps)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    
    return model


def test_diffusion():
    """Test diffusion sampler"""
    print("\nTesting diffusion sampler...")
    
    diffusion_config = load_yaml('configs/diffusion_config.yaml')
    diffusion = create_sampler(**diffusion_config)
    
    print(f"Diffusion sampler created successfully!")
    print(f"Number of timesteps: {diffusion.num_timesteps}")
    print(f"Model mean type: {diffusion.model_mean_type}")
    print(f"Model var type: {diffusion.model_var_type}")
    
    return diffusion


def test_training_step(model, diffusion, dataloader):
    """Test a single training step"""
    print("\nTesting training step...")
    
    device = torch.device('cpu')  # Use CPU for testing
    model = model.to(device)
    
    # Get a batch
    batch = next(iter(dataloader)).to(device)
    
    # Sample random timesteps
    batch_size = batch.shape[0]
    timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
    
    # Add noise to data
    noise = torch.randn_like(batch)
    noisy_batch = diffusion.q_sample(batch, timesteps, noise=noise)
    
    # Forward pass
    model_output = model(noisy_batch, timesteps)
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(model_output, noise)
    
    print(f"Training step completed successfully!")
    print(f"Batch shape: {batch.shape}")
    print(f"Noisy batch shape: {noisy_batch.shape}")
    print(f"Model output shape: {model_output.shape}")
    print(f"Loss: {loss.item():.4f}")


def main():
    """Run all tests"""
    print("=" * 50)
    print("AoA/Amplitude Diffusion Model Test Suite")
    print("=" * 50)
    
    try:
        # Test dataset
        dataset = test_dataset()
        
        # Test dataloader
        dataloader = test_dataloader(dataset)
        
        # Test model
        model = test_model()
        
        # Test diffusion
        diffusion = test_diffusion()
        
        # Test training step
        test_training_step(model, diffusion, dataloader)
        
        print("\n" + "=" * 50)
        print("All tests passed! Ready for training.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()