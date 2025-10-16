"""
Training script for diffusion model on AoA/Amplitude data with buildings.
This script trains on the ray tracing data with fixed buildings and strongest 3 paths.
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from tqdm import tqdm
import numpy as np

from guided_diffusion.gaussian_diffusion import create_sampler, extract_and_expand
from guided_diffusion.unet import create_model
from data.dataloader import get_dataset, get_dataloader
from data.aoa_amp_building_dataset import AoAAmpBuildingDataset  # Import to register the dataset
from util.logger import get_logger
import torch.multiprocessing as mp


def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, step, loss, checkpoint_dir, filename="checkpoint.pt"):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss
    }
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path}, step: {step}, loss: {loss:.4f}")
    return step, loss


def compute_loss(model, diffusion, batch, device):
    """Compute diffusion training loss"""
    
    batch = batch.to(device)
    
    # Sample random timesteps
    batch_size = batch.shape[0]
    timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
    
    # Generate noise
    noise = torch.randn_like(batch)
    
    # Add noise to data using the diffusion process coefficients
    coef1 = extract_and_expand(diffusion.sqrt_alphas_cumprod, timesteps, batch)
    coef2 = extract_and_expand(diffusion.sqrt_one_minus_alphas_cumprod, timesteps, batch)
    noisy_batch = coef1 * batch + coef2 * noise
    
    # Predict noise with the model
    model_output = model(noisy_batch, timesteps)
    
    # Handle learned variance case
    if model_output.shape[1] == batch.shape[1] * 2:  # learn_sigma=True
        # Split the output: first half is noise prediction, second half is variance
        predicted_noise = model_output[:, :batch.shape[1]]
        predicted_variance = model_output[:, batch.shape[1]:]
        target = noise
        
        # Only compute loss on the noise prediction part for now
        loss = F.mse_loss(predicted_noise, target, reduction='mean')
    else:
        # Standard case: model predicts just the noise
        target = noise
        loss = F.mse_loss(model_output, target, reduction='mean')
    
    return loss


def train_step(model, diffusion, optimizer, batch, device):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    loss = compute_loss(model, diffusion, batch, device)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def sample_and_save(model, diffusion, device, save_path, num_samples=4, data_channels=6, image_size=100):
    """Generate and save samples for visualization"""
    model.eval()
    
    with torch.no_grad():
        # Start from random noise
        shape = (num_samples, data_channels, image_size, image_size)
        img = torch.randn(shape, device=device)
        
        # Simple sampling loop (simplified DDPM sampling)
        for i in reversed(range(0, diffusion.num_timesteps, 50)):  # Sample every 50 steps for speed
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # Get model prediction
            with torch.no_grad():
                model_output = model(img, t)
                
                # Handle learned variance case
                if model_output.shape[1] == data_channels * 2:
                    predicted_noise = model_output[:, :data_channels]
                else:
                    predicted_noise = model_output
                
                # Simple denoising step
                if i > 0:
                    # Add some noise for non-final steps
                    noise = torch.randn_like(img)
                    
                    # Use diffusion coefficients for proper denoising
                    alpha_t = extract_and_expand(diffusion.alphas_cumprod, t, img)
                    beta_t = extract_and_expand(diffusion.betas, t, img)
                    
                    # Simplified denoising (approximation of DDPM sampling)
                    img = (img - beta_t * predicted_noise / torch.sqrt(1 - alpha_t)) / torch.sqrt(1 - beta_t)
                    
                    if i > 1:
                        img += torch.sqrt(beta_t) * noise
                else:
                    # Final step
                    img = img - predicted_noise
        
        # Convert to numpy and save
        samples_np = img.cpu().numpy()
        np.save(save_path, samples_np)
        print(f"Samples saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on AoA/Amplitude data with buildings")
    parser.add_argument('--model_config', type=str, default='configs/aoa_amp_building_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml',
                       help='Path to diffusion configuration file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/aoa_amp_building',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    parser.add_argument('--log_dir', type=str, default='./logs/aoa_amp_building',
                       help='Directory for tensorboard logs')
    parser.add_argument('--generate_data', action='store_true',
                       help='Generate new training data before training')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger()
    
    # Device setup
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    
    # Extract training parameters
    batch_size = model_config.get('batch_size', 8)
    learning_rate = float(model_config.get('learning_rate', 1e-4))
    num_epochs = model_config.get('num_epochs', 500)
    save_interval = model_config.get('save_interval', 12000)
    log_interval = model_config.get('log_interval', 50)
    epoch_save_interval = model_config.get('epoch_save_interval', 1)
    
    logger.info(f"Training parameters: lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}")
    
    # Get data channels from config
    data_channels = model_config.get('data_channels', 6)  # 6 channels for building data
    image_size = model_config.get('image_size', 100)
    
    # Create model
    model_params = {
        k: v
        for k, v in model_config.items()
        if k
        not in [
            'batch_size',
            'learning_rate',
            'num_epochs',
            'save_interval',
            'epoch_save_interval',
            'log_interval',
            'dataset',
            'dataloader',
            'data_channels',
        ]
    }
    
    model = create_model(**model_params)
    
    # Replace the input and output layers to match our data channels
    if data_channels != 3:
        # Replace input layer
        old_input_layer = model.input_blocks[0][0]
        new_input_layer = torch.nn.Conv2d(
            data_channels, 
            old_input_layer.out_channels,
            kernel_size=old_input_layer.kernel_size,
            stride=old_input_layer.stride,
            padding=old_input_layer.padding,
            bias=old_input_layer.bias is not None
        )
        model.input_blocks[0][0] = new_input_layer
        
        # Replace output layer
        old_output_layer = model.out[-1]
        expected_out_channels = data_channels * 2 if model_config.get('learn_sigma', False) else data_channels
        new_output_layer = torch.nn.Conv2d(
            old_output_layer.in_channels,
            expected_out_channels,
            kernel_size=old_output_layer.kernel_size,
            stride=old_output_layer.stride,
            padding=old_output_layer.padding,
            bias=old_output_layer.bias is not None
        )
        model.out[-1] = new_output_layer
    
    model = model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create diffusion sampler
    diffusion = create_sampler(**diffusion_config)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Generate data if requested
    # if args.generate_data:
    #     logger.info("Generating new training data...")
    #     from aoa_amp_building_data_gpu import generate_building_training_data_gpu_batch
        
    #     dataset_config = model_config['dataset']
    #     generate_building_training_data_gpu_batch(
    #         map_size=tuple(dataset_config['map_size']),
    #         grid_spacing=dataset_config['grid_spacing'],
    #         bs_grid_spacing=dataset_config['bs_grid_spacing'],
    #         building_configs=dataset_config['building_configs'],
    #         save_dir=dataset_config['root']
    #     )
    
    # Setup dataset and dataloader
    dataset_config = model_config['dataset']
    dataloader_config = model_config.get('dataloader', {})
    
    train_dataset = get_dataset(**dataset_config)
    # Use the dataset's custom dataloader method instead of the generic one
    if hasattr(train_dataset, 'get_dataloader'):
        # Create dataloader kwargs with our primary parameters
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': True
        }
        
        # Add any additional parameters from config, but don't override our primary ones
        for key, value in dataloader_config.items():
            if key not in dataloader_kwargs:
                dataloader_kwargs[key] = value
            
        train_dataloader = train_dataset.get_dataloader(**dataloader_kwargs)
    else:
        # Fallback to generic dataloader with CUDA-safe settings
        train_dataloader = get_dataloader(
            train_dataset, 
            batch_size, 
            num_workers=0,  # Force 0 workers for CUDA compatibility
            train=True
        )
    
    logger.info(f"Dataset size: {len(train_dataset)}")
    logger.info(f"Number of batches: {len(train_dataloader)}")
    
    if len(train_dataset) > 0:
        sample_shape = train_dataset[0].shape
        logger.info(f"Sample shape: {sample_shape}")
        logger.info(f"Sample range: [{train_dataset[0].min():.3f}, {train_dataset[0].max():.3f}]")
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint(model, optimizer, args.resume)
    
    # Training loop
    logger.info("Starting training...")
    step = start_step
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Training step
            loss = train_step(model, diffusion, optimizer, batch, device)
            
            epoch_loss += loss
            num_batches += 1
            step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            # # Logging
            # if step % log_interval == 0:
            #     writer.add_scalar('Train/Loss', loss, step)
            #     logger.info(f"Step {step}, Loss: {loss:.4f}")
            
            # # Save checkpoint 
            # if step % save_interval == 0:
            #     avg_loss = epoch_loss / num_batches
            #     save_checkpoint(model, optimizer, step, avg_loss, args.checkpoint_dir, 
            #                   f"checkpoint_step_{step}.pt")
                
            #     # Generate samples for visualization
            #     sample_path = os.path.join(args.checkpoint_dir, f"samples_step_{step}.npy")
            #     sample_and_save(model, diffusion, device, sample_path, 
            #                   data_channels=data_channels, image_size=image_size)
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        writer.add_scalar('Train/EpochLoss', avg_epoch_loss, epoch)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save end-of-epoch checkpoint
        if (epoch + 1) % epoch_save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                avg_epoch_loss,
                args.checkpoint_dir,
                f"checkpoint_epoch_{epoch+1}.pt"
            )

            sample_path = os.path.join(
                args.checkpoint_dir,
                f"samples_epoch_{epoch+1}.npy"
            )
            sample_and_save(
                model,
                diffusion,
                device,
                sample_path,
                data_channels=data_channels,
                image_size=image_size
            )
    
    logger.info("Training completed!")
    writer.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
