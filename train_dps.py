import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.measurements import get_operator
from guided_diffusion.condition_methods import get_conditioning_method
from data import get_dataset, get_dataloader, SmallTestDataset
from data.aoa_amp_dataset import AoAAmpDataset  # noqa: F401 registers dataset
from util.logger import get_logger

def load_yaml(file_path: str) -> dict:
    import yaml
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--diffusion_config', type=str, required=True)
    parser.add_argument('--train_config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    train_config = load_yaml(args.train_config)

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Create logger
    logger = get_logger()
    
    # Create model
    extra_keys = {
        'batch_size',
        'learning_rate',
        'num_epochs',
        'save_interval',
        'epoch_save_interval',
        'log_interval',
        'dataset',
        'dataloader',
        'data_channels',
    }
    model_params = {k: v for k, v in model_config.items() if k not in extra_keys}
    model = create_model(**model_params)

    # Adjust input/output layers when working with multi-channel AoA/Amp data
    data_channels = model_config.get('data_channels')
    if data_channels is None:
        data_channels = train_config.get('data_channels')
    if data_channels is None:
        train_dataset_cfg = train_config.get('train_dataset', {})
        if train_dataset_cfg.get('name') == 'aoa_amp':
            num_bs = train_dataset_cfg.get('num_bs', 1)
            num_bs = int(num_bs) if num_bs is not None else 1
            data_channels = 2 * num_bs

    if data_channels is None:
        data_channels = 3
    else:
        data_channels = int(data_channels)

    if data_channels != 3:
        old_input_layer = model.input_blocks[0][0]
        model.input_blocks[0][0] = torch.nn.Conv2d(
            in_channels=data_channels,
            out_channels=old_input_layer.out_channels,
            kernel_size=old_input_layer.kernel_size,
            stride=old_input_layer.stride,
            padding=old_input_layer.padding,
            bias=old_input_layer.bias is not None
        )

        old_output_layer = model.out[-1]
        learn_sigma = model_config.get('learn_sigma', False)
        expected_out_channels = data_channels * 2 if learn_sigma else data_channels
        model.out[-1] = torch.nn.Conv2d(
            in_channels=old_output_layer.in_channels,
            out_channels=expected_out_channels,
            kernel_size=old_output_layer.kernel_size,
            stride=old_output_layer.stride,
            padding=old_output_layer.padding,
            bias=old_output_layer.bias is not None
        )

    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['optimizer']['lr'])
    
    # Create sampler
    sampler = create_sampler(**diffusion_config)
    
    # Create data loader
    train_dataset = get_dataset(**train_config['train_dataset'])
    train_loader = get_dataloader(train_dataset, 
                                 batch_size=train_config['batch_size'],
                                 num_workers=train_config['num_workers'],
                                 train=True)
    
    # Create validation data loader (if specified in config)
    val_loader = None
    if 'val_dataset' in train_config:
        val_dataset = get_dataset(**train_config['val_dataset'])
        val_loader = get_dataloader(val_dataset,
                                   batch_size=train_config['batch_size'],
                                   num_workers=train_config['num_workers'],
                                   train=False)
    
    os.makedirs(train_config['log_dir'], exist_ok=True)
    os.makedirs(train_config['checkpoint_dir'], exist_ok=True)

    # Create TensorBoard writer
    writer = SummaryWriter(train_config['log_dir'])
    
    # Resume from checkpoint (if exists)
    start_epoch = 0
    if train_config.get('resume_checkpoint'):
        checkpoint = torch.load(train_config['resume_checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # 训练循环
    for epoch in range(start_epoch, train_config['epochs']):
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Compute loss
            t = torch.randint(0, sampler.num_timesteps, (data.shape[0],), device=device)
            loss_dict = sampler.training_losses(model, data, t)
            loss = loss_dict
            if isinstance(loss_dict, dict):
                if 'loss' in loss_dict:
                    loss = loss_dict['loss']
                elif 'mse_loss' in loss_dict:
                    loss = loss_dict['mse_loss']
                else:
                    tensor_values = [v for v in loss_dict.values() if torch.is_tensor(v)]
                    if not tensor_values:
                        raise ValueError('No tensor loss returned from sampler.training_losses')
                    loss = tensor_values[0]

            if not torch.is_tensor(loss):
                loss = torch.tensor(loss, device=device)

            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % train_config['log_interval'] == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                writer.add_scalar('Loss/train', loss.item(), 
                                epoch * len(train_loader) + batch_idx)
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        
        # Validation
        if val_loader is not None and (epoch + 1) % train_config['val_interval'] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_data = val_data.to(device)
                    t = torch.randint(0, sampler.num_timesteps, (val_data.shape[0],), device=device)
                    loss_dict = sampler.training_losses(model, val_data, t)
                    loss = loss_dict
                    if isinstance(loss_dict, dict):
                        if 'loss' in loss_dict:
                            loss = loss_dict['loss']
                        elif 'mse_loss' in loss_dict:
                            loss = loss_dict['mse_loss']
                        else:
                            tensor_values = [v for v in loss_dict.values() if torch.is_tensor(v)]
                            if not tensor_values:
                                raise ValueError('No tensor loss returned from sampler.training_losses')
                            loss = tensor_values[0]

                    if not torch.is_tensor(loss):
                        loss = torch.tensor(loss, device=device)

                    if loss.dim() > 0:
                        loss = loss.mean()

                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            logger.info(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % train_config['save_interval'] == 0:
            checkpoint_path = os.path.join(train_config['checkpoint_dir'], 
                                         f'checkpoint_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': val_loss if val_loader is not None else None
            }, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')

if __name__ == '__main__':
    main()
