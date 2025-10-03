#test
from functools import partial
from collections import defaultdict
import os
import argparse
import yaml
from typing import Optional, Sequence, Tuple

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from data.aoa_amp_dataset import AoAAmpDataset  # noqa: F401 register aoa dataset
from util.img_utils import clear_color, mask_generator, normalize_np
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_tensor_channels(
    tensor,
    out_dir: str,
    base_name: str,
    cmap: str = 'viridis',
    normalize: bool = True,
    channel_multipliers: Optional[Sequence[float]] = None,
    channel_value_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    channel_cmaps: Optional[Sequence[str]] = None,
):
    data = tensor.detach().cpu().clone()
    if data.ndim == 4:
        data = data.squeeze(0)
    if data.ndim == 2:
        data = data.unsqueeze(0)

    os.makedirs(out_dir, exist_ok=True)
    for idx in range(data.shape[0]):
        channel = data[idx].numpy()

        if channel_multipliers is not None and idx < len(channel_multipliers):
            channel = channel * channel_multipliers[idx]

        channel_name = os.path.join(out_dir, f"{base_name}_channel{idx + 1}.png")
        current_cmap = cmap
        if channel_cmaps is not None and idx < len(channel_cmaps) and channel_cmaps[idx]:
            current_cmap = channel_cmaps[idx]

        if normalize:
            img = normalize_np(channel)
            plt.imsave(channel_name, img, cmap=current_cmap)
        else:
            kwargs = {}
            if channel_value_ranges is not None and idx < len(channel_value_ranges):
                vmin, vmax = channel_value_ranges[idx]
                kwargs['vmin'] = vmin
                kwargs['vmax'] = vmax
            plt.imsave(channel_name, channel, cmap=current_cmap, **kwargs)


def save_tensor_npy(tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, tensor.detach().cpu().numpy())

def save_aoa_radians(tensor, out_dir: str, base_name: str):
    if tensor.ndim < 3:
        return
    arr = tensor.detach().cpu().numpy().copy()

    if arr.ndim == 4:
        if arr.shape[1] < 1:
            return
        aoa = arr[:, 0] * np.pi
    elif arr.ndim == 3:
        aoa = arr[0] * np.pi
        aoa = aoa[None, ...]
    else:
        return

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{base_name}_aoa_rad.npy"), aoa)



def save_tensor_channels_denormalized(tensor, out_dir: str, base_name: str):
    """
    Save AoA/Amplitude tensor with proper denormalization and colorbars
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import hsv_to_rgb
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Convert tensor to numpy and move to CPU
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu()
    data = tensor.numpy()
    
    # Handle batch dimension
    if len(data.shape) == 4:  # [B, C, H, W]
        data = data[0]  # Take first batch item
    
    # Extract channels
    aoa_normalized = data[0]  # Normalized AoA [-1, 1]
    amp_normalized = data[1]  # Normalized amplitude [-1, 1]
    
    # Denormalize to original ranges
    aoa_original = aoa_normalized * np.pi  # [-1,1] → [-π,π] radians
    amp_original = amp_normalized  # Keep normalized for now, or apply your specific denormalization
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Determine spatial extent (assuming 100m x 100m grid)
    extent = [-50, 50, -50, 50]
    
    # 1. AoA Map
    im1 = axes[0].imshow(aoa_original, extent=extent, origin='lower', 
                         cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0].set_title('Angle of Arrival (AoA) Map')
    axes[0].set_xlabel('X position (m)')
    axes[0].set_ylabel('Y position (m)')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('AoA (radians)')
    
    # 2. Amplitude Map  
    im2 = axes[1].imshow(amp_normalized, extent=extent, origin='lower', 
                         cmap='viridis', vmin=-1, vmax=1)
    axes[1].set_title('Amplitude Map (Normalized)')
    axes[1].set_xlabel('X position (m)')
    axes[1].set_ylabel('Y position (m)')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Normalized Amplitude')
    
    # 3. Combined View (AoA as hue, amplitude as brightness)
    aoa_norm_01 = (aoa_original + np.pi) / (2 * np.pi)  # [0, 1]
    amp_norm_01 = (amp_normalized + 1) / 2  # [-1, 1] → [0, 1]
    
    hsv_img = np.zeros((*aoa_normalized.shape, 3))
    hsv_img[:, :, 0] = aoa_norm_01  # Hue = AoA
    hsv_img[:, :, 1] = 1.0          # Full saturation
    hsv_img[:, :, 2] = amp_norm_01  # Value = amplitude
    
    rgb_img = hsv_to_rgb(hsv_img)
    
    axes[2].imshow(rgb_img, extent=extent, origin='lower')
    axes[2].set_title('Combined View\n(Color=AoA, Brightness=Amplitude)')
    axes[2].set_xlabel('X position (m)')
    axes[2].set_ylabel('Y position (m)')
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(out_dir, f'{base_name}_denormalized.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save individual channels as separate files
    # AoA only
    fig_aoa, ax_aoa = plt.subplots(figsize=(8, 6))
    im_aoa = ax_aoa.imshow(aoa_original, extent=extent, origin='lower', 
                           cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax_aoa.set_title('Angle of Arrival (AoA)')
    ax_aoa.set_xlabel('X position (m)')
    ax_aoa.set_ylabel('Y position (m)')
    cbar_aoa = plt.colorbar(im_aoa, ax=ax_aoa)
    cbar_aoa.set_label('AoA (radians)')
    plt.savefig(os.path.join(out_dir, f'{base_name}_aoa.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Amplitude only
    fig_amp, ax_amp = plt.subplots(figsize=(8, 6))
    im_amp = ax_amp.imshow(amp_normalized, extent=extent, origin='lower', 
                           cmap='viridis', vmin=-1, vmax=1)
    ax_amp.set_title('Amplitude (Normalized)')
    ax_amp.set_xlabel('X position (m)')
    ax_amp.set_ylabel('Y position (m)')
    cbar_amp = plt.colorbar(im_amp, ax=ax_amp)
    cbar_amp.set_label('Normalized Amplitude')
    plt.savefig(os.path.join(out_dir, f'{base_name}_amplitude.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_tensor_npy_denormalized(tensor, path: str):
    """
    Save tensor with denormalized values
    """
    import numpy as np
    
    # Convert tensor to numpy
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu()
    data = tensor.numpy()
    
    # Handle batch dimension
    if len(data.shape) == 4:
        data = data[0]
    
    # Denormalize AoA channel
    if data.shape[0] >= 2:
        data[0] = data[0] * np.pi  # AoA: [-1,1] → [-π,π]
        # Amplitude stays normalized for now
    
    np.save(path, data)


def compute_nmse(reference: torch.Tensor, estimate: torch.Tensor):
    ref = reference.detach()
    est = estimate.detach()

    mse = torch.mean((est - ref) ** 2)
    denom = torch.mean(ref ** 2)
    total_nmse = (mse / (denom + 1e-8)).item()

    per_channel = []
    if ref.dim() == 4:
        for channel in range(ref.shape[1]):
            ref_c = ref[:, channel]
            est_c = est[:, channel]
            mse_c = torch.mean((est_c - ref_c) ** 2)
            denom_c = torch.mean(ref_c ** 2)
            per_channel.append((mse_c / (denom_c + 1e-8)).item())

    return total_nmse, per_channel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # Prepare model parameters
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
        'model_path',
    }
    model_params = {k: v for k, v in model_config.items() if k not in extra_keys}

    # Determine data channels
    data_channels = model_config.get('data_channels')
    if data_channels is None:
        data_cfg = task_config.get('data', {})
        if data_cfg.get('name') == 'aoa_amp':
            num_bs = data_cfg.get('num_bs', 1)
            num_bs = int(num_bs) if num_bs is not None else 1
            data_channels = 2 * num_bs

    if data_channels is None:
        data_channels = 3
    else:
        data_channels = int(data_channels)

    # Load model and adapt I/O layers for custom channel counts
    model = create_model(**model_params)
    if data_channels != 3:
        old_in = model.input_blocks[0][0]
        model.input_blocks[0][0] = torch.nn.Conv2d(
            in_channels=data_channels,
            out_channels=old_in.out_channels,
            kernel_size=old_in.kernel_size,
            stride=old_in.stride,
            padding=old_in.padding,
            bias=old_in.bias is not None,
        )

        old_out = model.out[-1]
        learn_sigma = model_config.get('learn_sigma', False)
        expected_out = data_channels * 2 if learn_sigma else data_channels
        model.out[-1] = torch.nn.Conv2d(
            in_channels=old_out.in_channels,
            out_channels=expected_out,
            kernel_size=old_out.kernel_size,
            stride=old_out.stride,
            padding=old_out.padding,
            bias=old_out.bias is not None,
        )

    model = model.to(device)

    model_path = model_config.get('model_path')
    if model_path:
        logger.info(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn, record=False)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    data_config = task_config['data']
    data_kwargs = data_config.copy()
    dataset_name = data_kwargs.pop('name')
    root = data_kwargs.pop('root')

    transform = None
    is_aoa_dataset = dataset_name == 'aoa_amp'
    if not is_aoa_dataset:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = get_dataset(dataset_name, root, transforms=transform, **data_kwargs)
    else:
        dataset = get_dataset(dataset_name, root, **data_kwargs)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do Inference
    nmse_totals = []
    channel_nmse_records = defaultdict(list)

    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname_base = str(i).zfill(5)
        fname = fname_base + '.png'
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        total_nmse, per_channel_nmse = compute_nmse(ref_img, sample)
        nmse_totals.append(total_nmse)
        for idx, value in enumerate(per_channel_nmse):
            channel_nmse_records[idx].append(value)

        if per_channel_nmse:
            nmse_msg = ", ".join(
                f"ch{idx + 1}: {score:.4e}" for idx, score in enumerate(per_channel_nmse)
            )
            logger.info(
                f"NMSE (image {i}): total {total_nmse:.4e} ({nmse_msg})"
            )
        else:
            logger.info(f"NMSE (image {i}): total {total_nmse:.4e}")

        if is_aoa_dataset:
            channel_ranges = [(-np.pi, np.pi), (-1.0, 1.0)]
            channel_scales = [np.pi, 1.0]
            channel_cmaps = ['hsv', 'hsv']

            save_tensor_channels(
                y_n,
                os.path.join(out_path, 'input'),
                fname_base,
                cmap='viridis',
                normalize=False,
                channel_multipliers=channel_scales,
                channel_value_ranges=channel_ranges,
                channel_cmaps=channel_cmaps,
            )
            save_tensor_channels(
                ref_img,
                os.path.join(out_path, 'label'),
                fname_base,
                cmap='viridis',
                normalize=False,
                channel_multipliers=channel_scales,
                channel_value_ranges=channel_ranges,
                channel_cmaps=channel_cmaps,
            )
            save_tensor_channels(
                sample,
                os.path.join(out_path, 'recon'),
                fname_base,
                cmap='viridis',
                normalize=False,
                channel_multipliers=channel_scales,
                channel_value_ranges=channel_ranges,
                channel_cmaps=channel_cmaps,
            )

            save_tensor_npy(y_n, os.path.join(out_path, 'input', f'{fname_base}.npy'))
            # save_tensor_npy(ref_img, os.path.join(out_path, 'label', f'{fname_base}.npy'))
            save_tensor_npy_denormalized(ref_img, os.path.join(out_path, 'label', f'{fname}_label.npy'))
            save_tensor_npy(sample, os.path.join(out_path, 'recon', f'{fname_base}.npy'))

            save_aoa_radians(y_n, os.path.join(out_path, 'input'), fname_base)
            save_aoa_radians(ref_img, os.path.join(out_path, 'label'), fname_base)
            save_aoa_radians(sample, os.path.join(out_path, 'recon'), fname_base)
        else:
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
            plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
            plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

    if nmse_totals:
        avg_total_nmse = sum(nmse_totals) / len(nmse_totals)
        logger.info(f"Average NMSE over {len(nmse_totals)} samples: {avg_total_nmse:.4e}")
        for idx, values in channel_nmse_records.items():
            avg_channel_nmse = sum(values) / len(values)
            logger.info(f"Average NMSE channel {idx + 1}: {avg_channel_nmse:.4e}")

if __name__ == '__main__':
    main()
