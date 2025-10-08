"""Conditional sampling script tailored to the AoA/Amplitude building dataset."""

from functools import partial
from collections import defaultdict
import argparse
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import yaml

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from data.aoa_amp_building_dataset import AoAAmpBuildingDataset  # noqa: F401
from util.img_utils import clear_color, mask_generator, normalize_np
from util.logger import get_logger


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


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


def save_aoa_radians(tensor, out_dir: str, base_name: str, aoa_channels: int):
    arr = tensor.detach().cpu().numpy().copy()
    if arr.ndim == 4:
        aoa = arr[:, :aoa_channels] * np.pi
    elif arr.ndim == 3:
        aoa = arr[:aoa_channels] * np.pi
        aoa = aoa[None, ...]
    else:
        return

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{base_name}_aoa_rad.npy"), aoa)


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
    parser.add_argument('--model_config', required=True)
    parser.add_argument('--diffusion_config', required=True)
    parser.add_argument('--task_config', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()

    logger = get_logger()
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    logger.info(f"Device set to {device_str}.")

    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    extra_keys = {
        'batch_size', 'learning_rate', 'num_epochs', 'save_interval',
        'epoch_save_interval', 'log_interval', 'dataset', 'dataloader',
        'data_channels', 'model_path'
    }
    model_params = {k: v for k, v in model_config.items() if k not in extra_keys}

    data_channels = int(model_config.get('data_channels', 6))

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

    measure_cfg = task_config['measurement']
    operator = get_operator(device=device, **measure_cfg['operator'])
    noiser = get_noise(**measure_cfg['noise'])
    logger.info(f"Operation: {measure_cfg['operator']['name']} / Noise: {measure_cfg['noise']['name']}")

    cond_cfg = task_config['conditioning']
    cond_method = get_conditioning_method(cond_cfg['method'], operator, noiser, **cond_cfg['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method: {cond_cfg['method']}")

    diffusion = create_sampler(**diffusion_config)
    sample_fn = partial(
        diffusion.p_sample_loop,
        model=model,
        measurement_cond_fn=measurement_cond_fn,
        record=False,
    )

    out_path = os.path.join(args.save_dir, measure_cfg['operator']['name'])
    for sub in ['input', 'recon', 'label']:
        os.makedirs(os.path.join(out_path, sub), exist_ok=True)

    data_cfg = task_config['data']
    data_kwargs = data_cfg.copy()
    dataset_name = data_kwargs.pop('name')
    root = data_kwargs.pop('root')
    data_kwargs.pop('num_samples', None)

    if dataset_name != 'aoa_amp_building':
        raise ValueError('sample_condition_building.py is intended for aoa_amp_building dataset only.')

    dataset = get_dataset(dataset_name, root, **data_kwargs)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=True)

    if measure_cfg['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(**measure_cfg['mask_opt'])

    nmse_totals = []
    channel_nmse_records = defaultdict(list)

    aoa_channels = 3  # first three channels are AoA
    channel_ranges = [(-np.pi, np.pi)] * aoa_channels + [(-1.0, 1.0)] * (data_channels - aoa_channels)
    channel_scales = [np.pi] * aoa_channels + [1.0] * (data_channels - aoa_channels)
    channel_cmaps = ['hsv'] * aoa_channels + ['hsv'] * (data_channels - aoa_channels)

    for idx, ref_img in enumerate(loader):
        logger.info(f"Inference for image {idx}")
        fname_base = str(idx).zfill(5)
        ref_img = ref_img.to(device)

        if measure_cfg['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else:
            y = operator.forward(ref_img)
            y_n = noiser(y)

        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        total_nmse, per_channel_nmse = compute_nmse(ref_img, sample)
        nmse_totals.append(total_nmse)
        for c_idx, value in enumerate(per_channel_nmse):
            channel_nmse_records[c_idx].append(value)

        nmse_msg = ", ".join(
            f"ch{c_idx + 1}: {value:.4e}" for c_idx, value in enumerate(per_channel_nmse)
        ) if per_channel_nmse else ""
        logger.info(f"NMSE (image {idx}): total {total_nmse:.4e} {nmse_msg}")

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
        save_tensor_npy(ref_img, os.path.join(out_path, 'label', f'{fname_base}.npy'))
        save_tensor_npy(sample, os.path.join(out_path, 'recon', f'{fname_base}.npy'))

        save_aoa_radians(y_n, os.path.join(out_path, 'input'), fname_base, aoa_channels)
        save_aoa_radians(ref_img, os.path.join(out_path, 'label'), fname_base, aoa_channels)
        save_aoa_radians(sample, os.path.join(out_path, 'recon'), fname_base, aoa_channels)

    if nmse_totals:
        avg_total_nmse = sum(nmse_totals) / len(nmse_totals)
        logger.info(f"Average NMSE over {len(nmse_totals)} samples: {avg_total_nmse:.4e}")
        for c_idx, values in channel_nmse_records.items():
            avg_channel_nmse = sum(values) / len(values)
            logger.info(f"Average NMSE channel {c_idx + 1}: {avg_channel_nmse:.4e}")


if __name__ == '__main__':
    main()
