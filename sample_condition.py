#test
from functools import partial
import os
import argparse
import yaml

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


def save_tensor_channels(tensor, out_dir: str, base_name: str, cmap: str = 'viridis'):
    data = tensor.detach().cpu()
    if data.ndim == 4:
        data = data.squeeze(0)
    if data.ndim == 2:
        data = data.unsqueeze(0)

    os.makedirs(out_dir, exist_ok=True)
    for idx in range(data.shape[0]):
        channel = data[idx].numpy()
        img = normalize_np(channel)
        channel_name = os.path.join(out_dir, f"{base_name}_channel{idx + 1}.png")
        plt.imsave(channel_name, img, cmap=cmap)


def save_tensor_npy(tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, tensor.detach().cpu().numpy())


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
        if is_aoa_dataset:
            save_tensor_channels(y_n, os.path.join(out_path, 'input'), fname_base)
            save_tensor_channels(ref_img, os.path.join(out_path, 'label'), fname_base)
            save_tensor_channels(sample, os.path.join(out_path, 'recon'), fname_base)

            save_tensor_npy(y_n, os.path.join(out_path, 'input', f'{fname_base}.npy'))
            save_tensor_npy(ref_img, os.path.join(out_path, 'label', f'{fname_base}.npy'))
            save_tensor_npy(sample, os.path.join(out_path, 'recon', f'{fname_base}.npy'))
        else:
            plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
            plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
            plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

if __name__ == '__main__':
    main()
