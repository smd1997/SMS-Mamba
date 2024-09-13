import torch.nn as nn
import torch
import numpy as np
from functools import lru_cache
from typing import Union, Tuple, List
from scipy.ndimage import gaussian_filter
import time
import math

SEED = 1958
n_stages = 5
kwargs = {
        'T': 20,
        'hw': 256,
        'N': 2,
        'd_model': 512,
        'input_channels': 32,
        'UNet_base_num_features': 32,
        'UNet_max_num_features': 1024,
        'n_stages': n_stages,
        'kernel_sizes': 3,
        'strides': [2 if i>0 else 1 for i in range(n_stages)],
        'num_output_channels': 1,
        'conv_bias': True,
        'norm_op': nn.InstanceNorm2d,
        'conv_op': nn.Conv2d,
        'n_conv_per_stage': 2,
        'n_conv_per_stage_decoder': 2,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 
        'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 
        'nonlin_kwargs': {'inplace': True},
        'deep_supervision': True
}

d_model = 32
n_stages_vssm = 4
kwargs_vssm = {
    'input_channels': 1,
    'patch_size': 4,
    'd_model': d_model,
    'UNet_base_num_features': d_model,
    'UNet_max_num_features': 1024,
    'n_stages': n_stages_vssm,
    'kernel_sizes': 3,
    'strides': [2 if i>0 else 1 for i in range(n_stages_vssm)],
    'num_output_channels': 1,
    'conv_op': nn.Conv2d,
    'n_conv_per_stage': 2,
    'n_conv_per_stage_decoder': 2,
    'deep_supervision': True
}

d_model1 = 32
n_stages_vssm_unrolled = 5
kwargs_vssm_unrolled = {
    'input_channels': 1,
    'patch_size': 4,
    'd_model': d_model1,
    'UNet_base_num_features': d_model1,
    'UNet_max_num_features': 1024,
    'n_stages': n_stages_vssm_unrolled,
    'kernel_sizes': 3,
    'strides': [2 if i>0 else 1 for i in range(n_stages_vssm_unrolled)],
    'padding': 1,
    'num_output_channels': 1,
    'conv_op': nn.Conv2d,
    'n_conv_per_stage': 2,
    'n_conv_per_stage_decoder': 2,
    'deep_supervision': True
}

d_model = 32
n_stages = 4
kernel_sizes = 3
conv_op = nn.Conv3d
if conv_op == nn.Conv3d:
    padding = [(1, (kernel_sizes-1)//2, (kernel_sizes-1)//2) for _ in range(n_stages)]
    stride = [(1, 2, 2) if i>0 else (1, 1, 1) for i in range(n_stages)]
else:
    padding = [(kernel_sizes-1)//2 for _ in range(n_stages)]
    stride = [2 if i>0 else 1 for i in range(n_stages)]
    
kwargs_vssm_unrolled_LS = {
    'input_channels': 2,
    'patch_size': 4,
    'd_model': d_model,
    'UNet_base_num_features': d_model,
    'UNet_max_num_features': 1024,
    'n_stages': n_stages,
    'kernel_sizes': kernel_sizes,
    'strides': stride,
    'padding': padding,
    'num_output_channels': 2,
    'conv_op': nn.Conv3d,
    'n_conv_per_stage': 2,
    'n_conv_per_stage_decoder': 2,
    'deep_supervision': True
}

n_stages = 5
input_channels = 2
d_model = 32
conv_op = nn.Conv3d
if conv_op == nn.Conv3d:
    norm_op = nn.InstanceNorm3d
    padding = [(1, (kernel_sizes-1)//2, (kernel_sizes-1)//2) if i>0 else (1, 1, 1) for i in range(n_stages)]
    stride = [(1, 2, 2) if i>0 else (1, 1, 1) for i in range(n_stages)]
else:
    norm_op = nn.InstanceNorm2d
    padding = [(kernel_sizes-1)//2 if i>0 else 1 for i in range(n_stages)]
    stride = [2 if i>0 else 1 for i in range(n_stages)]
    
kwargs_unet_unrolled_LS = {
        'input_channels': input_channels,
        'UNet_base_num_features': d_model,
        'UNet_max_num_features': 1024,
        'n_stages': n_stages,
        'kernel_sizes': 3,
        'strides': stride,
        'num_output_channels': 1,
        'conv_bias': True,
        'norm_op': norm_op,
        'conv_op': conv_op,
        'n_conv_per_stage': 2,
        'n_conv_per_stage_decoder': 2,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 
        'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 
        'nonlin_kwargs': {'inplace': True},
        'deep_supervision': True
}

conv_op = nn.Conv2d
if conv_op == nn.Conv3d:
    norm_op = nn.InstanceNorm3d
    padding = [(1, (kernel_sizes-1)//2, (kernel_sizes-1)//2) if i>0 else (1, 1, 1) for i in range(n_stages)]
    stride = [(1, 2, 2) if i>0 else (1, 1, 1) for i in range(n_stages)]
else:
    norm_op = nn.InstanceNorm2d
    padding = [(kernel_sizes-1)//2 if i>0 else 1 for i in range(n_stages)]
    stride = [2 if i>0 else 1 for i in range(n_stages)]

kwargs_unet_unrolled = {
        'input_channels': input_channels,
        'UNet_base_num_features': d_model,
        'UNet_max_num_features': 1024,
        'n_stages': n_stages,
        'kernel_sizes': 3,
        'strides': stride,
        'num_output_channels': 1,
        'conv_bias': True,
        'norm_op': norm_op,
        'conv_op': conv_op,
        'n_conv_per_stage': 2,
        'n_conv_per_stage_decoder': 2,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 
        'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 
        'nonlin_kwargs': {'inplace': True},
        'deep_supervision': True
}

kwargs_unet = {
        'input_channels': input_channels,
        'UNet_base_num_features': d_model,
        'UNet_max_num_features': 1024,
        'n_stages': n_stages,
        'kernel_sizes': 3,
        'strides': stride,
        'num_output_channels': 1,
        'conv_bias': True,
        'norm_op': norm_op,
        'conv_op': conv_op,
        'n_conv_per_stage': 2,
        'n_conv_per_stage_decoder': 2,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 
        'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 
        'nonlin_kwargs': {'inplace': True},
        'deep_supervision': True
}

kwargs_plainunet = {
        'input_channels': input_channels,
        'features_per_stage': d_model,
        'n_stages': n_stages,
        'kernel_sizes': 3,
        'strides': stride,
        'num_classes': input_channels,
        'conv_bias': True,
        'norm_op': norm_op,
        'conv_op': conv_op,
        'n_conv_per_stage': 2,
        'n_conv_per_stage_decoder': 2,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 
        'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 
        'nonlin_kwargs': {'inplace': True},
        'deep_supervision': False
}

def collate(data):
    return data

def collate_batch(data):
    uk = []
    label = []
    mask = []
    for list in data:
        uk.append(list[0])
        label.append(list[1])
        mask.append(list[2])
    return [torch.stack(uk), torch.stack(label), torch.stack(mask)]

def get_time():
    return time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())

def print_to_log_file(log_file, *args, also_print_to_console=True):
    with open(log_file, 'a+') as f:
        for a in args:
            f.write(str(a))
            f.write(" ")
        f.write("\n")
    if also_print_to_console:
        print(*args)

def adjust_learning_rate(opt, epo, max_steps, initial_lr):
    exponent = 0.9
    new_lr = initial_lr * (1 - epo / max_steps) ** exponent
    for param_group in opt.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def fft(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-4, -3)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.fftn(data, dim=dim, norm=norm)
    if shift:
        data = torch.fft.fftshift(data, dim=dim)
    return data

def ifft(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-4, -3)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.ifftn(data, dim=dim, norm=norm)
    if shift:
        data = torch.fft.fftshift(data, dim=dim)
    return data

def fft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data

def ifft2c(data: torch.Tensor, norm: str = "ortho", shift = True, dim = (-2, -1)) -> torch.Tensor:
    if shift:
        data = torch.fft.ifftshift(data, dim=(dim[0]-1, dim[1]-1))
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=dim, norm=norm
        )
    )
    if shift:
        data = torch.fft.fftshift(data, dim=(dim[0]-1, dim[1]-1))
    return data

@lru_cache(maxsize=2)
def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float32, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.type(dtype).to(device)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

if __name__ == "__main__":
    gaussian_mask = compute_gaussian((128,128))
    pass