
from pathlib import Path
import json
import math
import os

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as v2
from ignite.metrics import SSIM
from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR

from pnpflow.models import UNet

_LPIPS_ALEX_MODEL = None


def define_model(num_channels, dim_image):
    # Exact OT UNet construction used by the original define_model(args).
    return UNet(
        input_channels=num_channels,
        input_height=dim_image,
        ch=32,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=6,
        attn_resolutions=(16, 8),
        resamp_with_conv=True,
    )


def load_model(model, checkpoint_path, device):
    # Exact OT checkpoint loading behavior, with eval/frozen parameters added
    # for inference-only use.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def gaussian_2d_kernel(sigma, size):
    """Generate a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
    y = torch.arange(-size // 2 + 1.0, size // 2 + 1.0)
    x, y = torch.meshgrid(x, y, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def upsample(x, sf):
    st = 0
    z = torch.zeros(
        (x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)
    ).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf):
    st = 0
    return x[..., st::sf, st::sf]


def square_mask(x, half_size_mask):
    d = x.shape[2] // 2
    mask = torch.ones_like(x)
    mask[:, :, d-half_size_mask:d+half_size_mask,
         d-half_size_mask:d+half_size_mask] = 0
    return mask * x


def random_mask(x, p, seed=None):
    # Exact deterministic masking behavior used in the original repository.
    np.random.seed(42)
    mask = torch.from_numpy(
        np.random.binomial(
            n=1,
            p=1-p,
            size=(x.shape[0], x.shape[2], x.shape[3]),
        )
    ).to(x.device)
    return mask.unsqueeze(1) * x


def create_downsampling_matrix(H, W, sf, device):
    # Retained because the original Superresolution constructor creates it.
    assert H % sf == 0 and W % sf == 0
    H_ds, W_ds = H // sf, W // sf
    matrix = torch.zeros((H_ds * W_ds, H * W), device=device)
    for i in range(H_ds):
        for j in range(W_ds):
            matrix[i * W_ds + j, i * sf * W + j * sf] = 1
    return matrix


def postprocess(img, dataset):
    # Exact OT postprocessing branch from the original repository.
    if dataset == "afhq_cat":
        img = (img + 1) / 2
    else:
        inv_transform = v2.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1.0 / 0.5, 1.0 / 0.5, 1.0 / 0.5],
        )
        img = inv_transform(img)
    return img


def _metric_images(clean_img, noisy_img, rec_img, dataset, problem, H_adj):
    clean = postprocess(clean_img.clone(), dataset)
    noisy = postprocess(noisy_img.clone(), dataset)
    rec = postprocess(rec_img.clone(), dataset)
    H_adj_noisy = postprocess(H_adj(noisy_img), dataset)
    if problem == "superresolution":
        noisy = H_adj_noisy
    return clean, noisy, rec


def compute_psnr(clean_img, noisy_img, rec_img, dataset, problem, H_adj):
    # Same PSNR function and dimensions as original pnpflow/utils.py.
    clean, noisy, rec = _metric_images(
        clean_img, noisy_img, rec_img, dataset, problem, H_adj
    )
    clean = clean.permute(0, 2, 3, 1).cpu().data
    noisy = noisy.permute(0, 2, 3, 1).cpu().data
    rec = rec.permute(0, 2, 3, 1).cpu().data
    psnr_rec = PSNR(rec, clean, data_range=1.0, dim=(1, 2, 3))
    psnr_noisy = PSNR(noisy, clean, data_range=1.0, dim=(1, 2, 3))
    return float(psnr_rec), float(psnr_noisy)


def compute_ssim(clean_img, noisy_img, rec_img, dataset, problem, H_adj):
    # Same Ignite SSIM implementation as original pnpflow/utils.py.
    clean, noisy, rec = _metric_images(
        clean_img, noisy_img, rec_img, dataset, problem, H_adj
    )
    clean = clean.cpu()
    noisy = noisy.cpu()
    rec = rec.cpu()
    ssim_metric = SSIM(data_range=1.0)
    ssim_metric_noisy = SSIM(data_range=1.0)
    ssim_metric.update((rec, clean))
    ssim_rec = ssim_metric.compute()
    ssim_metric_noisy.update((noisy, clean))
    ssim_noisy = ssim_metric_noisy.compute()
    return float(ssim_rec), float(ssim_noisy)


def compute_lpips(clean_img, noisy_img, rec_img, dataset, problem, H_adj):
    # Same LPIPS(AlexNet) metric path as original, cached once per process.
    global _LPIPS_ALEX_MODEL
    if torch.cuda.is_available():
    	device = torch.device("cuda")
    elif torch.backends.mps.is_available():
    	device = torch.device("mps")
    else:
    	device = torch.device("cpu")
    if _LPIPS_ALEX_MODEL is None:
        print("Initializing LPIPS AlexNet model once...")
        _LPIPS_ALEX_MODEL = lpips.LPIPS(net="alex").to(device)
        _LPIPS_ALEX_MODEL.eval()
        for parameter in _LPIPS_ALEX_MODEL.parameters():
            parameter.requires_grad_(False)

    clean, noisy, rec = _metric_images(
        clean_img, noisy_img, rec_img, dataset, problem, H_adj
    )
    clean = (2 * clean - 1).to(device)
    noisy = (2 * noisy - 1).to(device)
    rec = (2 * rec - 1).to(device)

    with torch.inference_mode():
        lpips_rec = _LPIPS_ALEX_MODEL(
            clean, rec, normalize=True
        ).mean().item()
        lpips_noisy = _LPIPS_ALEX_MODEL(
            clean, noisy, normalize=True
        ).mean().item()
    return lpips_rec, lpips_noisy


def save_run_configuration(path, configuration):
    Path(path).write_text(json.dumps(configuration, indent=2))
