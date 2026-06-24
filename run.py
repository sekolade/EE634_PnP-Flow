#!/usr/bin/env python3
"""
Inference-only PnP-Flow runner. There is no training, configuration file,
dataset download, model download, or demo notebook.

Every algorithm/runtime parameter is supplied explicitly on the command line.

OPTIONS
-------
--dataset: celeba | afhq_cat
--problem: denoising | gaussian_deblurring_FFT | superresolution |
           random_inpainting | inpainting
--noise-type: gaussian | laplace
--gamma-style: 1_minus_t | sqrt_1_minus_t | constant | alpha_1_minus_t

Problem-specific required options
---------------------------------
gaussian_deblurring_FFT: --sigma-blur --kernel-size
a superresolution run:   --superresolution-factor
random_inpainting:       --missing-ratio
inpainting:              --box-half-size
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from pnpflow.algorithm import PnPFlow
from pnpflow.degradations import (
    BoxInpainting,
    Denoising,
    GaussianDeblurring,
    RandomInpainting,
    Superresolution,
)
from pnpflow.io import make_loader, save_batch_outputs
from pnpflow.utils import (
    compute_lpips,
    compute_psnr,
    compute_ssim,
    define_model,
    load_model,
    save_run_configuration,
)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--dataset", required=True, choices=["celeba", "afhq_cat"])
    parser.add_argument("--problem", required=True, choices=[
        "denoising", "gaussian_deblurring_FFT", "superresolution",
        "random_inpainting", "inpainting",
    ])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", required=True, type=int)
    parser.add_argument("--num-channels", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--max-images", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--noise-type", required=True, choices=["gaussian", "laplace"])
    parser.add_argument("--sigma-noise", required=True, type=float)
    parser.add_argument("--steps-pnp", required=True, type=int)
    parser.add_argument("--lr-pnp", required=True, type=float)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--gamma-style", required=True, choices=[
        "1_minus_t", "sqrt_1_minus_t", "constant", "alpha_1_minus_t",
    ])

    parser.add_argument("--sigma-blur", type=float)
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--superresolution-factor", type=int)
    parser.add_argument("--missing-ratio", type=float)
    parser.add_argument("--box-half-size", type=int)
    return parser.parse_args()


def require(value, option, problem):
    if value is None:
        raise ValueError(f"{option} is required for problem={problem}")
    return value


def build_degradation(args, device):
    if args.problem == "denoising":
        return Denoising()
    if args.problem == "gaussian_deblurring_FFT":
        return GaussianDeblurring(
            require(args.sigma_blur, "--sigma-blur", args.problem),
            require(args.kernel_size, "--kernel-size", args.problem),
            "fft", args.num_channels, args.image_size, device,
        )
    if args.problem == "superresolution":
        return Superresolution(
            require(args.superresolution_factor,
                    "--superresolution-factor", args.problem),
            args.image_size,
            device=device,
        )
    if args.problem == "random_inpainting":
        return RandomInpainting(
            require(args.missing_ratio, "--missing-ratio", args.problem)
        )
    if args.problem == "inpainting":
        return BoxInpainting(
            require(args.box_half_size, "--box-half-size", args.problem)
        )
    raise ValueError(args.problem)


def main():
    args = parse_args()
    if args.num_channels != 3:
        raise ValueError("The supplied CelebA and AFHQ-Cat checkpoints use 3 channels")
    if args.max_images < 1 or args.batch_size < 1:
        raise ValueError("--max-images and --batch-size must be positive")

    if torch.cuda.is_available():
    	device = torch.device("cuda")
    elif torch.backends.mps.is_available():
    	device = torch.device("mps")
    else:
    	device = torch.device("cpu")

    print("Using device:", device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    configuration = vars(args).copy()
    configuration["device"] = str(device)
    save_run_configuration(output_dir / "run_config.json", configuration)

    model = define_model(args.num_channels, args.image_size)
    model = load_model(model, checkpoint, device)
    degradation = build_degradation(args, device)
    solver = PnPFlow(
        model=model,
        device=device,
        noise_type=args.noise_type,
        lr_pnp=args.lr_pnp,
        alpha=args.alpha,
        num_samples=args.num_samples,
        steps_pnp=args.steps_pnp,
        gamma_style=args.gamma_style,
    )

    loader = make_loader(args.input_dir, args.image_size, args.batch_size)
    metric_rows = []
    processed = 0

    for batch_index, (clean_img, names) in enumerate(loader):
        if processed >= args.max_images:
            break
        remaining = args.max_images - processed
        clean_img = clean_img[:remaining]
        names = list(names[:remaining])
        clean_device = clean_img.to(device)

        noisy_img, rec_img, snapshots = solver.reconstruct(
            clean_device, degradation, args.sigma_noise,
            seed=args.seed + batch_index,
        )

        # Metrics are computed only once, on the final reconstruction.
        psnr_rec, psnr_noisy = compute_psnr(
            clean_img, noisy_img, rec_img,
            args.dataset, args.problem, degradation.H_adj,
        )
        ssim_rec, ssim_noisy = compute_ssim(
            clean_img, noisy_img, rec_img,
            args.dataset, args.problem, degradation.H_adj,
        )
        lpips_rec, lpips_noisy = compute_lpips(
            clean_img, noisy_img, rec_img,
            args.dataset, args.problem, degradation.H_adj,
        )

        save_batch_outputs(
            output_dir, names, clean_img, noisy_img, rec_img, snapshots,
            degradation, args.dataset, args.problem,
        )

        metric_rows.append({
            "batch": batch_index,
            "num_images": len(names),
            "psnr_reconstruction": psnr_rec,
            "psnr_noisy": psnr_noisy,
            "ssim_reconstruction": ssim_rec,
            "ssim_noisy": ssim_noisy,
            "lpips_reconstruction": lpips_rec,
            "lpips_noisy": lpips_noisy,
        })
        processed += len(names)
        print(f"Processed {processed}/{min(args.max_images, len(loader.dataset))} images")

    if not metric_rows:
        raise RuntimeError("No images were processed")

    fieldnames = list(metric_rows[0].keys())
    with open(output_dir / "metrics_by_batch.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    total = sum(row["num_images"] for row in metric_rows)
    summary = {"processed_images": total}
    for key in fieldnames[2:]:
        summary[key] = sum(
            row[key] * row["num_images"] for row in metric_rows
        ) / total
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("Completed. Results:", output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
