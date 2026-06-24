# PnP-Flow

## 1. Introduction

PnP-Flow is a plug-and-play image reconstruction method currently designed for linear inverse problems of the form

\[
y = Hx + n,
\]

where \(y\) is the noisy observation, \(x\) is the unknown clean image, \(H\) is the degradation operator, and \(n\) is the measurement noise, which is Gaussian in the experiments considered in the paper.

By formulating the reconstruction problem as a maximum a posteriori (MAP) estimation problem, while treating \(x\) and \(y\) as random variables, and then applying proximal splitting, the formulation can be transformed into Plug-and-Play Forward-Backward Splitting (PnP-FBS). The corresponding formulas and derivation are provided in `report.pdf`. In PnP-FBS, a denoiser is used in place of the proximal operator associated with the image-prior term.

The denoiser used in this PnP-FBS method is obtained from a flow-matching model. Two separate flow-matching models are trained for the two datasets used in the paper: CelebA and AFHQ-Cat. The flow-matching model is specifically designed to produce paths that are as close as possible to straight lines from samples of a noise distribution to samples of the clean-data distribution. The model is time-dependent, meaning that the network also receives the current continuous time value \(t\) as an input. The flow-matching background and the methods used to encourage straight-line flow matching are explained in `report.pdf`.

Straight flow paths make it possible to move from the noise distribution toward the clean-data distribution using fewer numerical integration steps. A denoiser is then defined as a deterministic function of the learned straight-line flow-matching model.

At each iteration, the algorithm first applies a gradient step to enforce consistency with the observed measurement. If a denoising operation were applied immediately after this gradient step, the resulting method would correspond to a regular PnP-FBS algorithm. However, because the denoiser in PnP-Flow is designed for points located on a time-dependent straight flow path between the noise and data distributions, an additional reprojection step is introduced. This step projects the result of the gradient update onto the flow path corresponding to the current time. The algorithm then applies the time-dependent denoising step using the learned straight-line flow-matching model.

This project provides all inverse problems and datasets evaluated in the paper. It includes five inverse imaging problems:

- denoising,
- Gaussian deblurring,
- super-resolution,
- random inpainting,
- box inpainting,

and two datasets:

- CelebA,
- AFHQ-Cat.

There are two learned denoisers, one for each dataset. The reconstructed images are evaluated using PSNR, SSIM, and LPIPS in the same way as in the paper. The inverse-problem parameters and the tuned algorithm hyperparameters for each dataset and inverse-problem pair are taken exactly from the experimental settings provided in the paper.

## 2. Running the Code

Everything below is for running the code in the github to solve inverse problems for few images (demonstration). To reproduce the paper I run and saved the results over all the datasets. To see that code, go to the following link, it has its own instructions there: 

```https://colab.research.google.com/drive/1evivuYHVaerBox09D6nQ6BeWVaYCrPw9?usp=sharing```


### 2.1 Required Folders

Place arbitrary RGB input images from one of the two datasets, CelebA and AFHQ-Cat, directly into the `input_images/` folder.

Since the datasets are very large, a `sample_inputs/` folder is included. It contains example test-set images from both datasets. Copy some images from one of the following dataset folders into `input_images/`:

```text
sample_inputs/celeba/
sample_inputs/afhq_cat/
```

You can also select images that do not belong to either dataset from the `non_dataset/` folder under `sample_inputs/`:

```text
sample_inputs/non_dataset/
```

### 2.2 Models

The pretrained models are already included in the `models/` folder. There are two models, one for each dataset.

### 2.3 Running

1. Open a terminal in the main project folder, where `run.py` is located.

2. Create a virtual environment and install the required packages.

For macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. The `scripts.txt` file contains separate commands for all five inverse problems and both datasets evaluated in the paper.

The inverse-problem parameters and the tuned algorithm hyperparameters for each inverse-problem and dataset pair are taken exactly from the paper and included as command-line arguments.

Copy a complete command from `scripts.txt` and run it in the terminal.

### 2.4 Outputs

Each processed image receives the following output files:

- `clean.png`
- `noisy_measurement.png`
- `reconstruction.png`
- `reconstruction_snapshots.png`

The output folder also contains:

- `run_config.json`
- `metrics_summary.json`

The snapshot figure has seven panels: the noisy measurement, the result after the first complete PnP-Flow iteration, and five equally spaced reconstruction states up to the final iteration. Each reconstruction panel is annotated with its iteration number and current \(t\) value.

## 3. Findings and Results

We fully reproduced the experiments presented in the paper using both datasets and all five inverse problems. The quantitative results in the reproduced tables are almost identical to the values reported in the paper. Images used in the original paper were also included in the report as qualitative examples, and the reconstructed outputs show a high level of visual similarity to the paper's results.

The ranking of the five inverse problems changes depending on whether the average PSNR, SSIM, or LPIPS values are considered. Therefore, the quantitative metrics do not produce a single consistent ordering across all tasks.

Across all inverse problems, some reconstructed images contained artificially generated realistic structures that were not present in the original images. 

Based on my visual evaluation, the five inverse problems can be ranked as follows.

### 3.1 Random Inpainting

Random inpainting produced the most successful reconstructions. Unlike box inpainting, the missing pixels are distributed randomly across the image. Therefore, even when one pixel is masked, neighboring pixels may still be observed.

This preserves local spatial correlations and provides useful information throughout the image, allowing the reconstruction to remain very close to the original.

### 3.2 Denoising

Denoising was the second most successful task. A small amount of oversmoothing was visible in some reconstructed images, but the overall structures and appearances remained close to those of the original images.

### 3.3 Super-Resolution

Super-resolution ranked third in my visual evaluation. In the reconstructed images, unnaturally sharp transitions were sometimes observed between neighboring pixels in relatively smooth regions.

In textured regions, this problem was less noticeable, and the overall image quality was similar to that of the denoising results, although some oversmoothing was still present.

### 3.4 Gaussian Deblurring

Gaussian deblurring ranked fourth. Similar to denoising, the reconstructed images were sometimes oversmoothed.

In addition, some artificially sharp lines and patterns that were not present in the original images appeared within the smoothed regions.

### 3.5 Box Inpainting

Box inpainting produced the least successful results. Since a complete spatial block is removed, recovering the missing region is substantially more difficult.

If a structure or pattern is located entirely inside the missing block and no part of it remains visible outside the block, the corresponding information is completely lost. In some examples from the original paper, the reconstructed block also appears unrelated to the surrounding image and resembles an artificial copy-pasted region.

In my experiments, the method occasionally produced reasonable reconstructions for images from the original datasets, but it often performed poorly on similar images that were not part of the datasets.

### 3.6 Results on Images Outside the Original Datasets

For images collected from the internet that had the same general format and belonged to the same semantic category as the training datasets, the reconstruction quality strongly depended on how closely the images matched the training distribution.

Images that were visually very similar to CelebA or AFHQ-Cat achieved performance comparable to that observed on the original test sets. However, for images that belonged to the same broad category but differed more noticeably in pose, composition, background, texture, or overall appearance, the reconstruction quality was generally worse.

This indicates that the learned flow-matching prior performs best for in-distribution images and becomes less reliable as the input moves further away from the distribution represented by the training dataset.