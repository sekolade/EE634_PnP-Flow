from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as v2
from torchvision.utils import save_image

from pnpflow.utils import postprocess

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InputImageDataset(Dataset):
    def __init__(self, input_dir, image_size):
        self.paths = sorted(
            path for path in Path(input_dir).iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No supported images found in {input_dir}")
        self.transform = v2.Compose([
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.name


def make_loader(input_dir, image_size, batch_size):
    dataset = InputImageDataset(input_dir, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def visualization_measurement(noisy_img, degradation, dataset, problem):
    if problem == "superresolution":
        image = degradation.H_adj(noisy_img)
    else:
        image = noisy_img
    return postprocess(image.clone(), dataset).clamp(0, 1).cpu()


def save_batch_outputs(output_dir, names, clean_img, noisy_img, rec_img,
                       snapshots, degradation, dataset, problem):
    clean_vis = postprocess(clean_img.clone(), dataset).clamp(0, 1).cpu()
    noisy_vis = visualization_measurement(
        noisy_img, degradation, dataset, problem
    )
    rec_vis = postprocess(rec_img.clone(), dataset).clamp(0, 1).cpu()

    snapshot_vis = [
        (iteration, t, postprocess(x.clone(), dataset).clamp(0, 1).cpu())
        for iteration, t, x in snapshots
    ]

    for index, name in enumerate(names):
        image_dir = Path(output_dir) / Path(name).stem
        image_dir.mkdir(parents=True, exist_ok=True)
        save_image(clean_vis[index], image_dir / "clean.png")
        save_image(noisy_vis[index], image_dir / "noisy_measurement.png")
        save_image(rec_vis[index], image_dir / "reconstruction.png")

        panels = [("Noisy measurement", noisy_vis[index])]
        panels.extend([
            (f"iter={iteration}\nt={t:.4f}", tensor[index])
            for iteration, t, tensor in snapshot_vis
        ])
        if len(panels) != 7:
            raise RuntimeError(f"Expected 7 snapshot panels, got {len(panels)}")

        fig, axes = plt.subplots(1, 7, figsize=(24, 4))
        for axis, (title, tensor) in zip(axes, panels):
            axis.imshow(tensor.permute(1, 2, 0).numpy())
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(image_dir / "reconstruction_snapshots.png", dpi=180)
        plt.close(fig)
