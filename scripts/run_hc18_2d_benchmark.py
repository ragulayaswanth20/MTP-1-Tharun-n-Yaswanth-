"""HC-18 2D denoising benchmark for classical filters and GRN."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from run_grn_denoising import (  # noqa: E402
    PatchDiscriminator,
    SimpleSegmenter,
    UNetGenerator,
)
from utils.eval_denoising import (  # noqa: E402
    add_speckle_noise,
    compute_cnr,
    compute_enl,
    compute_epi,
    compute_psnr,
    compute_rmse,
    compute_ssim,
    filter_bilateral,
    filter_gaussian,
    filter_tv_chambolle,
)


class HC18Dataset(Dataset):
    def __init__(self, files: list[Path], image_size: int, sigma: float, seed: int):
        self.files = files
        self.image_size = image_size
        self.sigma = sigma
        self.seed = seed
        self.image_transform = T.Compose(
            [T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize([0.5], [0.5])]
        )
        self.mask_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image_path = self.files[index]
        image = np.array(Image.open(image_path).convert("L"))
        mask_path = image_path.with_name(f"{image_path.stem}_Annotation.png")
        mask = np.array(Image.open(mask_path).convert("L"))
        noise_seed = self.seed + int(image_path.stem.split("_")[0])
        noisy = add_speckle_noise(image, sigma=self.sigma, seed=noise_seed)
        return (
            self.image_transform(Image.fromarray(image)),
            self.image_transform(Image.fromarray(noisy)),
            (self.mask_transform(Image.fromarray(mask)) > 0).long().squeeze(0),
            image_path.name,
        )


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.glob("*.png")
        if not path.name.endswith("_Annotation.png")
        and path.with_name(f"{path.stem}_Annotation.png").exists()
    )


def train_grn(train_loader, device: torch.device, epochs: int):
    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    segmenter = SimpleSegmenter(2).to(device)
    opt_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_s = optim.Adam(segmenter.parameters(), lr=2e-4, betas=(0.5, 0.999))
    adversarial = nn.MSELoss()
    pixel = nn.L1Loss()
    segmentation = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        generator.train(); discriminator.train(); segmenter.train()
        for clean, noisy, mask, _ in train_loader:
            clean, noisy, mask = clean.to(device), noisy.to(device), mask.to(device)

            fake = generator(noisy).detach()
            discriminator_loss = 0.5 * (
                adversarial(discriminator(clean), torch.ones_like(discriminator(clean)))
                + adversarial(discriminator(fake), torch.zeros_like(discriminator(fake)))
            )
            opt_d.zero_grad(); discriminator_loss.backward(); opt_d.step()

            denoised = generator(noisy)
            generator_loss = (
                adversarial(discriminator(denoised), torch.ones_like(discriminator(denoised)))
                + 100.0 * pixel(denoised, clean)
                + 50.0 * segmentation(segmenter(denoised), mask)
            )
            opt_g.zero_grad(); generator_loss.backward(); opt_g.step()

            with torch.no_grad():
                denoised_for_segmentation = generator(noisy)
            segmenter_loss = segmentation(segmenter(denoised_for_segmentation), mask)
            opt_s.zero_grad(); segmenter_loss.backward(); opt_s.step()
        print(f"GRN epoch {epoch + 1}/{epochs} complete")
    return generator.eval()


def metrics(clean, result, mask):
    mask = mask > 0
    kernel = np.ones((15, 15), np.uint8)
    background = (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & ~mask
    return {
        "psnr": compute_psnr(clean, result),
        "ssim": compute_ssim(clean, result),
        "rmse": compute_rmse(clean, result),
        "enl": compute_enl(result, mask),
        "cnr": compute_cnr(result, mask, background),
        "epi": compute_epi(clean, result, mask),
    }


def evaluate_classical(files, output_dir, sigma, seed):
    methods = {"Noisy baseline": [], "Gaussian": [], "Bilateral": [], "TV-Chambolle": []}
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(files):
        clean = np.array(Image.open(image_path).convert("L"))
        mask = np.array(Image.open(image_path.with_name(f"{image_path.stem}_Annotation.png")).convert("L")) > 0
        noisy = add_speckle_noise(clean, sigma=sigma, seed=seed + int(image_path.stem.split("_")[0]))
        outputs = {
            "Noisy baseline": noisy,
            "Gaussian": filter_gaussian(noisy, sigma=1.2),
            "Bilateral": filter_bilateral(noisy, d=7, sigma_color=50.0, sigma_space=50.0),
            "TV-Chambolle": filter_tv_chambolle(noisy, weight=0.15),
        }
        for name, result in outputs.items():
            methods[name].append(metrics(clean, result, mask))
        if index == 0:
            Image.fromarray(clean).save(visual_dir / "clean.png")
            Image.fromarray(noisy).save(visual_dir / "noisy.png")
            for name, result in outputs.items():
                if name != "Noisy baseline":
                    Image.fromarray(result).save(visual_dir / f"{name.lower().replace('-', '').replace(' ', '_')}.png")
    return methods


def evaluate_grn(model, loader, output_dir, device, sigma, seed):
    values = []
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, (clean_t, noisy_t, mask_t, names) in enumerate(loader):
            result = model(noisy_t.to(device)).cpu().squeeze(1).numpy()
            clean_batch = ((clean_t.numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8).squeeze(1)
            result_batch = ((result * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            mask_batch = mask_t.numpy()
            for batch_index, name in enumerate(names):
                values.append(metrics(clean_batch[batch_index], result_batch[batch_index], mask_batch[batch_index]))
                if index == 0 and batch_index == 0:
                    Image.fromarray(result_batch[batch_index]).save(visual_dir / "grn.png")
    return values


def summarize(all_results):
    summary = []
    for method, rows in all_results.items():
        summary.append({
            "Method": method,
            "PSNR (dB)": f"{np.mean([r['psnr'] for r in rows]):.2f} +/- {np.std([r['psnr'] for r in rows]):.2f}",
            "SSIM": f"{np.mean([r['ssim'] for r in rows]):.4f} +/- {np.std([r['ssim'] for r in rows]):.4f}",
            "RMSE": f"{np.mean([r['rmse'] for r in rows]):.2f} +/- {np.std([r['rmse'] for r in rows]):.2f}",
            "ENL": f"{np.mean([r['enl'] for r in rows]):.2f} +/- {np.std([r['enl'] for r in rows]):.2f}",
            "CNR": f"{np.mean([r['cnr'] for r in rows]):.2f} +/- {np.std([r['cnr'] for r in rows]):.2f}",
            "EPI": f"{np.mean([r['epi'] for r in rows]):.3f} +/- {np.std([r['epi'] for r in rows]):.3f}",
            "images": len(rows),
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Ultrasound_Dataset/HC-18_Fetus head dataset/training_set")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/hc18_2d_results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    files = list_images(args.data_dir)
    if len(files) < 2:
        raise RuntimeError(f"Expected paired HC-18 images and annotations in {args.data_dir}")
    shuffled = files.copy(); random.Random(args.seed).shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1 - args.val_fraction)))
    train_files, val_files = shuffled[:split], shuffled[split:]
    print(f"HC-18 paired images: {len(files)}; train: {len(train_files)}; validation: {len(val_files)}")

    output_dir = args.output_dir; output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = HC18Dataset(train_files, 256, args.sigma, args.seed)
    val_dataset = HC18Dataset(val_files, 256, args.sigma, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    results = evaluate_classical(val_files, output_dir, args.sigma, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_grn(train_loader, device, args.epochs)
    results["GRN"] = evaluate_grn(model, val_loader, output_dir, device, args.sigma, args.seed)
    summary = summarize(results)
    with (output_dir / "hc18_2d_metrics.json").open("w") as handle:
        json.dump({"dataset": "HC-18", "split": "deterministic 80/20", "sigma": args.sigma, "results": summary}, handle, indent=2)
    with (output_dir / "hc18_2d_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
