"""Run a true 3D Gaussian denoising benchmark on the ordered CETUS sequence."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.eval_denoising import (  # noqa: E402
    add_speckle_noise,
    compute_cnr,
    compute_enl,
    compute_epi,
    compute_psnr,
    compute_rmse,
    compute_ssim,
    save_residual_map,
)

DATA_DIR = ROOT / "data" / "cetus_p01"
OUTPUT_DIR = ROOT / "output" / "benchmark_results"
DENOISED_DIR = OUTPUT_DIR / "3d_gaussian"
VISUAL_DIR = OUTPUT_DIR / "visuals"
SIGMA_NOISE = 0.2
SEED = 42
SIGMA_Z = 1.0
SIGMA_YX = 1.2


def roi_pair(mask: np.ndarray, shape: tuple[int, int]):
    if np.count_nonzero(mask) > 100:
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
        return mask, (dilated > 0) & (~mask)
    height, width = shape
    target = np.zeros(shape, dtype=bool)
    background = np.zeros(shape, dtype=bool)
    target[int(height * 0.3):int(height * 0.5), int(width * 0.3):int(width * 0.5)] = True
    background[int(height * 0.55):int(height * 0.75), int(width * 0.55):int(width * 0.75)] = True
    return target, background


def main() -> None:
    original_dir = DATA_DIR / "original"
    mask_dir = DATA_DIR / "seg"
    files = sorted(original_dir.glob("*.png"))
    if len(files) < 2:
        raise RuntimeError("A sequence of at least two CETUS frames is required for 3D filtering.")

    clean_volume = np.stack([np.array(Image.open(path).convert("L")) for path in files])
    noisy_volume = np.stack([
        add_speckle_noise(frame, sigma=SIGMA_NOISE, seed=SEED + index)
        for index, frame in enumerate(clean_volume)
    ])

    print(
        f"Applying 3D Gaussian filter to volume {noisy_volume.shape} "
        f"with sigma=({SIGMA_Z}, {SIGMA_YX}, {SIGMA_YX})..."
    )
    denoised_volume = np.clip(
        gaussian_filter(noisy_volume.astype(np.float32), sigma=(SIGMA_Z, SIGMA_YX, SIGMA_YX)),
        0,
        255,
    ).astype(np.uint8)

    DENOISED_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {key: [] for key in ["psnr", "ssim", "rmse", "enl", "cnr", "epi"]}

    for index, (clean, denoised, path) in enumerate(zip(clean_volume, denoised_volume, files)):
        filename = path.name
        Image.fromarray(denoised).save(DENOISED_DIR / filename)
        mask = np.array(Image.open(mask_dir / filename).convert("L")) > 0
        target, background = roi_pair(mask, clean.shape)
        metrics["psnr"].append(compute_psnr(clean, denoised))
        metrics["ssim"].append(compute_ssim(clean, denoised))
        metrics["rmse"].append(compute_rmse(clean, denoised))
        metrics["enl"].append(compute_enl(denoised, target))
        metrics["cnr"].append(compute_cnr(denoised, target, background))
        metrics["epi"].append(compute_epi(clean, denoised, target))

        if index == 40:
            Image.fromarray(clean).save(VISUAL_DIR / "slice40_3d_gaussian_clean.png")
            Image.fromarray(noisy_volume[index]).save(VISUAL_DIR / "slice40_3d_gaussian_noisy.png")
            Image.fromarray(denoised).save(VISUAL_DIR / "slice40_3d_gaussian.png")
            save_residual_map(clean, denoised, VISUAL_DIR / "slice40_3d_gaussian_residual.png")

    result = {
        "Method": "3D Gaussian Filter",
        "PSNR (dB)": f"{np.mean(metrics['psnr']):.2f} +/- {np.std(metrics['psnr']):.2f}",
        "SSIM": f"{np.mean(metrics['ssim']):.4f} +/- {np.std(metrics['ssim']):.4f}",
        "RMSE": f"{np.mean(metrics['rmse']):.2f} +/- {np.std(metrics['rmse']):.2f}",
        "ENL (Speckle Smoothness)": f"{np.mean(metrics['enl']):.2f} +/- {np.std(metrics['enl']):.2f}",
        "CNR (Tissue Contrast)": f"{np.mean(metrics['cnr']):.2f} +/- {np.std(metrics['cnr']):.2f}",
        "EPI (Edge Preservation)": f"{np.mean(metrics['epi']):.3f} +/- {np.std(metrics['epi']):.3f}",
        "dataset": "CETUS ordered sequence",
        "frames": len(files),
        "noise_sigma": SIGMA_NOISE,
        "filter_sigma": [SIGMA_Z, SIGMA_YX, SIGMA_YX],
    }
    with (OUTPUT_DIR / "3d_gaussian_metrics.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    with (OUTPUT_DIR / "3d_gaussian_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))
    print(f"Saved denoised frames to {DENOISED_DIR}")


if __name__ == "__main__":
    main()
