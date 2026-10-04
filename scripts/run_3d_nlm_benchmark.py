import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from skimage.restoration import denoise_nl_means

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eval_denoising import (
    add_speckle_noise,
    compute_cnr,
    compute_epi,
    compute_enl,
    compute_psnr,
    compute_rmse,
    compute_ssim,
    save_residual_map,
)

ROOT = "/home/teaching/MTP_1_Sneha/Denoising"
DATA_DIR = os.path.join(ROOT, "data", "cetus_p01")
OUT_DIR = os.path.join(ROOT, "output", "benchmark_results")
NLM_DIR = os.path.join(OUT_DIR, "3d_nlm")
VIS_DIR = os.path.join(OUT_DIR, "visuals")


def roi_pair(mask, shape):
    if np.count_nonzero(mask) > 100:
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
        return mask, (dilated > 0) & (~mask)

    height, width = shape
    target = np.zeros(shape, dtype=bool)
    background = np.zeros(shape, dtype=bool)
    target[int(height * 0.3):int(height * 0.5), int(width * 0.3):int(width * 0.5)] = True
    background[int(height * 0.55):int(height * 0.75), int(width * 0.55):int(width * 0.75)] = True
    return target, background


def main():
    original_dir = os.path.join(DATA_DIR, "original")
    mask_dir = os.path.join(DATA_DIR, "seg")
    files = sorted(glob.glob(os.path.join(original_dir, "*.png")))
    os.makedirs(NLM_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    clean_volume = np.stack([np.array(Image.open(path).convert("L")) for path in files])
    noisy_volume = np.stack([
        add_speckle_noise(clean, sigma=0.2, seed=42 + index)
        for index, clean in enumerate(clean_volume)
    ])

    noisy_float = noisy_volume.astype(np.float32) / 255.0
    print(f"Running 3D NLM on volume {noisy_float.shape}...")
    denoised_float = denoise_nl_means(
        noisy_float,
        h=0.08,
        patch_size=3,
        patch_distance=5,
        fast_mode=True,
        channel_axis=None,
    )
    denoised_volume = np.clip(denoised_float * 255.0, 0, 255).astype(np.uint8)

    metrics = {key: [] for key in ["psnr", "ssim", "rmse", "enl", "cnr", "epi"]}
    for index, (clean, denoised, path) in enumerate(zip(clean_volume, denoised_volume, files)):
        filename = os.path.basename(path)
        Image.fromarray(denoised).save(os.path.join(NLM_DIR, filename))
        mask = np.array(Image.open(os.path.join(mask_dir, filename)).convert("L")) > 0
        target, background = roi_pair(mask, clean.shape)
        metrics["psnr"].append(compute_psnr(clean, denoised))
        metrics["ssim"].append(compute_ssim(clean, denoised))
        metrics["rmse"].append(compute_rmse(clean, denoised))
        metrics["enl"].append(compute_enl(denoised, target))
        metrics["cnr"].append(compute_cnr(denoised, target, background))
        metrics["epi"].append(compute_epi(clean, denoised, target))

        if index == 40:
            Image.fromarray(clean).save(os.path.join(VIS_DIR, "slice40_3d_nlm.png"))
            save_residual_map(clean, denoised, os.path.join(VIS_DIR, "slice40_3d_nlm_residual.png"))

    result = {
        "Method": "3D NLM",
        "PSNR (dB) \u2191": f"{np.mean(metrics['psnr']):.2f} \u00b1 {np.std(metrics['psnr']):.2f}",
        "SSIM \u2191": f"{np.mean(metrics['ssim']):.4f} \u00b1 {np.std(metrics['ssim']):.4f}",
        "RMSE \u2193": f"{np.mean(metrics['rmse']):.2f} \u00b1 {np.std(metrics['rmse']):.2f}",
        "ENL (Speckle Smoothness) \u2191": f"{np.mean(metrics['enl']):.2f} \u00b1 {np.std(metrics['enl']):.2f}",
        "CNR (Tissue Contrast) \u2191": f"{np.mean(metrics['cnr']):.2f} \u00b1 {np.std(metrics['cnr']):.2f}",
        "EPI (Edge Preservation) \u2191": f"{np.mean(metrics['epi']):.3f} \u00b1 {np.std(metrics['epi']):.3f}",
    }
    result_path = os.path.join(OUT_DIR, "3d_nlm_metrics.json")
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2)

    unified_path = os.path.join(ROOT, "output", "unified_results.json")
    with open(unified_path) as handle:
        unified = json.load(handle)
    unified = [row for row in unified if row.get("Method") != "3D NLM"]
    unified.append(result)
    with open(unified_path, "w") as handle:
        json.dump(unified, handle, indent=2)

    with open(os.path.join(OUT_DIR, "3d_nlm_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(json.dumps(result, indent=2))
    print(f"Saved 3D NLM slices to {NLM_DIR}")


if __name__ == "__main__":
    main()
