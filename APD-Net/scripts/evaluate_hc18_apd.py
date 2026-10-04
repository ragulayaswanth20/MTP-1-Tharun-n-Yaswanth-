"""Re-evaluate a saved APD-Net checkpoint on the HC18 validation split."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.hc18_dataset import HC18Dataset, read_split
from metrics.apd_grn_metrics import image_metrics
from models.apdnet import APDNet


def fixed_noise(reference: torch.Tensor, sigma: float, seed: int, offset: int) -> torch.Tensor:
    noise = []
    for index in range(reference.shape[0]):
        generator = torch.Generator(device=reference.device).manual_seed(seed + offset + index)
        noise.append(torch.randn(reference[index].shape, generator=generator, device=reference.device))
    return (reference * (1.0 + sigma * torch.stack(noise))).clamp(0.0, 1.0)


def save_panel(reference: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, path: Path) -> None:
    error = np.abs(denoised - reference)
    error /= max(float(error.max()), 1e-8)
    images = [
        ("Original HC18 reference", reference),
        ("Synthetic noisy, sigma=0.20", noisy),
        ("APD-Net output", denoised),
        ("Absolute error", error),
    ]
    canvas = Image.new("RGB", (512, 560), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        row, column = divmod(index, 2)
        x, y = column * 256, row * 280
        array = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        canvas.paste(Image.fromarray(array).convert("RGB"), (x, y + 24))
        draw.text((x + 4, y + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/apd_reconstruction_fair.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/hc18_apd_rerun_20261004")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint_path = args.checkpoint or (ROOT / config["checkpoint"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records = read_split(ROOT / config["data"]["val_split"])
    loader = DataLoader(
        HC18Dataset(records, int(config["image_size"])),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model = APDNet(int(config["base_channels"])).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    sigma_results = []
    per_image_rows = []

    for sigma in config["noise"]["evaluation_sigmas"]:
        sigma = float(sigma)
        rows = []
        offset = 0
        saved_panels = 0
        with torch.no_grad():
            for batch in loader:
                reference = batch["image"].to(device)
                noisy = fixed_noise(
                    reference,
                    sigma,
                    int(config["noise"]["validation_seed"]),
                    offset,
                )
                denoised = model(noisy)["denoised"]
                reference_np = reference[:, 0].cpu().numpy()
                noisy_np = noisy[:, 0].cpu().numpy()
                denoised_np = denoised[:, 0].cpu().numpy()
                for index, filename in enumerate(batch["filename"]):
                    noisy_metrics = image_metrics(noisy_np[index], reference_np[index])
                    apd_metrics = image_metrics(denoised_np[index], reference_np[index])
                    row = {
                        "filename": filename,
                        "sigma": sigma,
                        **{f"noisy_{key}": value for key, value in noisy_metrics.items()},
                        **{f"apd_{key}": value for key, value in apd_metrics.items()},
                        "psnr_gain_db": apd_metrics["psnr_db"] - noisy_metrics["psnr_db"],
                        "ssim_gain": apd_metrics["ssim"] - noisy_metrics["ssim"],
                        "mae_reduction": noisy_metrics["mae"] - apd_metrics["mae"],
                    }
                    rows.append(row)
                    per_image_rows.append(row)
                    if abs(sigma - 0.20) < 1e-8 and saved_panels < 5:
                        save_panel(
                            reference_np[index],
                            noisy_np[index],
                            denoised_np[index],
                            args.output_dir / "figures" / f"{Path(filename).stem}_sigma_0.20.png",
                        )
                        saved_panels += 1
                offset += reference.shape[0]

        summary = {"sigma": sigma, "cases": len(rows)}
        for metric in ("psnr_db", "ssim", "mae", "psnr_gain_db", "ssim_gain", "mae_reduction"):
            for condition in ("noisy", "apd") if metric in {"psnr_db", "ssim", "mae"} else ("",):
                key = f"{condition}_{metric}" if condition else metric
                values = np.asarray([row[key] for row in rows], dtype=np.float64)
                summary[f"{key}_mean"] = float(values.mean())
                summary[f"{key}_std"] = float(values.std(ddof=1))
        sigma_results.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_image_rows)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sigma_results[0].keys())
        writer.writeheader()
        writer.writerows(sigma_results)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "dataset": "HC18",
                "split": "validation",
                "cases": len(records),
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "image_size": config["image_size"],
                "noise_model": "multiplicative Gaussian speckle",
                "reference_description": "original HC18 image; not speckle-free ground truth",
                "noise_seed": config["noise"]["validation_seed"],
                "results": sigma_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(sigma_results, indent=2))
    print(f"Saved HC18 rerun results to {args.output_dir}")


if __name__ == "__main__":
    main()