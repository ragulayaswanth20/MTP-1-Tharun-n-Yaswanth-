"""Compare classical denoisers and a trained USDRIFT2D on HC-18 validation data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hc18_research.data import HC18Dataset, read_split  # noqa: E402
from hc18_research.metrics.denoising import gradient_error, mae, psnr, ssim  # noqa: E402
from hc18_research.models import USDRIFT2D  # noqa: E402


def noisy_image(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    normalized = image.astype(np.float32) / 255.0
    return np.clip(normalized * (1.0 + rng.normal(0.0, sigma, normalized.shape)), 0.0, 1.0).astype(np.float32)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "psnr": psnr(prediction, target),
        "ssim": ssim(prediction, target),
        "mae": mae(prediction, target),
        "gradient_error": gradient_error(prediction, target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/usdrift2d.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/denoising")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    records = read_split(ROOT / config["val_split"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or (ROOT / config["output_dir"] / "best_model.pt")
    if checkpoint_path.exists():
        model = USDRIFT2D(
            base_channels=int(config["base_channels"]), cv_window=int(config["cv_window"]),
            use_frequency=bool(config["use_frequency"]), epsilon=float(config["log_epsilon"]),
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

    methods = {name: [] for name in ["Noisy", "Median", "Gaussian", "Bilateral", "TV-Chambolle"]}
    if model is not None:
        methods["USDRIFT2D"] = []
    with torch.no_grad():
        for index, record in enumerate(records):
            clean = np.asarray(Image.open(record.image_path).convert("L"), dtype=np.float32) / 255.0
            noisy = noisy_image((clean * 255).astype(np.uint8), float(config["speckle_sigma"]), 20260920 + index)
            outputs = {
                "Noisy": noisy,
                "Median": cv2.medianBlur((noisy * 255).astype(np.uint8), 5).astype(np.float32) / 255.0,
                "Gaussian": cv2.GaussianBlur(noisy, (5, 5), 1.2),
                "Bilateral": cv2.bilateralFilter(noisy, 7, 50.0 / 255.0, 50.0),
                "TV-Chambolle": denoise_tv_chambolle(noisy, weight=0.15),
            }
            if model is not None:
                input_tensor = torch.from_numpy(noisy.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
                outputs["USDRIFT2D"] = model(input_tensor).cpu().numpy()[0, 0]
            for name, prediction in outputs.items():
                methods[name].append(metrics(np.clip(prediction, 0.0, 1.0), clean))

    summary = []
    for method, rows in methods.items():
        summary.append({
            "method": method,
            "images": len(rows),
            "psnr": float(np.mean([row["psnr"] for row in rows])),
            "ssim": float(np.mean([row["ssim"] for row in rows])),
            "mae": float(np.mean([row["mae"] for row in rows])),
            "gradient_error": float(np.mean([row["gradient_error"] for row in rows])),
        })
    (output_dir / "denoising_metrics.json").write_text(json.dumps({"dataset": "HC-18", "split": "validation", "sigma": config["speckle_sigma"], "results": summary}, indent=2), encoding="utf-8")
    with (output_dir / "denoising_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
