"""Evaluate one fixed segmentation model on original and denoised HC-18 images."""

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
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hc18_research.data import read_split, load_grayscale  # noqa: E402
from hc18_research.metrics import dice_score, iou_score  # noqa: E402
from hc18_research.models import UNet, USDRIFT2D  # noqa: E402
from hc18_research.scripts_helper import noisy_image  # noqa: E402


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() or not target.any():
        return float("inf")
    pred_boundary = prediction ^ (cv2.erode(prediction.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    target_boundary = target ^ (cv2.erode(target.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    pred_distance = distance_transform_edt(~pred_boundary)
    target_distance = distance_transform_edt(~target_boundary)
    distances = np.concatenate([target_distance[pred_boundary], pred_distance[target_boundary]])
    return float(np.percentile(distances, 95))


def circumference_mm(mask: np.ndarray, pixel_size: float) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    return float(max(cv2.arcLength(contour, True) for contour in contours) * pixel_size)


def resize_float(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize((size[1], size[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def prediction(model, image: np.ndarray, device, image_size: tuple[int, int]) -> np.ndarray:
    resized = resize_float(image, image_size)
    tensor = torch.from_numpy(resized.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = (torch.sigmoid(model(tensor))[0, 0].cpu().numpy() >= 0.5).astype(np.uint8)
    return np.asarray(Image.fromarray(mask * 255).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST)) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-config", type=Path, default=ROOT / "configs/baseline_segmentation.yaml")
    parser.add_argument("--denoise-config", type=Path, default=ROOT / "configs/usdrift2d.yaml")
    parser.add_argument("--seg-checkpoint", type=Path, default=None)
    parser.add_argument("--denoise-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/segmentation")
    args = parser.parse_args()
    seg_config = yaml.safe_load(args.seg_config.read_text(encoding="utf-8"))
    denoise_config = yaml.safe_load(args.denoise_config.read_text(encoding="utf-8"))
    records = read_split(ROOT / seg_config["val_split"])
    output_dir = args.output_dir; output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (int(seg_config["image_size"]), int(seg_config["image_size"]))

    seg_model = UNet(base_channels=int(seg_config["base_channels"])).to(device)
    seg_checkpoint = args.seg_checkpoint or (ROOT / seg_config["output_dir"] / "best_model.pt")
    seg_model.load_state_dict(torch.load(seg_checkpoint, map_location=device)["model"])
    seg_model.eval()
    denoise_model = USDRIFT2D(
        base_channels=int(denoise_config["base_channels"]), cv_window=int(denoise_config["cv_window"]),
        use_frequency=bool(denoise_config["use_frequency"]), epsilon=float(denoise_config["log_epsilon"]),
    ).to(device)
    denoise_checkpoint = args.denoise_checkpoint or (ROOT / denoise_config["output_dir"] / "best_model.pt")
    denoise_model.load_state_dict(torch.load(denoise_checkpoint, map_location=device)["model"])
    denoise_model.eval()

    methods = ["Original", "Noisy", "Gaussian", "Bilateral", "TV-Chambolle", "USDRIFT2D"]
    rows_by_method = {method: [] for method in methods}
    for index, record in enumerate(records):
        image = load_grayscale(record.image_path).astype(np.float32) / 255.0
        mask = load_grayscale(record.mask_path) > 0
        noisy = noisy_image((image * 255).astype(np.uint8), float(denoise_config["speckle_sigma"]), 20260920 + index)
        outputs = {
            "Original": image,
            "Noisy": noisy,
            "Gaussian": cv2.GaussianBlur(noisy, (5, 5), 1.2),
            "Bilateral": cv2.bilateralFilter(noisy, 7, 50.0 / 255.0, 50.0),
            "TV-Chambolle": __import__("skimage.restoration", fromlist=["denoise_tv_chambolle"]).denoise_tv_chambolle(noisy, weight=0.15),
        }
        with torch.no_grad():
            denoise_input = torch.from_numpy(resize_float(noisy, image_size).copy()).float().unsqueeze(0).unsqueeze(0).to(device)
            denoised_small = denoise_model(denoise_input).cpu().numpy()[0, 0]
            outputs["USDRIFT2D"] = np.asarray(
                Image.fromarray((denoised_small * 255).astype(np.uint8)).resize(
                    (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        for method, output in outputs.items():
            predicted = prediction(seg_model, np.clip(output, 0.0, 1.0), device, image_size)
            rows_by_method[method].append({
                "dice": dice_score(predicted, mask),
                "iou": iou_score(predicted, mask),
                "hd95_pixels": hd95(predicted, mask),
                "hc_pred_mm": circumference_mm(predicted, record.pixel_size),
                "hc_error_mm": abs(circumference_mm(predicted, record.pixel_size) - record.head_circumference_mm),
            })

    summary = []
    for method, rows in rows_by_method.items():
        summary.append({
            "method": method, "images": len(rows),
            "dice": float(np.mean([row["dice"] for row in rows])),
            "iou": float(np.mean([row["iou"] for row in rows])),
            "hd95_pixels": float(np.mean([row["hd95_pixels"] for row in rows if np.isfinite(row["hd95_pixels"])])),
            "hc_mae_mm": float(np.mean([row["hc_error_mm"] for row in rows])),
        })
    (output_dir / "segmentation_comparison.json").write_text(json.dumps({"dataset": "HC-18", "split": "validation", "results": summary}, indent=2), encoding="utf-8")
    with (output_dir / "segmentation_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
