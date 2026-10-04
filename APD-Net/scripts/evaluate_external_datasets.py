"""Evaluate the saved APD-Net checkpoint on external breast-ultrasound images."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.apd_grn_metrics import image_metrics
from models.apdnet import APDNet


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    case_id: str
    image_path: Path


def discover(config: dict, dataset_name: str) -> tuple[list[ImageRecord], int]:
    dataset = config["datasets"][dataset_name]
    folder = (ROOT / dataset["root"]).resolve()
    records = []
    mask_count = 0
    if dataset["type"] == "numbered_bmp":
        expression = re.compile(dataset["case_regex"])
        images = sorted(folder.glob(dataset["image_pattern"]))
        masks = list(folder.glob(dataset["mask_pattern"]))
        mask_count = len(masks)
        for path in images:
            match = expression.fullmatch(path.name)
            if match:
                records.append(ImageRecord(dataset_name, match.group(1), path))
    elif dataset["type"] == "busi":
        files = sorted(folder.glob("*.png"))
        masks = [path for path in files if "_mask" in path.stem.lower()]
        images = [path for path in files if "_mask" not in path.stem.lower()]
        mask_count = len(masks)
        records.extend(ImageRecord(dataset_name, path.stem, path) for path in images)
    elif dataset["type"] == "breast_case_png":
        expression = re.compile(dataset["case_regex"])
        files = sorted(folder.glob(dataset["image_pattern"]))
        images = [path for path in files if expression.fullmatch(path.name)]
        masks = [path for path in files if path.stem.endswith(("_tumor", "_other1", "_other2", "_other3", "_other4"))]
        mask_count = len(masks)
        records.extend(
            ImageRecord(dataset_name, expression.fullmatch(path.name).group(1), path)
            for path in images
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset['type']}")
    return records, mask_count


def load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L")
        resized = gray.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32).copy() / 255.0


def make_panel(reference: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, output: Path, title: str) -> None:
    error = np.abs(denoised - reference)
    error /= max(float(error.max()), 1e-8)
    cells = [("Original reference", reference), ("Synthetic noisy", noisy), ("APD-Net", denoised), ("Absolute error", error)]
    width, height, label_height = 256, 256, 24
    canvas = Image.new("RGB", (2 * width, 2 * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, data) in enumerate(cells):
        row, column = divmod(index, 2)
        x, y = column * width, row * (height + label_height)
        gray = np.round(np.clip(data, 0, 1) * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(gray).convert("RGB"), (x, y + label_height))
        draw.text((x + 4, y + 4), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def summarize(rows: list[dict]) -> dict:
    result = {"dataset": rows[0]["dataset"], "cases": len(rows)}
    for key in (
        "noisy_psnr_db",
        "apd_psnr_db",
        "psnr_gain_db",
        "noisy_ssim",
        "apd_ssim",
        "ssim_gain",
        "noisy_mae",
        "apd_mae",
        "mae_reduction",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[f"{key}_median"] = float(np.median(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/external_datasets.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = APDNet(base_channels=32).to(device)
    checkpoint = torch.load(ROOT / config["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_dir = ROOT / config["output_dir"]
    examples_per_dataset = int(config.get("visual_examples_per_dataset", 3))
    sigma = float(config["noise"]["sigma"])
    noise_seed = int(config["noise"]["seed"])
    image_size = int(config["image_size"])
    all_rows = []
    summaries = []
    inventory = []
    offset = 0

    for dataset_name in config["datasets"]:
        records, mask_count = discover(config, dataset_name)
        dataset_rows = []
        saved_examples = 0
        for record in records:
            reference = load_image(record.image_path, image_size)
            generator = torch.Generator(device=device).manual_seed(noise_seed + offset)
            noise = torch.randn((1, 1, image_size, image_size), generator=generator, device=device)
            reference_tensor = torch.from_numpy(reference.copy()).to(device)[None, None]
            noisy = (reference_tensor * (1.0 + sigma * noise)).clamp(0.0, 1.0)
            with torch.no_grad():
                denoised = model(noisy)["denoised"]
            noisy_array = noisy[0, 0].cpu().numpy()
            denoised_array = denoised[0, 0].cpu().numpy()
            noisy_metrics = image_metrics(noisy_array, reference)
            apd_metrics = image_metrics(denoised_array, reference)
            row = {
                "dataset": dataset_name,
                "case_id": record.case_id,
                "image_path": str(record.image_path),
                "sigma": sigma,
                "noisy_psnr_db": noisy_metrics["psnr_db"],
                "apd_psnr_db": apd_metrics["psnr_db"],
                "psnr_gain_db": apd_metrics["psnr_db"] - noisy_metrics["psnr_db"],
                "noisy_ssim": noisy_metrics["ssim"],
                "apd_ssim": apd_metrics["ssim"],
                "ssim_gain": apd_metrics["ssim"] - noisy_metrics["ssim"],
                "noisy_mae": noisy_metrics["mae"],
                "apd_mae": apd_metrics["mae"],
                "mae_reduction": noisy_metrics["mae"] - apd_metrics["mae"],
            }
            dataset_rows.append(row)
            all_rows.append(row)
            if saved_examples < examples_per_dataset:
                panel = output_dir / "figures" / dataset_name / f"{record.case_id}_comparison.png"
                make_panel(reference, noisy_array, denoised_array, panel, dataset_name)
                saved_examples += 1
            offset += 1
        summaries.append(summarize(dataset_rows))
        inventory.append(
            {
                "dataset": dataset_name,
                "source_images_cases": len(records),
                "annotation_mask_files": mask_count,
                "images_evaluated": len(dataset_rows),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("per_image_metrics.csv", all_rows),
        ("dataset_comparison.csv", summaries),
        ("dataset_inventory.csv", inventory),
    ):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "checkpoint": config["checkpoint"],
                "checkpoint_epoch": checkpoint.get("epoch"),
                "image_size": image_size,
                "noise_model": config["noise"]["model"],
                "sigma": sigma,
                "noise_seed": noise_seed,
                "reference_description": "each original dataset image; not a speckle-free clinical reference",
                "total_images_evaluated": len(all_rows),
                "results": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))
    print(f"APD-Net external-dataset results saved to {output_dir}")


if __name__ == "__main__":
    main()