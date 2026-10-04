"""Compare image conditions using one fixed pretrained HC18 segmentor."""
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
from metrics.apd_grn_metrics import boundary_f1, contour_metrics, extract_boundary, head_circumference_mm, image_metrics
from models.apdnet import APDNet
from models.grn_components import UNetSegmentor


def fixed_noise(reference: torch.Tensor, sigma: float, seed: int, offset: int) -> torch.Tensor:
    noise = []
    for index in range(reference.shape[0]):
        generator = torch.Generator(device=reference.device).manual_seed(seed + offset + index)
        noise.append(torch.randn(reference[index].shape, generator=generator, device=reference.device))
    return (reference * (1.0 + sigma * torch.stack(noise))).clamp(0.0, 1.0)


def save_panel(case: dict, filename: str, output_dir: Path) -> None:
    reference = case["B0_reference"]
    gray = lambda image: np.repeat(np.round(np.clip(image, 0, 1)[..., None] * 255).astype(np.uint8), 3, axis=2)

    def overlay(condition: str) -> np.ndarray:
        result = gray(reference)
        ground_truth = case["ground_truth"].astype(bool)
        prediction = case[f"seg_{condition}"].astype(bool)
        result[ground_truth & ~prediction] = (255, 40, 40)
        result[prediction & ~ground_truth] = (40, 255, 80)
        result[ground_truth & prediction] = (255, 230, 30)
        return result

    apd_error = np.abs(case["B2_APD"] - reference)
    grn_error = np.abs(case["B3_APDGRN"] - reference)
    apd_error /= max(float(apd_error.max()), 1e-8)
    grn_error /= max(float(grn_error.max()), 1e-8)
    cells = [
        ("Reference image", gray(reference)),
        ("Noisy input", gray(case["B1_noisy"])),
        ("Ordinary APD-Net", gray(case["B2_APD"])),
        ("APD-GRN", gray(case["B3_APDGRN"])),
        ("GRN-native architecture adaptation", gray(case["B4_GRN_native_arch_adaptation"])),
        ("|APD - reference|", gray(apd_error)),
        ("|APD-GRN - reference|", gray(grn_error)),
        ("HC18 reference contour", gray(case["ground_truth"])),
        ("S(reference)", overlay("B0_reference")),
        ("S(noisy)", overlay("B1_noisy")),
        ("S(APD)", overlay("B2_APD")),
        ("S(APD-GRN)", overlay("B3_APDGRN")),
    ]
    cell = 256
    title = 24
    canvas = Image.new("RGB", (4 * cell, 3 * (cell + title)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(cells):
        row, column = divmod(index, 4)
        x, y = column * cell, row * (cell + title)
        canvas.paste(Image.fromarray(array), (x, y + title))
        draw.text((x + 4, y + 4), label, fill="black")
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / f"{Path(filename).stem}_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/apd_grn_stagec.yaml")
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--segmentor", type=Path, default=ROOT / "checkpoints/diagnostics/best_segmentor_reference.pt")
    parser.add_argument("--apd", type=Path, default=ROOT / "checkpoints/diagnostics/best_apd_reconstruction.pt")
    parser.add_argument("--apdgrn", type=Path, default=ROOT / "checkpoints/diagnostics/best_segmentation.pt")
    parser.add_argument(
        "--native-grn",
        type=Path,
        default=ROOT.parent / "output/grn_hc18_original_denoising/best_grn_hc18.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/diagnostics/input_conditions")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records = read_split(ROOT / config["data"]["val_split"])
    loader = DataLoader(
        HC18Dataset(records, int(config["image_size"])),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    seg_checkpoint = torch.load(args.segmentor, map_location=device, weights_only=False)
    segmentor = UNetSegmentor(num_classes=2, channels=tuple(config["segmentor_channels"])).to(device)
    segmentor.load_state_dict(seg_checkpoint["model"])
    segmentor.eval()
    threshold = float(seg_checkpoint.get("threshold", 0.5))

    apd_checkpoint = torch.load(args.apd, map_location=device, weights_only=False)
    apd = APDNet(int(config["base_channels"])).to(device)
    apd.load_state_dict(apd_checkpoint["model"])
    apd.eval()

    apdgrn_checkpoint = torch.load(args.apdgrn, map_location=device, weights_only=False)
    apdgrn = APDNet(int(config["base_channels"])).to(device)
    apdgrn.load_state_dict(apdgrn_checkpoint["generator"])
    apdgrn.eval()

    grn_src = ROOT.parent / "GRN/src"
    sys.path.insert(0, str(grn_src))
    from grn.models.generator import UNetGenerator

    native_grn_checkpoint = torch.load(args.native_grn, map_location=device, weights_only=False)
    native_grn = UNetGenerator().to(device)
    native_grn.load_state_dict(native_grn_checkpoint["generator"])
    native_grn.eval()

    names = ("B0_reference", "B1_noisy", "B2_APD", "B3_APDGRN", "B4_GRN_native_arch_adaptation")
    by_condition = {name: [] for name in names}
    output_rows = []
    visual_cases = {}
    offset = 0
    with torch.no_grad():
        for batch in loader:
            reference = batch["image"].to(device)
            target = batch["mask"][:, 0].numpy() > 0
            noisy = fixed_noise(reference, args.sigma, int(config["noise"]["evaluation_seed"]), offset)
            apd_image = apd(noisy)["denoised"]
            apdgrn_image = apdgrn(noisy)["denoised"]
            native_input = noisy * 2.0 - 1.0
            native_image = ((native_grn(native_input).clamp(-1, 1) + 1.0) * 0.5).clamp(0, 1)
            conditions = {
                "B0_reference": reference,
                "B1_noisy": noisy,
                "B2_APD": apd_image,
                "B3_APDGRN": apdgrn_image,
                "B4_GRN_native_arch_adaptation": native_image,
            }
            for condition, image in conditions.items():
                probabilities = torch.softmax(segmentor(image), dim=1)[:, 1].cpu().numpy()
                predicted = probabilities >= threshold
                for index, filename in enumerate(batch["filename"]):
                    target_contour = target[index]
                    prediction_contour = predicted[index]
                    record = next(item for item in records if item.filename == filename)
                    with __import__("PIL.Image", fromlist=["Image"]).open(record.image_path) as source:
                        original_shape = (source.height, source.width)
                    row = {"condition": condition, "filename": filename, "sigma": args.sigma}
                    row.update(contour_metrics(prediction_contour, target_contour))
                    row.update(
                        {f"image_{key}": value for key, value in image_metrics(
                            image[index, 0].cpu().numpy(), reference[index, 0].cpu().numpy()
                        ).items()}
                    )
                    for tolerance in (1, 2, 3, 5):
                        row[f"boundary_f1_{tolerance}px"] = boundary_f1(
                            prediction_contour, target_contour, tolerance
                        )
                    hc_prediction = head_circumference_mm(
                        prediction_contour,
                        record.pixel_size,
                        original_shape,
                        tuple(prediction_contour.shape),
                    )
                    row["hc_error_mm"] = abs(hc_prediction - record.hc_mm) if np.isfinite(hc_prediction) else float("nan")
                    by_condition[condition].append(row)
                    output_rows.append(row)
                    if filename not in visual_cases:
                        visual_cases[filename] = {
                            "ground_truth": target_contour,
                            "B0_reference": reference[index, 0].cpu().numpy(),
                        }
                    visual_cases[filename][condition] = image[index, 0].cpu().numpy()
                    visual_cases[filename][f"seg_{condition}"] = prediction_contour
            offset += reference.shape[0]

    summary = []
    for condition, rows in by_condition.items():
        metrics = [key for key in rows[0] if key not in {"condition", "filename", "sigma"}]
        summary.append(
            {
                "condition": condition,
                "images": len(rows),
                **{metric: float(np.nanmean([row[metric] for row in rows])) for metric in metrics},
            }
        )
    apdgrn_rows = by_condition["B3_APDGRN"]
    ranked = sorted(apdgrn_rows, key=lambda row: row["image_psnr_db"])
    selected = {
        "difficult": ranked[0],
        "average": min(ranked, key=lambda row: abs(row["image_psnr_db"] - np.median([item["image_psnr_db"] for item in ranked]))),
        "good": ranked[-1],
    }
    panel_dir = args.output_dir / "figures/segmentation_examples"
    for label, row in selected.items():
        save_panel(visual_cases[row["filename"]], row["filename"], panel_dir / label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "input_conditions_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in output_rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    with (args.output_dir / "input_conditions_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in summary for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "input_conditions_summary.json").write_text(
        json.dumps({"sigma": args.sigma, "shared_segmentor": str(args.segmentor), "results": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()