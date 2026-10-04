"""Train independent HC18 contour and filled-head segmentor baselines."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.hc18_dataset import read_split
from datasets.segmentation_diagnostics import HC18SegmentationDataset, prepare_filled_mask
from losses.total_loss import segmentation_loss
from metrics.apd_grn_metrics import boundary_f1, contour_metrics, extract_boundary, head_circumference_mm
from models.grn_components import UNetSegmentor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_predictions(model: UNetSegmentor, loader: DataLoader, device: torch.device) -> list[dict]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            probabilities = torch.softmax(model(image), dim=1)[:, 1].cpu().numpy()
            targets = batch["mask"][:, 0].numpy() > 0
            contours = batch["contour"][:, 0].numpy() > 0
            for index, filename in enumerate(batch["filename"]):
                predictions.append(
                    {
                        "filename": filename,
                        "probability": probabilities[index],
                        "target": targets[index],
                        "contour": contours[index],
                    }
                )
    return predictions


def evaluate_predictions(
    predictions: list[dict],
    records_by_name: dict,
    threshold: float,
    target_mode: str,
    tolerances: list[int],
) -> list[dict]:
    rows = []
    for item in predictions:
        filename = item["filename"]
        record = records_by_name[filename]
        predicted_region = item["probability"] >= threshold
        prediction_boundary = extract_boundary(predicted_region) if target_mode == "filled_mask" else predicted_region
        target_boundary = item["contour"]
        row = {
            "filename": filename,
            "threshold": threshold,
            **{f"target_{key}": value for key, value in contour_metrics(predicted_region, item["target"]).items()},
            **{f"contour_{key}": value for key, value in contour_metrics(prediction_boundary, target_boundary).items()},
        }
        for tolerance in tolerances:
            row[f"boundary_f1_{tolerance}px"] = boundary_f1(prediction_boundary, target_boundary, tolerance)
        with Image.open(record.image_path) as source:
            original_shape = (source.height, source.width)
        circumference = head_circumference_mm(
            prediction_boundary,
            record.pixel_size,
            original_shape,
            predicted_region.shape,
        )
        row["hc_predicted_mm"] = circumference
        row["hc_error_mm"] = abs(circumference - record.hc_mm) if np.isfinite(circumference) else float("nan")
        rows.append(row)
    return rows


def evaluate(
    model: UNetSegmentor,
    loader: DataLoader,
    records_by_name: dict,
    device: torch.device,
    threshold: float,
    target_mode: str,
    tolerances: list[int],
) -> list[dict]:
    predictions = collect_predictions(model, loader, device)
    return evaluate_predictions(predictions, records_by_name, threshold, target_mode, tolerances)
def mean_metric(rows: list[dict], name: str) -> float:
    values = np.asarray([row[name] for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def summarize(rows: list[dict]) -> dict:
    keys = [key for key in rows[0] if key != "filename"]
    return {"images": len(rows), **{key: mean_metric(rows, key) for key in keys}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/segmentor_diagnostics.yaml")
    parser.add_argument("--target", choices=("contour", "filled_mask", "both"), default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_size = int(config["image_size"])
    train_records = read_split(ROOT / config["split"]["train"])
    val_records = read_split(ROOT / config["split"]["validation"])
    overlap = {row.filename for row in train_records} & {row.filename for row in val_records}
    if overlap:
        raise ValueError(f"Train/validation split overlap: {sorted(overlap)[:5]}")
    filled_cache = ROOT / config["filled_mask_cache"]
    for record in train_records + val_records:
        prepare_filled_mask(record, filled_cache, image_size)

    output_root = ROOT / config["output_dir"]
    checkpoint_root = ROOT / config["checkpoint_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    selected_target = args.target or config.get("segmentation_target", "contour")
    modes = config["target_modes"] if selected_target == "both" else [selected_target]
    all_summaries = []

    for mode in modes:
        mode_dir = output_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        train_dataset = HC18SegmentationDataset(train_records, image_size, mode, True, filled_cache)
        val_dataset = HC18SegmentationDataset(val_records, image_size, mode, False, filled_cache)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        model = UNetSegmentor(num_classes=2, channels=tuple(config["segmentor_channels"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-5)
        records_by_name = {record.filename: record for record in val_records}
        patience = int(config["early_stopping_patience"])
        threshold_key = "boundary_f1_2px" if mode == "contour" else "target_dice"
        best_score = float("-inf")
        best_threshold = 0.5
        bad_epochs = 0
        history = []
        foreground_weight = float(config["foreground_weights"][mode])
        mode_checkpoint = checkpoint_root / f"best_segmentor_{mode}.pt"

        print(f"Reference-only segmentor: target={mode} train={len(train_records)} val={len(val_records)} device={device}")
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            losses = []
            for batch in train_loader:
                image = batch["image"].to(device, non_blocking=True)
                target = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(image)
                loss = segmentation_loss(logits, target, ce_weight=1.0, foreground_weight=foreground_weight)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))

            val_predictions = collect_predictions(model, val_loader, device)
            threshold_summaries = []
            threshold_rows = {}
            for threshold in config["thresholds"]:
                rows = evaluate_predictions(
                    val_predictions,
                    records_by_name,
                    float(threshold),
                    mode,
                    config["boundary_tolerances"],
                )
                summary = summarize(rows)
                threshold_summaries.append((summary[threshold_key], float(threshold), summary))
                threshold_rows[float(threshold)] = rows
            score, threshold, val_summary = max(threshold_summaries, key=lambda item: item[0])
            history_row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "selection_threshold": threshold,
                "selection_metric": threshold_key,
                "selection_score": score,
                **val_summary,
            }
            history.append(history_row)
            print(
                f"{mode} epoch={epoch}/{config['epochs']} train={history_row['train_loss']:.4f} "
                f"threshold={threshold:.2f} {threshold_key}={score:.4f} "
                f"dice={val_summary['target_dice']:.4f}"
            )
            if score > best_score:
                best_score = score
                best_threshold = threshold
                bad_epochs = 0
                best_rows = threshold_rows[threshold]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "target_mode": mode,
                        "threshold": threshold,
                        "channels": config["segmentor_channels"],
                        "validation_metrics": val_summary,
                        "config": config,
                    },
                    mode_checkpoint,
                )
                with (mode_dir / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=best_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(best_rows)
            else:
                bad_epochs += 1
            with (mode_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
            if bad_epochs >= patience:
                print(f"early stop {mode} after {epoch} epochs; best {threshold_key}={best_score:.4f}")
                break

        checkpoint = torch.load(mode_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        final_rows = evaluate_predictions(
            collect_predictions(model, val_loader, device),
            records_by_name,
            best_threshold,
            mode,
            config["boundary_tolerances"],
        )
        summary = {"target_mode": mode, "best_epoch": int(checkpoint["epoch"]), "threshold": best_threshold, **summarize(final_rows)}
        all_summaries.append(summary)
        (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if mode == "contour":
            torch.save(checkpoint, checkpoint_root / "best_segmentor.pt")
            torch.save(checkpoint, checkpoint_root / "best_segmentor_reference.pt")

    fields = list(all_summaries[0].keys())
    for filename in ("segmentor_baseline_metrics.csv", "contour_vs_filled_mask_metrics.csv"):
        with (output_root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_summaries)
    (output_root / "segmentor_baseline_metrics.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()