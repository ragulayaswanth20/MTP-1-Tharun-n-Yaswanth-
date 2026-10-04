"""Train the HC-18 baseline U-Net segmentation model."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from hc18_research.data import HC18Dataset, read_split  # noqa: E402
from hc18_research.losses import BCEDiceLoss  # noqa: E402
from hc18_research.metrics import dice_score, iou_score  # noqa: E402
from hc18_research.models import UNet  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    losses, dices, ious = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        predictions = (torch.sigmoid(logits).detach().cpu().numpy() >= 0.5)
        targets = masks.detach().cpu().numpy() >= 0.5
        losses.append(float(loss.item()))
        dices.extend(dice_score(pred, target) for pred, target in zip(predictions, targets))
        ious.extend(iou_score(pred, target) for pred, target in zip(predictions, targets))
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices)), "iou": float(np.mean(ious))}


def write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_segmentation.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))

    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")
    train_records = read_split(ROOT / config["train_split"])
    val_records = read_split(ROOT / config["val_split"])
    image_size = (int(config["image_size"]), int(config["image_size"]))
    train_loader = DataLoader(
        HC18Dataset(train_records, image_size),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
    )
    val_loader = DataLoader(
        HC18Dataset(val_records, image_size),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base_channels=int(config["base_channels"])).to(device)
    criterion = BCEDiceLoss(positive_weight=float(config.get("bce_positive_weight", 1.0))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history = []
    best_dice = -float("inf")

    print(f"Training baseline U-Net on {len(train_records)} images; validation={len(val_records)}; device={device}")
    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, output_dir / "last_model.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, output_dir / "best_model.pt")
        print(
            f"epoch={epoch}/{config['epochs']} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

    write_history(output_dir / "train_metrics.csv", history)
    write_history(output_dir / "val_metrics.csv", history)
    print(f"Saved baseline checkpoints and metrics to {output_dir}")


if __name__ == "__main__":
    main()
