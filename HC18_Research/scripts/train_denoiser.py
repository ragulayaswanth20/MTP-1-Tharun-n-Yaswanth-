"""Train USDRIFT2D on controlled synthetic speckle corruption."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc18_research.data import HC18Dataset, read_split  # noqa: E402
from hc18_research.losses import AnatomyAwareDenoisingLoss  # noqa: E402
from hc18_research.metrics.denoising import gradient_error, mae, psnr, ssim  # noqa: E402
from hc18_research.models import USDRIFT2D  # noqa: E402
from hc18_research.utils import multiplicative_speckle  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device, sigma):
    model.eval()
    loss_values, metric_values = [], {key: [] for key in ["psnr", "ssim", "mae", "gradient_error"]}
    with torch.no_grad():
        for batch in loader:
            clean = batch["image"].to(device)
            masks = batch["mask"].to(device)
            noisy = multiplicative_speckle(clean, sigma)
            prediction = model(noisy)
            loss, _ = criterion(prediction, clean, masks)
            loss_values.append(float(loss.item()))
            prediction_np = prediction.cpu().numpy()[:, 0]
            clean_np = clean.cpu().numpy()[:, 0]
            for pred, target in zip(prediction_np, clean_np):
                metric_values["psnr"].append(psnr(pred, target))
                metric_values["ssim"].append(ssim(pred, target))
                metric_values["mae"].append(mae(pred, target))
                metric_values["gradient_error"].append(gradient_error(pred, target))
    return {"loss": float(np.mean(loss_values)), **{key: float(np.mean(value)) for key, value in metric_values.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/usdrift2d.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config.yaml")

    image_size = (int(config["image_size"]), int(config["image_size"]))
    train_loader = DataLoader(HC18Dataset(read_split(ROOT / config["train_split"]), image_size), batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["num_workers"]))
    val_loader = DataLoader(HC18Dataset(read_split(ROOT / config["val_split"]), image_size), batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = USDRIFT2D(
        base_channels=int(config["base_channels"]),
        cv_window=int(config["cv_window"]),
        use_frequency=bool(config["use_frequency"]),
        epsilon=float(config["log_epsilon"]),
    ).to(device)
    criterion = AnatomyAwareDenoisingLoss(
        pixel_weight=float(config["pixel_weight"]), ssim_weight=float(config["ssim_weight"]),
        gradient_weight=float(config["gradient_weight"]), frequency_weight=float(config["frequency_weight"]),
        anatomy_weight=float(config["anatomy_weight"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history, best_psnr = [], -float("inf")
    print(f"Training USDRIFT2D on {len(train_loader.dataset)} images; validation={len(val_loader.dataset)}; device={device}")
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            clean = batch["image"].to(device)
            masks = batch["mask"].to(device)
            noisy = multiplicative_speckle(clean, float(config["speckle_sigma"]))
            prediction = model(noisy)
            loss, _ = criterion(prediction, clean, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_metrics = evaluate(model, val_loader, criterion, device, float(config["speckle_sigma"]))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "config": config}
        torch.save(checkpoint, output_dir / "last_model.pt")
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save(checkpoint, output_dir / "best_model.pt")
        print(f"epoch={epoch}/{config['epochs']} train_loss={row['train_loss']:.4f} val_psnr={val_metrics['psnr']:.3f} val_ssim={val_metrics['ssim']:.4f} val_mae={val_metrics['mae']:.4f}")

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    print(f"Saved USDRIFT2D checkpoint and metrics to {output_dir}")


if __name__ == "__main__":
    main()
