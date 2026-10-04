"""Pretrain APD-Net for variable-sigma HC18 reconstruction and contour edges."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.hc18_dataset import HC18Dataset, read_split
from losses.total_loss import dice_loss, make_edge_target, ssim_loss
from metrics.apd_grn_metrics import image_metrics
from models.apdnet import APDNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def noise_batch(reference: torch.Tensor, minimum: float, maximum: float) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas = torch.empty((reference.shape[0], 1, 1, 1), device=reference.device).uniform_(minimum, maximum)
    noisy = (reference * (1.0 + sigmas * torch.randn_like(reference))).clamp(0.0, 1.0)
    return noisy, sigmas.flatten()


def fixed_noise(reference: torch.Tensor, sigma: float, seed: int, offset: int) -> torch.Tensor:
    noise = []
    for index in range(reference.shape[0]):
        generator = torch.Generator(device=reference.device).manual_seed(seed + offset + index)
        noise.append(torch.randn(reference[index].shape, generator=generator, device=reference.device))
    return (reference * (1.0 + sigma * torch.stack(noise))).clamp(0.0, 1.0)


def restoration_loss(output: dict, reference: torch.Tensor, target: torch.Tensor, config: dict):
    l1 = F.l1_loss(output["denoised"], reference)
    ssim = ssim_loss(output["denoised"], reference)
    frequency = F.l1_loss(
        torch.log1p(torch.fft.fft2(output["denoised"]).abs()),
        torch.log1p(torch.fft.fft2(reference).abs()),
    )
    edge_target = make_edge_target(target, "annotation")
    edge_bce = F.binary_cross_entropy_with_logits(
        output["edge"],
        edge_target,
        pos_weight=output["edge"].new_tensor(float(config["edge_positive_weight"])),
    )
    edge = edge_bce + float(config["lambda_edge_dice"]) * dice_loss(output["edge"], edge_target)
    weighted = (
        float(config["lambda_l1"]) * l1
        + float(config["lambda_ssim"]) * ssim
        + float(config["lambda_frequency"]) * frequency
        + float(config["lambda_edge"]) * edge
    )
    return weighted, {
        "L_L1": l1,
        "L_SSIM": ssim,
        "L_frequency": frequency,
        "L_edge": edge,
        "L_total": weighted,
    }


def evaluate(model, loader, device, sigma: float, seed: int) -> dict:
    model.eval()
    rows = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            reference = batch["image"].to(device)
            noisy = fixed_noise(reference, sigma, seed, offset)
            denoised = model(noisy)["denoised"]
            for filename, prediction, target in zip(
                batch["filename"], denoised[:, 0].cpu().numpy(), reference[:, 0].cpu().numpy()
            ):
                rows.append({"filename": filename, **image_metrics(prediction, target)})
            offset += reference.shape[0]
    values = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in ("psnr_db", "ssim", "mae")}
    return {
        "images": len(rows),
        **{key: float(scores.mean()) for key, scores in values.items()},
        **{f"{key}_std": float(scores.std(ddof=1)) for key, scores in values.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/apd_reconstruction_pretrain.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_records = read_split(ROOT / config["data"]["train_split"])
    val_records = read_split(ROOT / config["data"]["val_split"])
    train_loader = DataLoader(
        HC18Dataset(train_records, int(config["image_size"])),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        HC18Dataset(val_records, int(config["image_size"])),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model = APDNet(int(config["base_channels"])).to(device)
    pretrained = torch.load(ROOT / config["initialize_from"], map_location=device, weights_only=False)
    model.load_state_dict(pretrained["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    checkpoint_path = ROOT / config["checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    best_psnr = float("-inf")
    bad_epochs = 0
    history = []
    print(f"APD reconstruction pretrain: train={len(train_records)} val={len(val_records)} device={device}")

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        epoch_losses = {}
        for batch in train_loader:
            reference = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            noisy, _sigmas = noise_batch(
                reference,
                float(config["noise"]["sigma_min"]),
                float(config["noise"]["sigma_max"]),
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(noisy)
            loss, parts = restoration_loss(output, reference, target, config["loss"])
            loss.backward()
            optimizer.step()
            for name, value in parts.items():
                epoch_losses[name] = epoch_losses.get(name, 0.0) + float(value.detach())

        train_summary = {name: value / max(len(train_loader), 1) for name, value in epoch_losses.items()}
        validation = evaluate(
            model,
            val_loader,
            device,
            float(config["noise"]["validation_sigma"]),
            int(config["noise"]["validation_seed"]),
        )
        history_row = {"epoch": epoch, **train_summary, "validation_sigma": config["noise"]["validation_sigma"], **validation}
        history.append(history_row)
        print(
            f"epoch={epoch}/{config['epochs']} L={train_summary['L_total']:.4f} "
            f"val_PSNR={validation['psnr_db']:.3f} val_SSIM={validation['ssim']:.4f}"
        )
        if validation["psnr_db"] > best_psnr:
            best_psnr = validation["psnr_db"]
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": config, "validation_metrics": validation},
                checkpoint_path,
            )
        else:
            bad_epochs += 1
        with (results_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        if bad_epochs >= int(config["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch}; best validation PSNR={best_psnr:.3f}")
            break

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    sigma_results = {}
    all_rows = []
    for sigma in config["noise"]["evaluation_sigmas"]:
        sigma = float(sigma)
        rows = []
        model.eval()
        offset = 0
        with torch.no_grad():
            for batch in val_loader:
                reference = batch["image"].to(device)
                noisy = fixed_noise(reference, sigma, int(config["noise"]["validation_seed"]), offset)
                denoised = model(noisy)["denoised"]
                for filename, prediction, target in zip(
                    batch["filename"], denoised[:, 0].cpu().numpy(), reference[:, 0].cpu().numpy()
                ):
                    rows.append({"filename": filename, "sigma": sigma, **image_metrics(prediction, target)})
                offset += reference.shape[0]
        sigma_results[f"{sigma:.2f}"] = {
            "images": len(rows),
            **{metric: float(np.mean([row[metric] for row in rows])) for metric in ("psnr_db", "ssim", "mae")},
        }
        all_rows.extend(rows)
    (results_dir / "evaluation.json").write_text(
        json.dumps({"best_epoch": int(best["epoch"]), "evaluation": sigma_results}, indent=2),
        encoding="utf-8",
    )
    with (results_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print("saved reconstruction checkpoint", checkpoint_path)


if __name__ == "__main__":
    main()