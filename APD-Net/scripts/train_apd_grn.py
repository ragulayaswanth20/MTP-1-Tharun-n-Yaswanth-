"""Train task-guided APD-GRN on HC18 with synthetic multiplicative speckle."""
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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.hc18_dataset import HC18Dataset, read_split
from losses.total_loss import APDGRNLoss, make_edge_target, segmentation_loss
from metrics.apd_grn_metrics import boundary_f1, contour_metrics, head_circumference_mm, image_metrics
from models.apd_grn import APDGRN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(config: dict) -> APDGRN:
    ablation = config["ablation"]
    return APDGRN(
        base_channels=int(config["base_channels"]),
        num_classes=int(config["num_classes"]),
        use_spatial=bool(ablation["spatial_branch"]),
        use_frequency=bool(ablation["frequency_branch"]),
        use_attention=bool(ablation["anatomy_attention"]),
        use_edge=bool(ablation["edge_head"]),
        use_edge_feedback=bool(ablation["edge_feedback"]),
        use_segmentation_feedback=bool(ablation["segmentation_feedback"]),
        use_patchgan=bool(ablation["patchgan"]),
        segmentor_channels=tuple(int(width) for width in config["segmentor_channels"]),
    )


def make_objective(config: dict) -> APDGRNLoss:
    loss_config = config["loss"]
    ablation = config["ablation"]
    edge_enabled = ablation["edge_head"] and ablation["edge_supervision"]
    weights = {
        "restore": float(loss_config["lambda_restore"]),
        "l1": float(loss_config["lambda_l1"]),
        "ssim": float(loss_config["lambda_ssim"]),
        "frequency": float(loss_config["lambda_frequency"]),
        "edge": float(loss_config["lambda_edge"]) if edge_enabled else 0.0,
        "edge_dice": float(loss_config["lambda_edge_dice"]),
        "edge_positive_weight": float(loss_config["edge_positive_weight"]),
        "seg": float(loss_config["lambda_seg"]) if ablation["segmentation_feedback"] else 0.0,
        "seg_ce": float(loss_config["lambda_seg_ce"]),
        "seg_foreground_weight": float(loss_config["segmentation_foreground_weight"]),
        "adv": float(loss_config["lambda_adv"]) if ablation["patchgan"] else 0.0,
    }
    return APDGRNLoss(weights, edge_target_mode=loss_config["edge_target_mode"])


def sample_training_noise(reference: torch.Tensor, noise_config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    minimum = float(noise_config["train_sigma_min"])
    maximum = float(noise_config["train_sigma_max"])
    sigma = torch.empty((reference.shape[0], 1, 1, 1), device=reference.device).uniform_(minimum, maximum)
    noisy = (reference * (1.0 + sigma * torch.randn_like(reference))).clamp(0.0, 1.0)
    return noisy, sigma.flatten()


def fixed_noise(reference: torch.Tensor, sigma: float, seed: int, offset: int) -> torch.Tensor:
    noise = []
    for index in range(reference.shape[0]):
        generator = torch.Generator(device=reference.device).manual_seed(seed + offset + index)
        noise.append(torch.randn(reference[index].shape, generator=generator, device=reference.device))
    return (reference * (1.0 + sigma * torch.stack(noise))).clamp(0.0, 1.0)


def mean_finite(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def summarize_rows(rows: list[dict]) -> dict[str, float | int]:
    keys = [key for key in rows[0] if key != "filename"]
    result: dict[str, float | int] = {"images": len(rows)}
    result.update({key: mean_finite(rows, key) for key in keys})
    return result


def image_panel(images: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    cell_size = 256
    title_height = 24
    canvas = Image.new("RGB", (3 * cell_size, 3 * (cell_size + title_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, array) in enumerate(images):
        row, column = divmod(index, 3)
        image = Image.fromarray(np.asarray(np.clip(array, 0.0, 1.0) * 255.0, dtype=np.uint8)).convert("RGB")
        x = column * cell_size
        y = row * (cell_size + title_height)
        canvas.paste(image, (x, y + title_height))
        draw.text((x + 4, y + 4), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_examples(
    batch: dict,
    output: dict,
    edge_target: torch.Tensor,
    output_dir: Path,
    saved: int,
    maximum: int,
) -> int:
    reference = batch["image"][:, 0].cpu().numpy()
    mask = batch["mask"][:, 0].cpu().numpy()
    denoised = output["denoised"][:, 0].detach().cpu().numpy()
    residual = output["residual"][:, 0].detach().cpu().numpy()
    edge = torch.sigmoid(output["edge"][:, 0]).detach().cpu().numpy()
    original_seg = output["reference_segmentation"].argmax(1).detach().cpu().numpy()
    enhanced_seg = output["enhanced_segmentation"].argmax(1).detach().cpu().numpy()
    edge_gt = edge_target[:, 0].cpu().numpy()
    noisy = batch["noisy"][:, 0].cpu().numpy()

    for index, name in enumerate(batch["filename"]):
        if saved >= maximum:
            break
        residual_map = np.abs(residual[index])
        residual_map /= max(float(residual_map.max()), 1e-8)
        panels = [
            ("Reference image", reference[index]),
            ("Synthetic noisy input", noisy[index]),
            ("APD-GRN denoised", denoised[index]),
            ("Absolute residual", residual_map),
            ("Predicted boundary", edge[index]),
            ("Reference boundary", edge_gt[index]),
            ("S(reference)", original_seg[index]),
            ("S(enhanced)", enhanced_seg[index]),
            ("Ground-truth contour", mask[index]),
        ]
        image_panel(panels, output_dir / f"{Path(name).stem}_apd_grn.png")
        saved += 1
    return saved


def evaluate_model(
    model: APDGRN,
    loader: DataLoader,
    records_by_name: dict,
    objective: APDGRNLoss,
    device: torch.device,
    sigma: float,
    seed: int,
    save_dir: Path | None = None,
    num_examples: int = 0,
) -> list[dict]:
    model.eval()
    rows = []
    saved_examples = 0
    offset = 0
    with torch.no_grad():
        for batch in loader:
            reference = batch["image"].to(device)
            target = batch["mask"].to(device)
            noisy = fixed_noise(reference, sigma, seed, offset)
            output = model(noisy, reference)
            edge_target = make_edge_target(target, objective.edge_target_mode)
            noisy_np = noisy[:, 0].cpu().numpy()
            reference_np = reference[:, 0].cpu().numpy()
            denoised_np = output["denoised"][:, 0].cpu().numpy()
            edge_np = torch.sigmoid(output["edge"][:, 0]).cpu().numpy() >= 0.5
            mask_np = target[:, 0].cpu().numpy() > 0
            edge_gt_np = edge_target[:, 0].cpu().numpy() > 0
            original_seg_np = output["reference_segmentation"].argmax(1).cpu().numpy() == 1
            enhanced_seg_np = output["enhanced_segmentation"].argmax(1).cpu().numpy() == 1
            residual_np = output["residual"][:, 0].cpu().numpy()

            for index, filename in enumerate(batch["filename"]):
                record = records_by_name[filename]
                with Image.open(record.image_path) as source:
                    original_shape = (source.height, source.width)
                row = {"filename": filename, "sigma": sigma}
                row.update({f"noisy_{key}": value for key, value in image_metrics(noisy_np[index], reference_np[index]).items()})
                row.update(image_metrics(denoised_np[index], reference_np[index]))
                row.update({f"original_seg_{key}": value for key, value in contour_metrics(original_seg_np[index], mask_np[index]).items()})
                row.update({f"enhanced_seg_{key}": value for key, value in contour_metrics(enhanced_seg_np[index], mask_np[index]).items()})
                row.update({f"edge_{key}": value for key, value in contour_metrics(edge_np[index], edge_gt_np[index]).items()})
                for tolerance in (1, 2, 3, 5):
                    row[f"original_seg_boundary_f1_{tolerance}px"] = boundary_f1(
                        original_seg_np[index], mask_np[index], tolerance
                    )
                    row[f"enhanced_seg_boundary_f1_{tolerance}px"] = boundary_f1(
                        enhanced_seg_np[index], mask_np[index], tolerance
                    )
                predicted_hc = head_circumference_mm(
                    enhanced_seg_np[index],
                    record.pixel_size,
                    original_shape,
                    reference_np[index].shape,
                )
                row["hc_predicted_mm"] = predicted_hc
                row["hc_error_mm"] = abs(predicted_hc - record.hc_mm) if np.isfinite(predicted_hc) else float("nan")
                rows.append(row)

            if save_dir is not None and saved_examples < num_examples:
                example_batch = dict(batch)
                example_batch["noisy"] = noisy.cpu()
                saved_examples = save_examples(
                    example_batch,
                    output,
                    edge_target.cpu(),
                    save_dir,
                    saved_examples,
                    num_examples,
                )
            offset += reference.shape[0]
    return rows


def set_requires_grad(module: torch.nn.Module | None, enabled: bool) -> None:
    if module is not None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared_norm = sum(gradient.detach().pow(2).sum() for gradient in gradients if gradient is not None)
    return float(torch.sqrt(squared_norm).item()) if isinstance(squared_norm, torch.Tensor) else 0.0


def generator_gradient_diagnostics(losses: dict[str, torch.Tensor], model: APDGRN) -> dict[str, float]:
    parameters = [parameter for parameter in model.generator.parameters() if parameter.requires_grad]
    terms = {
        "restore_raw": "L_restore_raw",
        "restore_weighted": "L_restore_weighted",
        "seg_raw": "L_seg_enhanced",
        "seg_weighted": "L_seg_weighted",
        "edge_raw": "L_edge",
        "edge_weighted": "L_edge_weighted",
        "adv_raw": "L_adv",
        "adv_weighted": "L_adv_weighted",
    }
    norms = {f"gradnorm_{name}": gradient_norm(losses[key], parameters) for name, key in terms.items()}
    denominator = max(norms["gradnorm_restore_weighted"], 1e-12)
    norms["gradnorm_ratio_seg_to_restore"] = norms["gradnorm_seg_weighted"] / denominator
    norms["gradnorm_ratio_edge_to_restore"] = norms["gradnorm_edge_weighted"] / denominator
    return norms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/apd_grn.yaml")
    parser.add_argument("--ablation", choices=("M1", "M2", "M3", "M4", "M5", "M6", "M7"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.ablation:
        presets = {
            "M1": {"spatial_branch": True, "frequency_branch": False, "anatomy_attention": False, "edge_head": False, "edge_feedback": False, "edge_supervision": False, "segmentation_feedback": False, "patchgan": False},
            "M2": {"spatial_branch": False, "frequency_branch": True, "anatomy_attention": False, "edge_head": False, "edge_feedback": False, "edge_supervision": False, "segmentation_feedback": False, "patchgan": False},
            "M3": {"spatial_branch": True, "frequency_branch": True, "anatomy_attention": False, "edge_head": False, "edge_feedback": False, "edge_supervision": False, "segmentation_feedback": False, "patchgan": False},
            "M4": {"spatial_branch": True, "frequency_branch": True, "anatomy_attention": True, "edge_head": False, "edge_feedback": False, "edge_supervision": False, "segmentation_feedback": False, "patchgan": False},
            "M5": {"spatial_branch": True, "frequency_branch": True, "anatomy_attention": True, "edge_head": True, "edge_feedback": True, "edge_supervision": True, "segmentation_feedback": False, "patchgan": False},
            "M6": {"spatial_branch": True, "frequency_branch": True, "anatomy_attention": True, "edge_head": True, "edge_feedback": True, "edge_supervision": True, "segmentation_feedback": True, "patchgan": False},
            "M7": {"spatial_branch": True, "frequency_branch": True, "anatomy_attention": True, "edge_head": True, "edge_feedback": True, "edge_supervision": True, "segmentation_feedback": True, "patchgan": True},
        }
        config["ablation"].update(presets[args.ablation])
        config["train_segmentor"] = True
    if args.run_name:
        config["results_dir"] = f"results/diagnostics/ablations/{args.run_name}"
        config["checkpoint_reconstruction"] = f"checkpoints/diagnostics/ablations/{args.run_name}_best_reconstruction.pt"
        config["checkpoint_segmentation"] = f"checkpoints/diagnostics/ablations/{args.run_name}_best_segmentation.pt"
        config["checkpoint_joint"] = f"checkpoints/diagnostics/ablations/{args.run_name}_best_joint.pt"
    seed_everything(int(config["seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_records = read_split(ROOT / config["data"]["train_split"])
    val_records = read_split(ROOT / config["data"]["val_split"])
    if {record.filename for record in train_records} & {record.filename for record in val_records}:
        raise ValueError("HC18 training and validation splits overlap")
    image_size = int(config["image_size"])
    train_loader = DataLoader(
        HC18Dataset(train_records, image_size),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        HC18Dataset(val_records, image_size),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = make_model(config).to(device)
    initial_checkpoint_path = config.get("initialize_generator_from")
    if initial_checkpoint_path:
        initial_checkpoint = torch.load(ROOT / initial_checkpoint_path, map_location=device, weights_only=False)
        model.generator.load_state_dict(initial_checkpoint.get("generator", initial_checkpoint.get("model")))
        print(f"Initialized APD generator from {initial_checkpoint_path}")
    initial_segmentor_path = config.get("initialize_segmentor_from")
    if initial_segmentor_path:
        initial_segmentor = torch.load(ROOT / initial_segmentor_path, map_location=device, weights_only=False)
        model.segmentor.load_state_dict(initial_segmentor.get("segmentor", initial_segmentor.get("model")))
        print(f"Initialized segmentor from {initial_segmentor_path}")
    objective = make_objective(config)
    learning_rates = config.get("learning_rates", {})
    learning_rate = float(config.get("learning_rate", 0.0002))
    optimizer_g = torch.optim.AdamW(model.generator.parameters(), lr=float(learning_rates.get("generator", learning_rate)))
    train_segmentor = bool(config.get("train_segmentor", True))
    optimizer_s = (
        torch.optim.AdamW(model.segmentor.parameters(), lr=float(learning_rates.get("segmentor", learning_rate)))
        if train_segmentor
        else None
    )
    optimizer_d = (
        torch.optim.AdamW(model.discriminator.parameters(), lr=float(learning_rates.get("discriminator", learning_rate)))
        if model.discriminator is not None
        else None
    )
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5)
    scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=10, gamma=0.5) if optimizer_s else None
    scheduler_d = (
        torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.5)
        if optimizer_d is not None
        else None
    )

    checkpoint_paths = {
        "reconstruction": ROOT / config.get("checkpoint_reconstruction", config.get("checkpoint", "checkpoints/apd_grn_best.pt")),
        "segmentation": ROOT / config.get("checkpoint_segmentation", config.get("checkpoint", "checkpoints/apd_grn_best.pt")),
        "joint": ROOT / config.get("checkpoint_joint", config.get("checkpoint", "checkpoints/apd_grn_best.pt")),
    }
    for path in checkpoint_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    records_by_name = {record.filename: record for record in val_records}
    selection_sigma = float(config.get("checkpoint_selection_sigma", 0.20))
    history = []
    best_scores = {name: float("-inf") for name in checkpoint_paths}
    gradient_rows = []

    print(f"APD-GRN HC18: train={len(train_records)} val={len(val_records)} device={device}")
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        accumulated = {}
        epoch_gradient_rows = []
        for batch_index, batch in enumerate(train_loader):
            reference = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            noisy, _sigmas = sample_training_noise(reference, config["noise"])

            if optimizer_d is not None:
                optimizer_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_for_d = model.generator(noisy)["denoised"]
                real_logits = model.discriminator(reference)
                fake_logits = model.discriminator(fake_for_d.detach())
                loss_d = objective.discriminator_loss(real_logits, fake_logits)
                loss_d.backward()
                optimizer_d.step()
            else:
                loss_d = reference.new_zeros(())

            optimizer_g.zero_grad(set_to_none=True)
            set_requires_grad(model.segmentor, False)
            set_requires_grad(model.discriminator, False)
            generator_outputs = model(noisy, reference)
            loss_g, generator_losses = objective(generator_outputs, reference, target)
            diagnostic_steps = int(config.get("gradient_diagnostic_steps_per_epoch", 0))
            if batch_index < diagnostic_steps:
                epoch_gradient_rows.append(
                    generator_gradient_diagnostics(generator_losses, model)
                )
            loss_g.backward()
            optimizer_g.step()
            set_requires_grad(model.segmentor, True)
            set_requires_grad(model.discriminator, True)

            if optimizer_s is not None:
                optimizer_s.zero_grad(set_to_none=True)
                with torch.no_grad():
                    enhanced_for_segmentor = model.generator(noisy)["denoised"]
                original_logits = model.segmentor(reference)
                enhanced_logits = model.segmentor(enhanced_for_segmentor.detach())
                segmentation_losses = {
                    "L_seg_original": segmentation_loss(
                        original_logits,
                        target,
                        float(config["loss"]["lambda_seg_ce"]),
                        float(config["loss"]["segmentation_foreground_weight"]),
                    ),
                    "L_seg_enhanced": segmentation_loss(
                        enhanced_logits,
                        target,
                        float(config["loss"]["lambda_seg_ce"]),
                        float(config["loss"]["segmentation_foreground_weight"]),
                    ),
                }
                loss_s = 0.5 * (segmentation_losses["L_seg_original"] + segmentation_losses["L_seg_enhanced"])
                loss_s.backward()
                optimizer_s.step()
            else:
                segmentation_losses = {}
                loss_s = reference.new_zeros(())

            batch_losses = dict(generator_losses)
            batch_losses.update(segmentation_losses)
            batch_losses["L_S"] = loss_s
            batch_losses["L_D"] = loss_d
            batch_losses["L_edge_raw"] = generator_losses["L_edge"]
            batch_losses["L_seg_raw"] = generator_losses["L_seg_enhanced"]
            batch_losses["L_adv_raw"] = generator_losses["L_adv"]
            for name, value in batch_losses.items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value.detach().item())

        scheduler_g.step()
        if scheduler_s is not None:
            scheduler_s.step()
        if scheduler_d is not None:
            scheduler_d.step()
        train_summary = {name: value / max(len(train_loader), 1) for name, value in accumulated.items()}
        gradient_summary = {
            name: float(np.mean([row[name] for row in epoch_gradient_rows]))
            for name in epoch_gradient_rows[0]
        } if epoch_gradient_rows else {}

        val_rows = evaluate_model(
            model,
            val_loader,
            records_by_name,
            objective,
            device,
            selection_sigma,
            int(config["noise"]["evaluation_seed"]),
        )
        val_summary = summarize_rows(val_rows)
        criteria = config.get("checkpoint_criteria", {})
        segmentation_metric = criteria.get("segmentation_metric", "enhanced_seg_boundary_f1_2px")
        psnr_min = float(criteria.get("reconstruction_psnr_min", 25.0))
        psnr_max = float(criteria.get("reconstruction_psnr_max", 40.0))
        normalized_psnr = float(np.clip((val_summary["psnr_db"] - psnr_min) / max(psnr_max - psnr_min, 1e-8), 0.0, 1.0))
        reconstruction_score = val_summary["psnr_db"] + 5.0 * val_summary["ssim"]
        segmentation_score = val_summary[segmentation_metric]
        reconstruction_weight = float(criteria.get("joint_reconstruction_weight", 0.5))
        segmentation_weight = float(criteria.get("joint_segmentation_weight", 0.5))
        joint_score = (
            reconstruction_weight * normalized_psnr + segmentation_weight * segmentation_score
        ) / max(reconstruction_weight + segmentation_weight, 1e-8)
        epoch_record = {
            "epoch": epoch,
            **train_summary,
            **gradient_summary,
            "validation_sigma": selection_sigma,
            "reconstruction_score": reconstruction_score,
            "segmentation_score": segmentation_score,
            "joint_score": joint_score,
            **val_summary,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch}/{config['epochs']} G={train_summary['L_total_G']:.4f} "
            f"D={train_summary['L_D']:.4f} S={train_summary['L_S']:.4f} "
            f"val_PSNR={val_summary['psnr_db']:.3f} "
            f"val_boundary_F1@2={val_summary['enhanced_seg_boundary_f1_2px']:.4f} "
            f"grad_seg/rest={gradient_summary.get('gradnorm_ratio_seg_to_restore', float('nan')):.3e}"
        )
        payload = {
            "generator": model.generator.state_dict(),
            "segmentor": model.segmentor.state_dict(),
            "discriminator": model.discriminator.state_dict() if model.discriminator is not None else None,
            "epoch": epoch,
            "config": config,
            "validation_sigma": selection_sigma,
            "validation_metrics": val_summary,
            "selection_scores": {
                "reconstruction": reconstruction_score,
                "segmentation": segmentation_score,
                "joint": joint_score,
            },
        }
        for name, score in (
            ("reconstruction", reconstruction_score),
            ("segmentation", segmentation_score),
            ("joint", joint_score),
        ):
            if score > best_scores[name]:
                best_scores[name] = score
                torch.save(payload, checkpoint_paths[name])
        (results_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        with (results_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        if gradient_summary:
            gradient_rows.append({"epoch": epoch, **gradient_summary})
            with (results_dir / "gradient_norms.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=gradient_rows[0].keys())
                writer.writeheader()
                writer.writerows(gradient_rows)

    evaluation = {}
    checkpoint_rows = []
    all_rows = []
    example_sigma = float(config["example_sigma"])
    examples_dir = results_dir / f"examples_sigma_{example_sigma:.2f}"
    for checkpoint_name, checkpoint_path in checkpoint_paths.items():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.generator.load_state_dict(checkpoint["generator"])
        model.segmentor.load_state_dict(checkpoint["segmentor"])
        checkpoint_evaluation = {}
        for sigma in config["noise"]["evaluation_sigmas"]:
            sigma = float(sigma)
            save_dir = (
                examples_dir
                if checkpoint_name == "joint" and abs(sigma - example_sigma) < 1e-8
                else None
            )
            rows = evaluate_model(
                model,
                val_loader,
                records_by_name,
                objective,
                device,
                sigma,
                int(config["noise"]["evaluation_seed"]),
                save_dir=save_dir,
                num_examples=int(config["num_visual_examples"]) if save_dir is not None else 0,
            )
            summary = summarize_rows(rows)
            checkpoint_evaluation[f"{sigma:.2f}"] = summary
            checkpoint_rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "epoch": int(checkpoint["epoch"]),
                    "sigma": sigma,
                    **summary,
                }
            )
            for row in rows:
                all_rows.append({"checkpoint": checkpoint_name, **row})
        evaluation[checkpoint_name] = checkpoint_evaluation

    (results_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "dataset": "HC18",
                "reference_description": "original HC18 ultrasound image; not speckle-free ground truth",
                "validation_images": len(val_records),
                "noise_model": config["noise"]["model"],
                "training_sigma_range": [config["noise"]["train_sigma_min"], config["noise"]["train_sigma_max"]],
                "evaluation": evaluation,
                "checkpoint_paths": {key: str(value) for key, value in checkpoint_paths.items()},
                "training_history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (results_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    with (results_dir / "checkpoint_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=checkpoint_rows[0].keys())
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    print(json.dumps(evaluation, indent=2))
    print(f"Checkpoints: {checkpoint_paths}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()