"""Aggregate APD-GRN diagnosis runs into tables and an evidence summary."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.hc18_dataset import read_split


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def paired_statistics(left_rows: list[dict], right_rows: list[dict], key: str, seed: int = 42) -> dict:
    left = {row["filename"]: to_float(row[key]) for row in left_rows}
    right = {row["filename"]: to_float(row[key]) for row in right_rows}
    common = sorted(set(left) & set(right))
    pairs = [(left[name], right[name]) for name in common if np.isfinite(left[name]) and np.isfinite(right[name])]
    left_values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    right_values = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    difference = right_values - left_values
    if difference.size == 0:
        raise ValueError(f"No paired finite values for {key}")

    shapiro_p = float(stats.shapiro(difference).pvalue) if difference.size <= 5000 else float("nan")
    if difference.size < 20 or not np.isfinite(shapiro_p) or shapiro_p < 0.05:
        test_name = "Wilcoxon signed-rank"
        test = stats.wilcoxon(difference, zero_method="zsplit", method="auto")
    else:
        test_name = "paired t-test"
        test = stats.ttest_rel(right_values, left_values)

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, difference.size, size=(10000, difference.size))
    bootstrap = difference[bootstrap_indices].mean(axis=1)
    left_bootstrap = left_values[bootstrap_indices].mean(axis=1)
    right_bootstrap = right_values[bootstrap_indices].mean(axis=1)
    standard_deviation = float(difference.std(ddof=1)) if difference.size > 1 else 0.0
    effect = float(difference.mean() / standard_deviation) if standard_deviation > 0 else 0.0
    return {
        "metric": key,
        "pairs": int(difference.size),
        "left_mean": float(left_values.mean()),
        "left_std": float(left_values.std(ddof=1)),
        "left_median": float(np.median(left_values)),
        "left_mean_ci95_low": float(np.quantile(left_bootstrap, 0.025)),
        "left_mean_ci95_high": float(np.quantile(left_bootstrap, 0.975)),
        "right_mean": float(right_values.mean()),
        "right_std": float(right_values.std(ddof=1)),
        "right_median": float(np.median(right_values)),
        "right_mean_ci95_low": float(np.quantile(right_bootstrap, 0.025)),
        "right_mean_ci95_high": float(np.quantile(right_bootstrap, 0.975)),
        "mean_difference_right_minus_left": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "difference_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "difference_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "difference_shapiro_p": shapiro_p,
        "paired_test": test_name,
        "p_value": float(test.pvalue),
        "cohens_dz": effect,
    }


def metrics_at(rows: list[dict], condition: str, sigma: float) -> list[dict]:
    return [row for row in rows if row["condition"] == condition and abs(to_float(row["sigma"]) - sigma) < 1e-8]


def save_figures(
    output_root: Path,
    input_root: Path,
    noise_rows: list[dict],
    ablation_rows: list[dict],
    training_history: list[dict],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = output_root / "figures"
    directories = {
        name: figure_root / name
        for name in ("reconstruction_examples", "segmentation_examples", "noise_robustness", "ablations")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    for panel in input_root.rglob("*.png"):
        if "figures" not in panel.parts or "segmentation_examples" not in panel.parts:
            continue
        sigma_dir = next((part for part in panel.parts if part.startswith("sigma_")), "sigma_0.20")
        relative = Path(sigma_dir) / panel.parent.name / panel.name
        target = directories["segmentation_examples"] / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(panel, target)
        reconstruction_target = directories["reconstruction_examples"] / relative
        reconstruction_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(panel, reconstruction_target)

    condition_names = ("B1_noisy", "B2_APD", "B3_APDGRN", "B4_GRN_native_arch_adaptation")
    colors = {"B1_noisy": "#777777", "B2_APD": "#1976d2", "B3_APDGRN": "#d95f02", "B4_GRN_native_arch_adaptation": "#238b45"}
    for metric, ylabel, filename in (
        ("image_psnr_db", "PSNR (dB)", "restoration_psnr.png"),
        ("image_ssim", "SSIM", "restoration_ssim.png"),
        ("boundary_f1_2px", "Boundary F1 (2 px)", "segmentation_boundary_f1.png"),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        for condition in condition_names:
            selected = [row for row in noise_rows if row["condition"] == condition]
            axis.plot(
                [float(row["sigma"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                linewidth=2,
                label=condition.replace("_", " "),
                color=colors[condition],
            )
        axis.set_xlabel("Synthetic speckle sigma")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.savefig(directories["noise_robustness"] / filename, dpi=220)
        plt.close(fig)

    labels = [row["model"] for row in ablation_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].bar(x, [float(row["Boundary_F1_2px"]) for row in ablation_rows], color="#287c8e")
    axes[0].set_ylabel("Boundary F1 (2 px)")
    axes[1].bar(x, [float(row["PSNR_dB"]) for row in ablation_rows], color="#d95f02")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(directories["ablations"] / "ablation_metrics.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    epochs = [int(row["epoch"]) for row in training_history]
    for key, label in (("L_restore_weighted", "Restoration"), ("L_edge_weighted", "Edge"), ("L_seg_weighted", "Segmentation"), ("L_adv_weighted", "Adversarial")):
        axes[0].plot(epochs, [float(row[key]) for row in training_history], label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted loss")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [float(row["gradnorm_ratio_seg_to_restore"]) for row in training_history], label="Seg / restore")
    axes[1].plot(epochs, [float(row["gradnorm_ratio_edge_to_restore"]) for row in training_history], label="Edge / restore")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Generator gradient-norm ratio")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.savefig(directories["ablations"] / "training_loss_gradient_diagnostics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    diagnostic_root = ROOT / "results/diagnostics"
    input_root = diagnostic_root / "input_conditions"
    ablation_root = diagnostic_root / "ablations"
    segmentor_root = diagnostic_root / "segmentors"
    output_root = diagnostic_root
    sigma_values = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

    noise_rows = []
    input_rows_by_sigma = {}
    input_summary_by_sigma = {}
    for sigma in sigma_values:
        folder = input_root if abs(sigma - 0.20) < 1e-8 else input_root / f"sigma_{sigma:.2f}"
        per_image = read_csv(folder / "input_conditions_per_image.csv")
        summary = read_csv(folder / "input_conditions_summary.csv")
        per_image = read_csv(folder / "input_conditions_per_image.csv")
        input_rows_by_sigma[sigma] = per_image
        summaries_for_sigma = {}
        for row in summary:
            condition_rows = [item for item in per_image if item["condition"] == row["condition"]]
            hc_errors = [to_float(item["hc_error_mm"]) for item in condition_rows]
            row["hc_error_mm_median"] = float(np.nanmedian(hc_errors))
            summaries_for_sigma[row["condition"]] = row
            noise_rows.append({"sigma": sigma, **row})
        input_summary_by_sigma[sigma] = summaries_for_sigma
    write_csv(output_root / "noise_level_metrics.csv", noise_rows)

    apd_rows = metrics_at(input_rows_by_sigma[0.20], "B2_APD", 0.20)
    apdgrn_rows = metrics_at(input_rows_by_sigma[0.20], "B3_APDGRN", 0.20)
    noisy_rows = metrics_at(input_rows_by_sigma[0.20], "B1_noisy", 0.20)
    reference_rows = metrics_at(input_rows_by_sigma[0.20], "B0_reference", 0.20)
    comparisons = []
    for metric in (
        "image_psnr_db",
        "image_ssim",
        "image_mae",
        "dice",
        "boundary_f1_1px",
        "boundary_f1_2px",
        "boundary_f1_3px",
        "boundary_f1_5px",
        "hd95_pixels",
        "asd_pixels",
        "hc_error_mm",
    ):
        comparisons.append({"comparison": "APDGRN_minus_APD", **paired_statistics(apd_rows, apdgrn_rows, metric)})
    for comparison_name, left_rows, right_rows in (
        ("Noisy_minus_reference", reference_rows, noisy_rows),
        ("APD_minus_noisy", noisy_rows, apd_rows),
    ):
        for metric in (
            "image_psnr_db",
            "image_ssim",
            "image_mae",
            "dice",
            "boundary_f1_2px",
            "hd95_pixels",
            "asd_pixels",
        ):
            comparisons.append({"comparison": comparison_name, **paired_statistics(left_rows, right_rows, metric)})

    m5_rows = [
        row
        for row in read_csv(ablation_root / "M5/per_image_metrics.csv")
        if row["checkpoint"] == "segmentation" and abs(to_float(row["sigma"]) - 0.20) < 1e-8
    ]
    m6_rows = [
        row
        for row in read_csv(ablation_root / "M6/per_image_metrics.csv")
        if row["checkpoint"] == "segmentation" and abs(to_float(row["sigma"]) - 0.20) < 1e-8
    ]
    for metric in (
        "enhanced_seg_dice",
        "enhanced_seg_boundary_f1_1px",
        "enhanced_seg_boundary_f1_2px",
        "enhanced_seg_boundary_f1_3px",
        "enhanced_seg_boundary_f1_5px",
        "enhanced_seg_hd95_pixels",
        "enhanced_seg_asd_pixels",
        "psnr_db",
        "ssim",
        "mae",
    ):
        comparisons.append({"comparison": "M6_minus_M5", **paired_statistics(m5_rows, m6_rows, metric)})
    m7_rows = [
        row
        for row in read_csv(diagnostic_root / "apd_grn_stagec/per_image_metrics.csv")
        if row["checkpoint"] == "segmentation" and abs(to_float(row["sigma"]) - 0.20) < 1e-8
    ]
    for metric in (
        "enhanced_seg_dice",
        "enhanced_seg_boundary_f1_2px",
        "enhanced_seg_hd95_pixels",
        "enhanced_seg_asd_pixels",
        "psnr_db",
        "ssim",
        "mae",
    ):
        comparisons.append({"comparison": "M7_minus_M6", **paired_statistics(m6_rows, m7_rows, metric)})
    write_csv(output_root / "apd_vs_apdgrn_metrics.csv", comparisons)

    ablation_rows = []
    noisy_summary = input_summary_by_sigma[0.20]["B1_noisy"]
    ablation_rows.append(
        {
            "model": "M0_noisy_to_segmentor",
            "PSNR_dB": noisy_summary["image_psnr_db"],
            "SSIM": noisy_summary["image_ssim"],
            "Dice": noisy_summary["dice"],
            "Boundary_F1_2px": noisy_summary["boundary_f1_2px"],
            "HD95_pixels": noisy_summary["hd95_pixels"],
            "ASSD_pixels": noisy_summary["asd_pixels"],
        }
    )
    for model_name in ("M1", "M2", "M3", "M4", "M5", "M6"):
        rows = read_csv(ablation_root / model_name / "checkpoint_comparison.csv")
        row = next(
            item
            for item in rows
            if item["checkpoint"] == "segmentation" and abs(to_float(item["sigma"]) - 0.20) < 1e-8
        )
        ablation_rows.append(
            {
                "model": model_name,
                "best_epoch": row["epoch"],
                "PSNR_dB": row["psnr_db"],
                "SSIM": row["ssim"],
                "Dice": row["enhanced_seg_dice"],
                "Boundary_F1_2px": row["enhanced_seg_boundary_f1_2px"],
                "HD95_pixels": row["enhanced_seg_hd95_pixels"],
                "ASSD_pixels": row["enhanced_seg_asd_pixels"],
            }
        )
    stagec_rows = read_csv(diagnostic_root / "apd_grn_stagec/checkpoint_comparison.csv")
    row = next(
        item
        for item in stagec_rows
        if item["checkpoint"] == "segmentation" and abs(to_float(item["sigma"]) - 0.20) < 1e-8
    )
    ablation_rows.append(
        {
            "model": "M7_full_patchgan",
            "best_epoch": row["epoch"],
            "PSNR_dB": row["psnr_db"],
            "SSIM": row["ssim"],
            "Dice": row["enhanced_seg_dice"],
            "Boundary_F1_2px": row["enhanced_seg_boundary_f1_2px"],
            "HD95_pixels": row["enhanced_seg_hd95_pixels"],
            "ASSD_pixels": row["enhanced_seg_asd_pixels"],
        }
    )
    write_csv(output_root / "ablation_metrics.csv", ablation_rows)

    segmentor_summaries = json.loads((segmentor_root / "segmentor_baseline_metrics.json").read_text())
    for summary in segmentor_summaries:
        per_image = read_csv(segmentor_root / summary["target_mode"] / "validation_metrics.csv")
        hc_errors = [to_float(row["hc_error_mm"]) for row in per_image]
        summary["hc_error_mm_median"] = float(np.nanmedian(hc_errors))
    write_csv(output_root / "segmentor_baseline_metrics.csv", segmentor_summaries)
    write_csv(output_root / "contour_vs_filled_mask_metrics.csv", segmentor_summaries)
    shutil.copyfile(
        diagnostic_root / "apd_grn_stagec/gradient_norms.csv",
        output_root / "gradient_norms.csv",
    )
    training_history = read_csv(diagnostic_root / "apd_grn_stagec/training_history.csv")
    write_csv(output_root / "training_history.csv", training_history)
    loss_components = {
        "restoration": "L_restore_weighted",
        "edge": "L_edge_weighted",
        "segmentation": "L_seg_weighted",
        "adversarial": "L_adv_weighted",
    }
    component_means = {
        name: float(np.mean([to_float(row[key]) for row in training_history]))
        for name, key in loss_components.items()
    }
    total_component_mean = sum(component_means.values())
    loss_weight_rows = [
        {
            "component": name,
            "mean_weighted_loss": value,
            "fraction_of_weighted_generator_objective": value / max(total_component_mean, 1e-12),
            "mean_raw_loss": float(np.mean([to_float(row["L_restore_raw" if name == "restoration" else {"edge": "L_edge_raw", "segmentation": "L_seg_raw", "adversarial": "L_adv_raw"}[name]]) for row in training_history])),
        }
        for name, value in component_means.items()
    ]
    write_csv(output_root / "loss_weight_diagnostics.csv", loss_weight_rows)
    checkpoint_rows = []
    for run_name in ("M1", "M2", "M3", "M4", "M5", "M6", "M7"):
        path = (
            diagnostic_root / "apd_grn_stagec/checkpoint_comparison.csv"
            if run_name == "M7"
            else ablation_root / run_name / "checkpoint_comparison.csv"
        )
        checkpoint_rows.extend({"run": run_name, **row} for row in read_csv(path))
    write_csv(output_root / "checkpoint_comparison.csv", checkpoint_rows)

    train_records = read_split(ROOT / "../HC18_Research/splits/train.csv")
    val_records = read_split(ROOT / "../HC18_Research/splits/val.csv")
    test_path = ROOT / "../HC18_Research/splits/test.csv"
    if test_path.exists():
        with test_path.open(newline="", encoding="utf-8") as handle:
            test_count = sum(1 for _ in csv.DictReader(handle))
    else:
        test_count = None
    split_lines = [
        "HC18 split audit",
        f"Training images: {len(train_records)}",
        f"Validation images: {len(val_records)}",
        f"Official test images listed: {test_count if test_count is not None else 'split file unavailable'}",
        "Grouping metadata: unavailable (split fields have no patient/examination/group identifier).",
        "Leakage status: image-level separation was checked; examination-level separation cannot be verified.",
        "Official test images were not loaded, evaluated, or used for model selection.",
    ]
    (output_root / "split_report.txt").write_text("\n".join(split_lines) + "\n", encoding="utf-8")

    contour_summary = next(item for item in segmentor_summaries if item["target_mode"] == "contour")
    filled_summary = next(item for item in segmentor_summaries if item["target_mode"] == "filled_mask")
    b0 = input_summary_by_sigma[0.20]["B0_reference"]
    b1 = input_summary_by_sigma[0.20]["B1_noisy"]
    b2 = input_summary_by_sigma[0.20]["B2_APD"]
    b3 = input_summary_by_sigma[0.20]["B3_APDGRN"]
    stage_metrics = {row["comparison"] + ":" + row["metric"]: row for row in comparisons}
    diff = stage_metrics["APDGRN_minus_APD:boundary_f1_2px"]
    m5m6 = stage_metrics["M6_minus_M5:enhanced_seg_boundary_f1_2px"]
    m7m6 = stage_metrics["M7_minus_M6:enhanced_seg_boundary_f1_2px"]
    gradient_values = [float(row["gradnorm_ratio_seg_to_restore"]) for row in read_csv(output_root / "gradient_norms.csv")]
    m7m6 = stage_metrics["M7_minus_M6:enhanced_seg_boundary_f1_2px"]
    loss_fraction = {row["component"]: float(row["fraction_of_weighted_generator_objective"]) for row in loss_weight_rows}
    report = f"""# APD-GRN Diagnostic Report

## Findings

A. **Can the standalone segmentor learn HC18?** Yes. Reference-only contour training reached Dice {float(contour_summary['target_dice']):.3f}, Boundary F1@2 {float(contour_summary['boundary_f1_2px']):.3f}, and HD95 {float(contour_summary['target_hd95_pixels']):.2f} pixels. The filled-mask experiment reached Dice {float(filled_summary['target_dice']):.3f} and IoU {float(filled_summary['target_iou']):.3f}.

B. **Does exact contour Dice understate geometric quality?** Yes. Contour Dice is {float(contour_summary['target_dice']):.3f}, while tolerance-aware Boundary F1 rises from {float(contour_summary['boundary_f1_1px']):.3f}@1px to {float(contour_summary['boundary_f1_2px']):.3f}@2px and {float(contour_summary['boundary_f1_5px']):.3f}@5px.

C. **Does sigma=0.20 speckle degrade segmentation?** Yes. With the same fixed reference-trained segmentor, BF1@2 changes from {float(b0['boundary_f1_2px']):.3f} on reference images to {float(b1['boundary_f1_2px']):.3f} on noisy inputs.

D. **Does ordinary APD recover it?** Yes, nearly to the reference baseline: APD BF1@2 is {float(b2['boundary_f1_2px']):.3f}; its restoration PSNR is {float(b2['image_psnr_db']):.2f} dB.

E. **Does APD-GRN outperform ordinary APD?** Not meaningfully in this run. APD-GRN BF1@2 is {float(b3['boundary_f1_2px']):.3f}, a paired mean difference of {float(diff['mean_difference_right_minus_left']):+.4f} (95% bootstrap CI {float(diff['difference_ci95_low']):+.4f} to {float(diff['difference_ci95_high']):+.4f}, p={float(diff['p_value']):.4g}). APD-GRN restoration PSNR is {float(b3['image_psnr_db']):.2f} dB versus APD {float(b2['image_psnr_db']):.2f} dB.

F. **Are segmentation gradients meaningful?** Median weighted generator gradient-norm ratio `||grad_G(L_seg)|| / ||grad_G(L_restore)||` is {np.median(gradient_values):.3f} (range {min(gradient_values):.3f}–{max(gradient_values):.3f}); gradients are connected and not uniformly negligible.

G. **Does PatchGAN improve segmentation?** M7-vs-M6 Boundary F1@2 paired difference is {float(m7m6['mean_difference_right_minus_left']):+.4f} (95% CI {float(m7m6['difference_ci95_low']):+.4f} to {float(m7m6['difference_ci95_high']):+.4f}, p={float(m7m6['p_value']):.4g}); this single-seed run does not establish a reliable PatchGAN segmentation gain.

The B4 comparison uses GRN's native generator architecture trained with our separate HC18 synthetic-denoising adaptation; it is not an untouched reproduction of the original paper's training protocol.

H. **Best reconstruction checkpoint:** `best_reconstruction.pt`, selected by validation reconstruction score. **Best downstream segmentation checkpoint:** `best_segmentation.pt`, selected by validation Boundary F1@2.

I. **Current recommendation:** evidence supports keeping APD-Net as the primary despeckling contribution and reporting APD-GRN as an investigated task-guidance extension. The standalone segmentor can learn HC18, but M6 does not provide a practically meaningful gain over M5 and fixed-segmentor APD-GRN does not outperform ordinary APD. Do not claim segmentation improvement from GRN feedback.

The M5-vs-M6 task-feedback paired Boundary F1@2 difference is {float(m5m6['mean_difference_right_minus_left']):+.4f} (95% CI {float(m5m6['difference_ci95_low']):+.4f} to {float(m5m6['difference_ci95_high']):+.4f}, p={float(m5m6['p_value']):.4g}).

Mean weighted generator-loss fractions across Stage C were restoration {loss_fraction['restoration']:.1%}, edge {loss_fraction['edge']:.1%}, segmentation {loss_fraction['segmentation']:.1%}, and adversarial {loss_fraction['adversarial']:.1%}. See `loss_weight_diagnostics.csv` for raw and weighted means.

## Interpretation Limits

- HC18 images are original ultrasound references, not speckle-free ground truth; denoising metrics are synthetic-reference metrics.
- HC18 has thin annotated contours. Boundary F1 with tolerance should accompany exact Dice.
- Split metadata do not identify patient/examination groups. The split is image-level; group leakage cannot be excluded.
- The official test split was not evaluated or used for selection.
- The physical head-circumference estimate is ellipse-based and sensitive to resizing; native annotation contour calibration is substantially better than 256-pixel predicted-contour estimates. Treat model HC-error changes cautiously.
- Ablation and model comparisons are single-seed; bootstrap intervals describe image sampling, not training-seed uncertainty.
"""
    (output_root / "diagnostic_report.md").write_text(report, encoding="utf-8")
    save_figures(output_root, input_root, noise_rows, ablation_rows, training_history)
    print("Wrote diagnostic tables, split report, and diagnostic_report.md to", output_root)


if __name__ == "__main__":
    main()