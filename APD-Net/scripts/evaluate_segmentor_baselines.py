"""Re-evaluate saved contour and filled-mask segmentors with current metrics."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.hc18_dataset import read_split
from datasets.segmentation_diagnostics import HC18SegmentationDataset
from models.grn_components import UNetSegmentor
from train_segmentor_baseline import evaluate, summarize


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/segmentor_diagnostics.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records = read_split(ROOT / config["split"]["validation"])
    records_by_name = {record.filename: record for record in records}
    cache = ROOT / config["filled_mask_cache"]
    output_root = ROOT / config["output_dir"]
    checkpoint_root = ROOT / config["checkpoint_dir"]
    summaries = []

    for target_mode in config["target_modes"]:
        checkpoint_path = checkpoint_root / f"best_segmentor_{target_mode}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = UNetSegmentor(num_classes=2, channels=tuple(checkpoint["channels"])).to(device)
        model.load_state_dict(checkpoint["model"])
        dataset = HC18SegmentationDataset(
            records,
            int(config["image_size"]),
            target_mode,
            False,
            cache,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        rows = evaluate(
            model,
            loader,
            records_by_name,
            device,
            float(checkpoint["threshold"]),
            target_mode,
            config["boundary_tolerances"],
        )
        summary = {
            "target_mode": target_mode,
            "best_epoch": int(checkpoint["epoch"]),
            "threshold": float(checkpoint["threshold"]),
            **summarize(rows),
            "hc_error_mm_median": float(__import__("numpy").nanmedian([row["hc_error_mm"] for row in rows])),
        }
        mode_dir = output_root / target_mode
        with (mode_dir / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "segmentor_baseline_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    with (output_root / "contour_vs_filled_mask_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    (output_root / "segmentor_baseline_metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
