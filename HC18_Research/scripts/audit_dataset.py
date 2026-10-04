"""Audit the local HC-18 dataset without modifying source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not path.stem.endswith("_Annotation")
    )


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_images(files: list[Path], mask_suffix: str | None = None) -> dict:
    dimensions = Counter()
    modes = Counter()
    intensity_min = []
    intensity_max = []
    intensity_mean = []
    mask_areas = []
    corrupted = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    for path in files:
        try:
            with Image.open(path) as image:
                image.load()
                array = np.asarray(image.convert("L"))
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                intensity_min.append(int(array.min()))
                intensity_max.append(int(array.max()))
                intensity_mean.append(float(array.mean()))
                hashes[file_hash(path)].append(path.name)

            if mask_suffix:
                mask_path = path.with_name(f"{path.stem}{mask_suffix}")
                if mask_path.exists():
                    with Image.open(mask_path) as mask_image:
                        mask = np.asarray(mask_image.convert("L")) > 0
                        mask_areas.append(float(mask.mean()))
        except Exception as error:  # pragma: no cover - depends on corrupt files
            corrupted.append({"filename": path.name, "error": str(error)})

    duplicate_groups = [names for names in hashes.values() if len(names) > 1]
    return {
        "count": len(files),
        "dimensions": dict(dimensions),
        "modes": dict(modes),
        "intensity_min": min(intensity_min) if intensity_min else None,
        "intensity_max": max(intensity_max) if intensity_max else None,
        "intensity_mean": {
            "min": min(intensity_mean) if intensity_mean else None,
            "max": max(intensity_mean) if intensity_mean else None,
            "mean": float(np.mean(intensity_mean)) if intensity_mean else None,
        },
        "mask_area_fraction": {
            "min": min(mask_areas) if mask_areas else None,
            "max": max(mask_areas) if mask_areas else None,
            "mean": float(np.mean(mask_areas)) if mask_areas else None,
        },
        "corrupted_files": corrupted,
        "exact_duplicate_groups": duplicate_groups,
    }


def make_figures(files: list[Path], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sample = files[:3]
    for index, path in enumerate(sample):
        image = np.asarray(Image.open(path).convert("L"))
        mask_path = path.with_name(f"{path.stem}_Annotation.png")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0 if mask_path.exists() else None
        figure, axes = plt.subplots(1, 3 if mask is not None else 1, figsize=(12, 4))
        axes = np.atleast_1d(axes)
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Ultrasound")
        axes[0].axis("off")
        if mask is not None:
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Annotation")
            axes[1].axis("off")
            axes[2].imshow(image, cmap="gray")
            axes[2].imshow(mask, cmap="Reds", alpha=0.35)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
        figure.tight_layout()
        figure.savefig(output_dir / f"sample_{index}.png", dpi=150)
        plt.close(figure)


def write_csv(report: dict, path: Path) -> None:
    rows = []
    for split_name, split_report in report["splits"].items():
        rows.extend([
            {"section": split_name, "metric": "count", "value": split_report.get("count")},
            {"section": split_name, "metric": "dimensions", "value": json.dumps(split_report.get("dimensions", {}))},
            {"section": split_name, "metric": "modes", "value": json.dumps(split_report.get("modes", {}))},
            {"section": split_name, "metric": "intensity_min", "value": split_report.get("intensity_min")},
            {"section": split_name, "metric": "intensity_max", "value": split_report.get("intensity_max")},
            {"section": split_name, "metric": "corrupted_files", "value": len(split_report.get("corrupted_files", []))},
            {"section": split_name, "metric": "exact_duplicate_groups", "value": len(split_report.get("exact_duplicate_groups", []))},
        ])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    training_dir = root / "training_set"
    test_dir = root / "test_set"
    training_files = image_files(training_dir) if training_dir.exists() else []
    test_files = image_files(test_dir) if test_dir.exists() else []
    train_metadata = read_metadata(root / "training_set_pixel_size_and_HC.csv")
    test_metadata = read_metadata(root / "test_set_pixel_size.csv")

    training_names = {path.name for path in training_files}
    train_mask_missing = [
        path.name for path in training_files
        if not path.with_name(f"{path.stem}_Annotation.png").exists()
    ]
    train_csv_missing = sorted(training_names - set(train_metadata))
    train_csv_extra = sorted(set(train_metadata) - training_names)

    test_archive = root / "test_set.zip"
    archive_test_names = []
    if test_archive.exists():
        with zipfile.ZipFile(test_archive) as archive:
            archive_test_names = sorted(
                Path(name).name for name in archive.namelist()
                if name.lower().endswith(tuple(IMAGE_SUFFIXES))
                and not Path(name).stem.endswith("_Annotation")
            )
    test_names = {path.name for path in test_files} or set(archive_test_names)

    report = {
        "dataset_root": str(root),
        "test_policy": "Official test images are audited by archive metadata only and are never used for training or validation.",
        "splits": {
            "training": summarize_images(training_files, "_Annotation.png"),
            "test_extracted": summarize_images(test_files),
            "test_archive": {"count": len(archive_test_names), "filenames": archive_test_names[:10]},
        },
        "correspondence": {
            "training_missing_masks": train_mask_missing,
            "training_missing_csv_rows": train_csv_missing,
            "training_extra_csv_rows": train_csv_extra,
            "test_missing_csv_rows": sorted(test_names - set(test_metadata)),
            "test_extra_csv_rows": sorted(set(test_metadata) - test_names),
        },
        "metadata": {
            "training_rows": len(train_metadata),
            "test_rows": len(test_metadata),
            "pixel_size_min": min((float(row["pixel size"]) for row in train_metadata.values()), default=None),
            "pixel_size_max": max((float(row["pixel size"]) for row in train_metadata.values()), default=None),
            "head_circumference_min_mm": min((float(row["head circumference (mm)"]) for row in train_metadata.values()), default=None),
            "head_circumference_max_mm": max((float(row["head circumference (mm)"]) for row in train_metadata.values()), default=None),
        },
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report, args.report_dir / "dataset_report.csv")
    make_figures(training_files, args.report_dir / "figures")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
