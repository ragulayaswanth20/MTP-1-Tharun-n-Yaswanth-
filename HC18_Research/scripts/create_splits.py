"""Create deterministic HC-18 train/validation/test reference CSVs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    training_dir = root / "training_set"
    train_metadata = read_rows(root / "training_set_pixel_size_and_HC.csv")
    filenames = sorted(
        row["filename"] for row in train_metadata.values()
        if (training_dir / row["filename"]).exists()
        and (training_dir / f"{Path(row['filename']).stem}_Annotation.png").exists()
    )
    if len(filenames) != len(train_metadata):
        raise RuntimeError("Training metadata does not match image and mask files.")

    shuffled = filenames.copy()
    random.Random(args.seed).shuffle(shuffled)
    split_index = int(len(shuffled) * (1.0 - args.val_fraction))
    train_names = set(shuffled[:split_index])
    val_names = set(shuffled[split_index:])

    train_rows = []
    val_rows = []
    for filename in filenames:
        source = train_metadata[filename]
        row = {
            "filename": filename,
            "image_path": str(training_dir / filename),
            "mask_path": str(training_dir / f"{Path(filename).stem}_Annotation.png"),
            "pixel_size": source["pixel size"],
            "head_circumference_mm": source["head circumference (mm)"],
            "split_seed": str(args.seed),
        }
        (train_rows if filename in train_names else val_rows).append(row)

    test_metadata = read_rows(root / "test_set_pixel_size.csv")
    test_archive = root / "test_set.zip"
    test_rows = [
        {
            "filename": filename,
            "pixel_size": row["pixel size"],
            "source_archive": str(test_archive),
            "labels_available": "false",
        }
        for filename, row in sorted(test_metadata.items())
    ]

    fields = ["filename", "image_path", "mask_path", "pixel_size", "head_circumference_mm", "split_seed"]
    write_rows(args.output_dir / "train.csv", fields, train_rows)
    write_rows(args.output_dir / "val.csv", fields, val_rows)
    write_rows(args.output_dir / "test.csv", ["filename", "pixel_size", "source_archive", "labels_available"], test_rows)

    (args.output_dir / "README.txt").write_text(
        "HC-18 split policy\n"
        f"seed={args.seed}\n"
        f"train={len(train_rows)}\n"
        f"validation={len(val_rows)}\n"
        f"official_test={len(test_rows)}\n"
        "The official test set is archive-referenced only and was not used for training, validation, tuning, or model selection.\n"
        "Patient identifiers were not present in the available metadata; the split is image-level.\n",
        encoding="utf-8",
    )
    print(f"Created train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)} with seed={args.seed}")


if __name__ == "__main__":
    main()
