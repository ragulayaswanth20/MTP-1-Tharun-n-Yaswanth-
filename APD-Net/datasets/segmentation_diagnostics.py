"""HC18 datasets for contour and filled fetal-head segmentation experiments."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from datasets.hc18_dataset import Record, load_gray, read_split, resize


def fill_closed_contour(contour: np.ndarray) -> np.ndarray:
    binary = np.asarray(contour > 0, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Cannot fill an empty HC18 annotation")
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, contourIdx=-1, color=1, thickness=cv2.FILLED)
    return filled.astype(np.uint8)


def prepare_filled_mask(record: Record, cache_dir: Path, image_size: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{Path(record.filename).stem}_filled.png"
    if not output.exists():
        source = load_gray(record.mask_path)
        filled = fill_closed_contour(source)
        resized = resize(filled, image_size, mask=True)
        Image.fromarray(resized * 255).save(output)
    return output


class HC18SegmentationDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        image_size: int = 256,
        target_mode: str = "contour",
        augment: bool = False,
        filled_mask_dir: Path | None = None,
    ) -> None:
        if target_mode not in {"contour", "filled_mask"}:
            raise ValueError(f"Unsupported segmentation target: {target_mode}")
        self.records = records
        self.image_size = image_size
        self.target_mode = target_mode
        self.augment = augment
        self.filled_mask_dir = filled_mask_dir
        if target_mode == "filled_mask" and filled_mask_dir is None:
            raise ValueError("filled_mask_dir is required for filled-mask training")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        record = self.records[index]
        image = load_gray(record.image_path)
        contour = load_gray(record.mask_path) > 0
        if self.target_mode == "filled_mask":
            filled_path = prepare_filled_mask(record, self.filled_mask_dir, self.image_size)
            target = load_gray(filled_path) > 0
        else:
            target = contour
        image = resize(image, self.image_size)
        contour = resize(contour.astype(np.uint8), self.image_size, mask=True) > 0
        target = resize(target.astype(np.uint8), self.image_size, mask=True) > 0

        if self.augment:
            rotations = random.randrange(4)
            image = np.rot90(image, rotations)
            contour = np.rot90(contour, rotations)
            target = np.rot90(target, rotations)
            if random.random() < 0.5:
                image, contour, target = image[:, ::-1], contour[:, ::-1], target[:, ::-1]
            if random.random() < 0.5:
                image, contour, target = image[::-1, :], contour[::-1, :], target[::-1, :]
            image_pil = Image.fromarray(np.ascontiguousarray(image))
            image_pil = ImageEnhance.Brightness(image_pil).enhance(random.uniform(0.9, 1.1))
            image = np.asarray(ImageEnhance.Contrast(image_pil).enhance(random.uniform(0.9, 1.1)))

        image_tensor = torch.from_numpy(np.ascontiguousarray(image).copy()).float().unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(np.ascontiguousarray(target).copy()).float().unsqueeze(0)
        contour_tensor = torch.from_numpy(np.ascontiguousarray(contour).copy()).float().unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": target_tensor,
            "contour": contour_tensor,
            "filename": record.filename,
            "pixel_size": record.pixel_size,
            "head_circumference_mm": record.hc_mm,
        }


def make_diagnostic_datasets(
    split_path: str | Path,
    image_size: int,
    target_mode: str,
    augment: bool,
    filled_mask_dir: Path,
) -> tuple[list[Record], HC18SegmentationDataset]:
    records = read_split(split_path)
    if target_mode == "filled_mask":
        for record in records:
            prepare_filled_mask(record, filled_mask_dir, image_size)
    dataset = HC18SegmentationDataset(
        records,
        image_size=image_size,
        target_mode=target_mode,
        augment=augment,
        filled_mask_dir=filled_mask_dir,
    )
    return records, dataset