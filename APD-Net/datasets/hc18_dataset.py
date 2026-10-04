"""Configurable HC-18 dataset access for APD-Net."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

@dataclass(frozen=True)
class Record:
    filename: str
    image_path: Path
    mask_path: Path
    pixel_size: float
    hc_mm: float

def read_split(path: str | Path) -> list[Record]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [Record(row["filename"], Path(row["image_path"]), Path(row["mask_path"]), float(row["pixel_size"]), float(row["head_circumference_mm"])) for row in csv.DictReader(handle)]

def load_gray(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)

def resize(array: np.ndarray, size: int, mask: bool = False) -> np.ndarray:
    interpolation = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array).resize((size, size), interpolation), dtype=np.uint8)

class HC18Dataset(Dataset):
    def __init__(self, records: list[Record], image_size: int = 256):
        self.records, self.image_size = records, image_size
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]
        image = resize(load_gray(record.image_path), self.image_size)
        mask = resize(load_gray(record.mask_path), self.image_size, True)
        return {"image": torch.from_numpy(image.copy()).float()[None] / 255.0, "mask": (torch.from_numpy(mask.copy()).float()[None] > 0).float(), "filename": record.filename, "pixel_size": record.pixel_size, "hc_mm": record.hc_mm}
