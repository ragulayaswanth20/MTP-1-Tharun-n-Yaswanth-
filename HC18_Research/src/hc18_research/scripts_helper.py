"""Small shared helpers for experiment scripts."""

from __future__ import annotations

import numpy as np


def noisy_image(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    normalized = image.astype(np.float32) / 255.0
    noise = rng.normal(0.0, sigma, normalized.shape).astype(np.float32)
    return np.clip(normalized * (1.0 + noise), 0.0, 1.0).astype(np.float32)
