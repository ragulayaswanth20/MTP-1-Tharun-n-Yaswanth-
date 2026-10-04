"""Reference-based synthetic denoising metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((prediction.astype(np.float32) - target.astype(np.float32)) ** 2)
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(target, prediction, data_range=1.0))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction.astype(np.float32) - target.astype(np.float32))))


def gradient_error(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_y, pred_x = np.gradient(prediction.astype(np.float32))
    target_y, target_x = np.gradient(target.astype(np.float32))
    return float(np.mean(np.abs(pred_y - target_y) + np.abs(pred_x - target_x)))
