"""Segmentation metrics for binary fetal-head masks."""

from __future__ import annotations

import numpy as np


def dice_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    return float((2.0 * intersection + eps) / (prediction.sum() + target.sum() + eps))


def iou_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return float((intersection + eps) / (union + eps))
