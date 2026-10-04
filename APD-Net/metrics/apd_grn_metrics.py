"""Image-restoration and HC18 contour metrics for APD-GRN evaluation."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.metrics import structural_similarity


def image_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0.0, 1.0)
    reference = np.clip(reference, 0.0, 1.0)
    mse = float(np.mean((prediction - reference) ** 2))
    return {
        "psnr_db": float(10.0 * np.log10(1.0 / max(mse, 1e-12))),
        "ssim": float(structural_similarity(reference, prediction, data_range=1.0)),
        "mae": float(np.mean(np.abs(prediction - reference))),
    }


def contour_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    dice = float((2.0 * intersection) / denominator) if denominator else 1.0
    union = np.logical_or(prediction, target).sum()
    iou = float(intersection / union) if union else 1.0
    if not prediction.any() or not target.any():
        return {"dice": dice, "iou": iou, "hd95_pixels": float("nan"), "asd_pixels": float("nan")}

    target_distance = distance_transform_edt(~target)
    prediction_distance = distance_transform_edt(~prediction)
    distances = np.concatenate((target_distance[prediction], prediction_distance[target]))
    return {
        "dice": dice,
        "iou": iou,
        "hd95_pixels": float(np.percentile(distances, 95)),
        "asd_pixels": float(np.mean(distances)),
    }


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (mask > 0) & (eroded == 0)


def boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance_pixels: int = 2) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() or not target.any():
        return float(prediction.any() == target.any())
    distance_to_target = distance_transform_edt(~target)
    distance_to_prediction = distance_transform_edt(~prediction)
    precision = float((distance_to_target[prediction] <= tolerance_pixels).mean())
    recall = float((distance_to_prediction[target] <= tolerance_pixels).mean())
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def head_circumference_mm(
    contour: np.ndarray,
    pixel_size_mm: float,
    original_shape: tuple[int, int],
    resized_shape: tuple[int, int],
) -> float:
    binary = np.asarray(contour, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return float("nan")
    largest = max(contours, key=lambda candidate: cv2.arcLength(candidate, closed=True))
    if len(largest) < 5:
        return float("nan")
    original_height, original_width = original_shape
    resized_height, resized_width = resized_shape
    points_xy = largest[:, 0, :].astype(np.float32)
    points_xy[:, 0] *= pixel_size_mm * original_width / resized_width
    points_xy[:, 1] *= pixel_size_mm * original_height / resized_height
    ellipse = cv2.fitEllipse(points_xy.reshape(-1, 1, 2))
    semi_major, semi_minor = (max(ellipse[1]) / 2.0, min(ellipse[1]) / 2.0)
    perimeter = np.pi * (
        3.0 * (semi_major + semi_minor)
        - np.sqrt((3.0 * semi_major + semi_minor) * (semi_major + 3.0 * semi_minor))
    )
    return float(perimeter)