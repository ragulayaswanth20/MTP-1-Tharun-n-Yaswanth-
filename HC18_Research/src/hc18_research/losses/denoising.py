"""Differentiable losses for synthetic-reference denoising."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def gradient_map(image: torch.Tensor) -> torch.Tensor:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-6)


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    pad = window // 2
    mean_pred = F.avg_pool2d(prediction, window, stride=1, padding=pad, count_include_pad=False)
    mean_target = F.avg_pool2d(target, window, stride=1, padding=pad, count_include_pad=False)
    mean_pred_sq = F.avg_pool2d(prediction * prediction, window, stride=1, padding=pad, count_include_pad=False)
    mean_target_sq = F.avg_pool2d(target * target, window, stride=1, padding=pad, count_include_pad=False)
    cross = F.avg_pool2d(prediction * target, window, stride=1, padding=pad, count_include_pad=False)
    var_pred = (mean_pred_sq - mean_pred * mean_pred).clamp_min(0.0)
    var_target = (mean_target_sq - mean_target * mean_target).clamp_min(0.0)
    covariance = cross - mean_pred * mean_target
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_pred * mean_target + c1) * (2 * covariance + c2)) / ((mean_pred**2 + mean_target**2 + c1) * (var_pred + var_target + c2))
    return 1.0 - score.mean()


def boundary_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pooled_max = F.max_pool2d(target, 3, stride=1, padding=1)
    pooled_min = -F.max_pool2d(-target, 3, stride=1, padding=1)
    boundary = (pooled_max - pooled_min).clamp_min(0.0)
    weight = 1.0 + 4.0 * boundary
    return (weight * (prediction - target).abs()).mean()


def frequency_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_spectrum = torch.fft.rfft2(prediction, norm="ortho").abs()
    target_spectrum = torch.fft.rfft2(target, norm="ortho").abs()
    return F.l1_loss(torch.log1p(pred_spectrum), torch.log1p(target_spectrum))


class AnatomyAwareDenoisingLoss(nn.Module):
    def __init__(self, pixel_weight=1.0, ssim_weight=0.2, gradient_weight=0.1, frequency_weight=0.1, anatomy_weight=0.5):
        super().__init__()
        self.weights = {
            "pixel": pixel_weight,
            "ssim": ssim_weight,
            "gradient": gradient_weight,
            "frequency": frequency_weight,
            "anatomy": anatomy_weight,
        }

    def forward(self, prediction, target, mask):
        components = {
            "pixel": F.l1_loss(prediction, target),
            "ssim": ssim_loss(prediction, target),
            "gradient": F.l1_loss(gradient_map(prediction), gradient_map(target)),
            "frequency": frequency_loss(prediction, target),
            "anatomy": boundary_loss(prediction * mask, target * mask),
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        return total, components
