"""Synthetic multiplicative ultrasound speckle utilities."""

from __future__ import annotations

import torch


def multiplicative_speckle(image: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply I_noisy = clamp(I * (1 + n)), n ~ N(0, sigma^2)."""
    noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype) * sigma
    return (image * (1.0 + noise)).clamp(0.0, 1.0)


def log_transform(image: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.log(image.clamp_min(0.0) + epsilon)


def inverse_log_transform(log_image: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return (torch.exp(log_image) - epsilon).clamp(0.0, 1.0)


def coefficient_of_variation(image: torch.Tensor, window_size: int = 7, epsilon: float = 1e-6) -> torch.Tensor:
    padding = window_size // 2
    mean = torch.nn.functional.avg_pool2d(image, window_size, stride=1, padding=padding, count_include_pad=False)
    second_moment = torch.nn.functional.avg_pool2d(image * image, window_size, stride=1, padding=padding, count_include_pad=False)
    variance = (second_moment - mean * mean).clamp_min(0.0)
    return torch.sqrt(variance + epsilon) / (mean.abs() + epsilon)
