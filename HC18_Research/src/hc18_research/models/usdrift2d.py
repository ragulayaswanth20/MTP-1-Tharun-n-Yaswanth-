"""Working USDRIFT2D architecture for anatomy-aware speckle denoising."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from hc18_research.utils.noise import coefficient_of_variation, inverse_log_transform, log_transform


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class USDRIFT2D(nn.Module):
    """Modular 2D denoiser with log residual, CV gate, spatial/frequency branches."""

    def __init__(self, base_channels: int = 16, cv_window: int = 7, use_frequency: bool = True, epsilon: float = 1e-3):
        super().__init__()
        self.cv_window = cv_window
        self.use_frequency = use_frequency
        self.epsilon = epsilon
        self.spatial = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.frequency = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            ResidualBlock(base_channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(base_channels + 1, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 1), nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        log_image = log_transform(image, self.epsilon)
        spatial_features = self.spatial(log_image)
        if self.use_frequency:
            spectrum = torch.fft.rfft2(image, norm="ortho").abs()
            spectrum = torch.log1p(spectrum)
            spectrum = F.interpolate(spectrum, size=image.shape[-2:], mode="bilinear", align_corners=False)
            frequency_features = self.frequency(spectrum)
            features = spatial_features + frequency_features
        else:
            features = spatial_features
        cv_map = coefficient_of_variation(image, self.cv_window)
        gate = self.gate(torch.cat([features, cv_map], dim=1))
        residual = self.residual_head(features * gate)
        return inverse_log_transform(log_image - residual, self.epsilon)
