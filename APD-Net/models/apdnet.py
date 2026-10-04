"""APD-Net: Anatomy-Preserving Speckle Denoising Network."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(min(groups, channels), channels), nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(min(groups, channels), channels))
    def forward(self, x): return F.gelu(x + self.block(x))

class MultiScaleFeatureBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        branch = channels // 3
        self.branches = nn.ModuleList([nn.Conv2d(channels, branch, 3, padding=1), nn.Conv2d(channels, branch, 5, padding=2), nn.Conv2d(channels, branch, 3, padding=2, dilation=2)])
        self.project = nn.Conv2d(branch * 3, channels, 1)
    def forward(self, x): return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))

class AnatomyAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 8, 4)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())
    def forward(self, x):
        channel = self.channel(x) * x
        spatial = torch.cat([channel.mean(1, keepdim=True), channel.amax(1, keepdim=True)], 1)
        return channel * self.spatial(spatial)

class APDNet(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        use_frequency: bool = True,
        use_attention: bool = True,
        *,
        use_spatial: bool = True,
        use_edge: bool = True,
        edge_feedback: bool = False,
        edge_scale: float = 0.5,
    ):
        super().__init__()
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency
        self.use_attention = use_attention
        self.use_edge = use_edge
        self.edge_feedback = edge_feedback
        self.edge_scale = edge_scale
        c = base_channels
        self.input = nn.Conv2d(1, c, 3, padding=1)
        self.multi = MultiScaleFeatureBlock(c)
        self.blocks = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.frequency = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.GELU(), ResidualBlock(c))
        self.attention = AnatomyAttention(c)
        self.edge_head = nn.Conv2d(c, 1, 1)
        self.residual = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 1, 3, padding=1))
    def forward(self, image):
        feature_paths = []
        spatial_features = None
        frequency_features = None
        if self.use_spatial:
            spatial_features = self.blocks(self.multi(self.input(torch.log(image.clamp_min(0) + 1e-3))))
            feature_paths.append(spatial_features)
        if self.use_frequency:
            spectrum = torch.log1p(torch.fft.rfft2(image, norm="ortho").abs())
            spectrum = F.interpolate(spectrum, image.shape[-2:], mode="bilinear", align_corners=False)
            frequency_features = self.frequency(spectrum)
            feature_paths.append(frequency_features)
        if not feature_paths:
            raise ValueError("APD-Net requires at least one of the spatial or frequency branches")
        features = feature_paths[0]
        for feature_path in feature_paths[1:]:
            features = features + feature_path
        attended_features = features + self.attention(features) if self.use_attention else features
        edge = self.edge_head(attended_features) if self.use_edge else torch.zeros_like(image)
        residual_features = attended_features
        if self.use_edge and self.edge_feedback:
            residual_features = attended_features * (1.0 + self.edge_scale * torch.sigmoid(edge))
        residual = self.residual(residual_features)
        denoised = (image - residual).clamp(0, 1)
        return {
            "image": denoised,
            "denoised": denoised,
            "residual": residual,
            "edge": edge,
            "spatial_features": spatial_features,
            "frequency_features": frequency_features,
            "attended_features": attended_features,
        }
