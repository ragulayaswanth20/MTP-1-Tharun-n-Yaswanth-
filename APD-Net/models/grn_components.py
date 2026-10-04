"""Segmentation and adversarial components used by the APD-GRN wrapper."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNetSegmentor(nn.Module):
    """2D U-Net with the GRN paper's (16, 32, 64, 128, 256) channel widths."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("UNetSegmentor requires at least two channel stages")
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in channels:
            self.encoders.append(ConvBlock(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for index in range(len(channels) - 2, -1, -1):
            width = channels[index]
            self.upsamplers.append(nn.ConvTranspose2d(channels[index + 1], width, 2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width))
        self.output = nn.Conv2d(channels[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.encoders[-1](x)
        for upsample, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            x = upsample(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
        return self.output(x)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, widths[0], 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for index in range(1, len(widths)):
            layers.extend(
                [
                    nn.Conv2d(widths[index - 1], widths[index], 4, stride=2 if index < 3 else 1, padding=1, bias=False),
                    nn.BatchNorm2d(widths[index]),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        layers.append(nn.Conv2d(widths[-1], 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)