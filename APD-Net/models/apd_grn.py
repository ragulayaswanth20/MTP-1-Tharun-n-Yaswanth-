"""Task-guided APD-Net generator combined with a shared segmentor and PatchGAN."""
from __future__ import annotations

import torch
from torch import nn

from models.apdnet import APDNet
from models.grn_components import PatchGANDiscriminator, UNetSegmentor


class APDGRN(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        num_classes: int = 2,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_attention: bool = True,
        use_edge: bool = True,
        use_edge_feedback: bool = True,
        use_segmentation_feedback: bool = True,
        use_patchgan: bool = True,
        segmentor_channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.generator = APDNet(
            base_channels=base_channels,
            use_spatial=use_spatial,
            use_frequency=use_frequency,
            use_attention=use_attention,
            use_edge=use_edge,
            edge_feedback=use_edge_feedback,
        )
        self.segmentor = UNetSegmentor(num_classes=num_classes, channels=segmentor_channels)
        self.discriminator = PatchGANDiscriminator() if use_patchgan else None
        self.use_edge = use_edge
        self.use_segmentation_feedback = use_segmentation_feedback
        self.use_patchgan = use_patchgan

    def forward(self, noisy: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor | None]:
        generated = self.generator(noisy)
        denoised = generated["denoised"]
        reference_segmentation = self.segmentor(reference)
        enhanced_segmentation = self.segmentor(denoised)
        discriminator_fake = self.discriminator(denoised) if self.discriminator is not None else None
        return {
            "reference_segmentation": reference_segmentation,
            "enhanced_segmentation": enhanced_segmentation,
            "denoised": denoised,
            "residual": generated["residual"],
            "edge": generated["edge"],
            "discriminator_fake": discriminator_fake,
            "spatial_features": generated["spatial_features"],
            "frequency_features": generated["frequency_features"],
            "attended_features": generated["attended_features"],
        }