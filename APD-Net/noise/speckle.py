"""Controlled synthetic multiplicative speckle models."""
from __future__ import annotations
import torch

def apply_speckle(image: torch.Tensor, strength: float = 0.2, model: str = "gaussian", seed: int | None = None) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=image.device).manual_seed(seed)
    if model == "gaussian":
        factor = 1.0 + strength * torch.randn(image.shape, generator=generator, device=image.device)
    elif model == "gamma":
        shape = max(1.0 / max(strength * strength, 1e-6), 1.0)
        factor = torch.distributions.Gamma(shape, shape).sample(image.shape).to(image.device)
    elif model == "rayleigh":
        factor = torch.sqrt(torch.randn(image.shape, generator=generator, device=image.device).pow(2) + torch.randn(image.shape, generator=generator, device=image.device).pow(2))
        factor = factor / factor.mean().clamp_min(1e-6)
        factor = 1.0 + strength * (factor - 1.0)
    else:
        raise ValueError(f"Unknown speckle model: {model}")
    return (image * factor).clamp(0.0, 1.0)
