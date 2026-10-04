from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.total_loss import APDGRNLoss, make_edge_target
from models.apd_grn import APDGRN
from models.apdnet import APDNet


def small_apd_grn(**kwargs) -> APDGRN:
    return APDGRN(
        base_channels=16,
        segmentor_channels=(8, 16, 24, 32, 40),
        **kwargs,
    )


def test_apd_grn_output_shapes_and_shared_segmentor() -> None:
    model = small_apd_grn()
    calls = []
    hook = model.segmentor.register_forward_hook(lambda _module, _inputs, _output: calls.append(1))
    image = torch.rand(2, 1, 64, 64)
    outputs = model(image, image)
    hook.remove()

    assert outputs["denoised"].shape == image.shape
    assert outputs["residual"].shape == image.shape
    assert outputs["edge"].shape == image.shape
    assert outputs["reference_segmentation"].shape == (2, 2, 64, 64)
    assert outputs["enhanced_segmentation"].shape == (2, 2, 64, 64)
    assert outputs["discriminator_fake"] is not None
    assert len(calls) == 2


def test_apd_spatial_and_frequency_ablation_shapes() -> None:
    image = torch.rand(1, 1, 32, 32)
    for options in (
        {"use_spatial": True, "use_frequency": False},
        {"use_spatial": False, "use_frequency": True},
    ):
        result = APDNet(base_channels=16, **options)(image)
        assert result["denoised"].shape == image.shape
        assert result["edge"].shape == image.shape


def test_enhanced_segmentation_gradient_reaches_apd_generator() -> None:
    model = small_apd_grn(use_patchgan=False)
    for parameter in model.segmentor.parameters():
        parameter.requires_grad_(False)
    noisy = torch.rand(2, 1, 64, 64)
    reference = torch.rand_like(noisy)
    outputs = model(noisy, reference)
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    loss = F.cross_entropy(outputs["enhanced_segmentation"], target)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.generator.parameters())


def test_joint_loss_supervises_edge_and_restoration_heads() -> None:
    model = small_apd_grn(use_patchgan=False)
    reference = torch.rand(2, 1, 64, 64)
    noisy = (reference * (1.0 + 0.1 * torch.randn_like(reference))).clamp(0, 1)
    mask = torch.zeros_like(reference)
    mask[:, :, 12:52, 12] = 1
    mask[:, :, 12:52, 52] = 1
    mask[:, :, 12, 12:53] = 1
    mask[:, :, 52, 12:53] = 1
    assert torch.equal(make_edge_target(mask), mask)

    outputs = model(noisy, reference)
    total, losses = APDGRNLoss()(outputs, reference, mask)
    total.backward()

    assert torch.isfinite(total)
    assert model.generator.edge_head.weight.grad is not None
    assert model.generator.residual[-1].weight.grad is not None
    assert "L_seg_enhanced" in losses and "L_seg_original" in losses