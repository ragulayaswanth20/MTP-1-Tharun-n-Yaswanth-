"""Configurable APD-Net multi-objective loss."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def gradients(x):
    gx=F.pad(x[..., :,1:]-x[..., :, :-1],(0,1,0,0)); gy=F.pad(x[...,1:,:]-x[...,:-1,:],(0,0,0,1)); return torch.sqrt(gx*gx+gy*gy+1e-6)
def dice_loss(logits,target):
    p=torch.sigmoid(logits); inter=(p*target).sum((1,2,3)); den=p.sum((1,2,3))+target.sum((1,2,3)); return (1-(2*inter+1)/(den+1)).mean()
def ssim_loss(x,y):
    w=11; p=w//2; mx=F.avg_pool2d(x,w,1,p,count_include_pad=False); my=F.avg_pool2d(y,w,1,p,count_include_pad=False); vx=F.avg_pool2d(x*x,w,1,p,count_include_pad=False)-mx*mx; vy=F.avg_pool2d(y*y,w,1,p,count_include_pad=False)-my*my; cov=F.avg_pool2d(x*y,w,1,p,count_include_pad=False)-mx*my; c1=.01**2;c2=.03**2; score=((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)); return 1-score.mean()
class APDLoss(nn.Module):
    def __init__(self, weights=None): super().__init__(); self.weights=weights or {"pixel":1.,"ssim":1.,"edge":.5,"lap":.2,"speckle":.5,"frequency":.1,"seg":.5}
    def forward(self,pred,target,mask=None):
        edge=F.l1_loss(gradients(pred),gradients(target)); lap=F.l1_loss(F.avg_pool2d(pred,3,1,1),F.avg_pool2d(target,3,1,1)); speckle=F.l1_loss(torch.log(pred+1e-3),torch.log(target+1e-3)); freq=F.l1_loss(torch.log1p(torch.fft.rfft2(pred).abs()),torch.log1p(torch.fft.rfft2(target).abs())); parts={"pixel":F.smooth_l1_loss(pred,target),"ssim":ssim_loss(pred,target),"edge":edge,"lap":lap,"speckle":speckle,"frequency":freq,"seg":torch.zeros((),device=pred.device)}
        if mask is not None: parts["seg"]=dice_loss(mask[0],mask[1]) if isinstance(mask,tuple) else torch.zeros((),device=pred.device)
        return sum(self.weights[k]*parts[k] for k in parts),parts


def make_edge_target(mask: torch.Tensor, mode: str = "annotation", kernel_size: int = 3) -> torch.Tensor:
    """Use HC18's annotated contour directly, or derive contours from filled masks."""
    mask = mask.float()
    if mode == "annotation":
        return (mask > 0).float()
    if mode == "morphological_gradient":
        padding = kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size, stride=1, padding=padding)
        return (dilated - eroded).clamp(0.0, 1.0)
    raise ValueError(f"Unknown edge target mode: {mode}")


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ce_weight: float = 1.0,
    foreground_weight: float = 1.0,
) -> torch.Tensor:
    target = target[:, 0].long() if target.ndim == 4 else target.long()
    classes = logits.shape[1]
    one_hot = F.one_hot(target, num_classes=classes).permute(0, 3, 1, 2).float()
    probabilities = torch.softmax(logits, dim=1)
    dimensions = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dimensions)
    denominator = probabilities.sum(dimensions) + one_hot.sum(dimensions)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    class_weights = logits.new_ones(classes)
    if classes > 1:
        class_weights[1:] = foreground_weight
    return dice + ce_weight * F.cross_entropy(logits, target, weight=class_weights)


class APDGRNLoss(nn.Module):
    """Configurable joint restoration, edge, segmentation, and GAN objective."""

    def __init__(self, weights: dict[str, float] | None = None, edge_target_mode: str = "annotation") -> None:
        super().__init__()
        self.weights = weights or {
            "restore": 1.0,
            "l1": 1.0,
            "ssim": 1.0,
            "frequency": 0.1,
            "edge": 1.0,
            "edge_dice": 1.0,
            "seg": 1.0,
            "seg_ce": 1.0,
            "seg_foreground_weight": 20.0,
            "edge_positive_weight": 20.0,
            "adv": 0.01,
        }
        self.edge_target_mode = edge_target_mode

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        reference: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        denoised = outputs["denoised"]
        edge_logits = outputs["edge"]
        segmentation_original = outputs["reference_segmentation"]
        segmentation_enhanced = outputs["enhanced_segmentation"]
        if denoised is None or edge_logits is None or segmentation_original is None or segmentation_enhanced is None:
            raise ValueError("APD-GRN outputs are missing a required prediction")

        edge_target = make_edge_target(mask, self.edge_target_mode)
        l1 = F.l1_loss(denoised, reference)
        ssim = ssim_loss(denoised, reference)
        frequency = F.l1_loss(
            torch.log1p(torch.fft.fft2(denoised).abs()),
            torch.log1p(torch.fft.fft2(reference).abs()),
        )
        edge_bce = F.binary_cross_entropy_with_logits(
            edge_logits,
            edge_target,
            pos_weight=edge_logits.new_tensor(self.weights["edge_positive_weight"]),
        )
        edge_dice = dice_loss(edge_logits, edge_target)
        edge = edge_bce + self.weights["edge_dice"] * edge_dice
        seg_original = segmentation_loss(
            segmentation_original,
            mask,
            self.weights["seg_ce"],
            self.weights["seg_foreground_weight"],
        )
        seg_enhanced = segmentation_loss(
            segmentation_enhanced,
            mask,
            self.weights["seg_ce"],
            self.weights["seg_foreground_weight"],
        )
        adversarial_logits = outputs.get("discriminator_fake")
        adversarial = (
            F.mse_loss(adversarial_logits, torch.ones_like(adversarial_logits))
            if adversarial_logits is not None
            else reference.new_zeros(())
        )
        restore_raw = l1 + ssim + frequency
        restore = (
            self.weights["l1"] * l1
            + self.weights["ssim"] * ssim
            + self.weights["frequency"] * frequency
        )
        weighted_restore = self.weights["restore"] * restore
        weighted_edge = self.weights["edge"] * edge
        weighted_seg = self.weights["seg"] * seg_enhanced
        weighted_adversarial = self.weights["adv"] * adversarial
        total_generator = (
            weighted_restore
            + weighted_edge
            + weighted_seg
            + weighted_adversarial
        )
        losses = {
            "L_restore_raw": restore_raw,
            "L_restore": restore,
            "L_L1": l1,
            "L_SSIM": ssim,
            "L_frequency": frequency,
            "L_edge": edge,
            "L_seg_original": seg_original,
            "L_seg_enhanced": seg_enhanced,
            "L_S": 0.5 * (seg_original + seg_enhanced),
            "L_adv": adversarial,
            "L_restore_weighted": weighted_restore,
            "L_edge_weighted": weighted_edge,
            "L_seg_weighted": weighted_seg,
            "L_adv_weighted": weighted_adversarial,
            "L_total_G": total_generator,
        }
        return total_generator, losses

    @staticmethod
    def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            F.mse_loss(real_logits, torch.ones_like(real_logits))
            + F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
        )
