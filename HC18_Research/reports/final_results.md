# HC-18 Research Results

## Scope

These are preliminary results from the isolated `HC18_Research` project. The official 335-image HC-18 test set was not extracted or used. All reported metrics use the deterministic 799/200 train/validation split and seed `20260920`.

The synthetic denoising experiment uses multiplicative speckle:

$$
I_{\mathrm{noisy}} = \operatorname{clip}(I(1+n), 0, 1),
\qquad n \sim \mathcal{N}(0, 0.2^2).
$$

The original HC-18 image is used as a reference only for this controlled synthetic experiment. It is not claimed to be a noise-free clinical ground truth.

## Synthetic Denoising

| Method | PSNR | SSIM | MAE | Gradient error |
|---|---:|---:|---:|---:|
| Noisy | 27.0703 | 0.7399 | 0.02522 | 0.03581 |
| Median | 35.1392 | 0.9404 | 0.00952 | 0.00988 |
| Gaussian | 34.3745 | **0.9504** | 0.00906 | **0.00982** |
| Bilateral | 34.5779 | 0.9396 | 0.00977 | 0.01092 |
| TV-Chambolle | 33.8422 | 0.9141 | 0.01227 | 0.00998 |
| USDRIFT2D | **35.7933** | 0.9478 | **0.00871** | 0.00987 |

USDRIFT2D achieved the highest PSNR and lowest MAE in this preliminary experiment. Gaussian achieved the highest SSIM and lowest gradient error by a small margin. These findings are not a clinical denoising claim and require repeated seeds, additional noise levels, and real-noise evaluation.

## Downstream Segmentation

The same fixed preliminary U-Net checkpoint was applied to original, noisy, classical-denoised, and USDRIFT2D validation images.

| Input | Dice | IoU | HD95 pixels | Head-circumference MAE (mm) |
|---|---:|---:|---:|---:|
| Original | 0.2174 | 0.1226 | 39.25 | 86.08 |
| Noisy | 0.2171 | 0.1224 | 37.16 | 85.85 |
| Gaussian | 0.2172 | 0.1225 | 39.91 | 89.49 |
| Bilateral | 0.2171 | 0.1224 | 39.58 | 89.17 |
| TV-Chambolle | 0.2150 | 0.1211 | 42.51 | 95.09 |
| USDRIFT2D | 0.2161 | 0.1218 | 41.04 | 92.35 |

The current denoising improvements did not translate into improved downstream segmentation. This is a measured negative result, not evidence that denoising cannot help segmentation. The segmentation baseline is preliminary and requires further optimization, boundary-focused losses, and stronger validation before drawing conclusions.

## Dataset Audit

- 999 labeled training images
- 999 matching annotation masks
- 335 official test images referenced through the untouched archive
- No corrupted training images detected
- No missing training masks detected
- No exact duplicate training images detected
- Training image dimensions are not uniform: 975 are `800x540`, with 24 variants
- Patient identifiers were not present, so the split is image-level

## Current Limitations

- Only one deterministic split and one synthetic noise strength were evaluated.
- The synthetic reference is not a measured clean clinical target.
- The self-supervised real-ultrasound experiment is not implemented yet.
- DnCNN, U-Net denoiser, and full ablation runs are not implemented yet.
- HD95 and head-circumference values reflect the preliminary segmentation checkpoint.
- The official test set remains intentionally unevaluated.
