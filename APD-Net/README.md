# APD-Net

APD-Net is an isolated research project for anatomy-preserving fetal-ultrasound speckle denoising. It does not modify the existing CETUS, GRN, MedGS, UltraGS, or `HC18_Research` projects.

## Scientific constraint

HC18 does not provide paired noisy/clean clinical ultrasound. The current paired metrics use controlled synthetic multiplicative speckle with the original image treated as a pseudo-clean reference:

`I_noisy = clip(I_reference * (1 + n), 0, 1)`, `n ~ N(0, 0.2^2)`.

The official 335-image test set is not used by the current experiment.

## Implemented

- APD-Net residual denoiser
- Multi-scale spatial branch
- FFT magnitude frequency branch
- Channel/spatial Anatomy Attention Module
- Configurable multi-objective loss: pixel, SSIM, gradient, Laplacian, log-speckle, frequency, and segmentation terms
- HC18 loader using the verified split in `../HC18_Research/splits/`
- Stage 1 synthetic training
- Validation metrics: PSNR, SSIM, MAE, and gradient error
- Measurement report in `report.tex`

## Run

From this directory:

```bash
python -m py_compile models/apdnet.py losses/total_loss.py scripts/train_apd.py scripts/evaluate_apd.py
python scripts/train_apd.py --config configs/apdnet.yaml
python scripts/evaluate_apd.py --config configs/apdnet.yaml
```

Outputs:

- `checkpoints/apd_stage1_best.pt`
- `results/apd_stage1_metrics.json`
- `results/apd_stage1_metrics.csv`
- `results/apd_metrics.json`
- `results/apd_metrics.csv`

## APD-GRN Joint Training

The task-guided APD-GRN experiment retains APD-Net as the generator and adds one shared U-Net segmentor and an optional PatchGAN. The training objective combines restoration (L1, SSIM, and frequency), HC18 contour supervision, enhanced-image segmentation feedback, and adversarial loss. The original HC18 image is the reference image; synthetic multiplicative Gaussian speckle is sampled per image with sigma uniformly drawn from the configured range.

Run from this directory:

```bash
python -m py_compile models/apdnet.py models/apd_grn.py models/grn_components.py losses/total_loss.py metrics/apd_grn_metrics.py scripts/train_apd_grn.py
python scripts/train_apd_grn.py --config configs/apd_grn.yaml
```

The config controls branch and objective ablations, noise range, evaluation levels, and loss weights. The generator warm-starts from `checkpoints/apd_stage1_best.pt`. Outputs are written to:

- `checkpoints/apd_grn_best.pt`
- `results/apd_grn/evaluation.json` and `results/apd_grn/per_image_metrics.csv`
- `results/apd_grn/examples_sigma_0.20/`

The 20-epoch run selected epoch 14 using validation PSNR at sigma 0.20. On 200 validation images it achieved 34.87 dB PSNR and 0.9529 SSIM at sigma 0.20. Exact-pixel contour Dice was 0.261 and head-circumference MAE was 3.98 mm; the sparse contour labels make exact-pixel overlap a strict measure. Results use synthetic corruption and original HC18 images as references, not speckle-free clinical ground truth. The official test set remains unused.

## Measured Stage 1 validation result

On 200 held-out validation images after the configured five-epoch run:

- Noisy baseline: 27.11 dB PSNR, 0.8152 SSIM, 0.02527 MAE
- APD-Net: 34.51 dB PSNR, 0.9494 SSIM, 0.01122 MAE

These are synthetic-reference results, not clinical denoising claims.

## Remaining validation

## Segmentation and APD-GRN Diagnostics

The diagnostic workflow isolates the contour and filled-mask U-Net segmentors, warm-starts APD reconstruction training, then jointly fine-tunes APD-GRN from both checkpoints. Original HC18 annotations remain unchanged; derived filled masks are cached separately.

From this directory:

```bash
python scripts/train_segmentor_baseline.py --config configs/segmentor_diagnostics.yaml --target both
python scripts/evaluate_segmentor_baselines.py
python scripts/train_apd_reconstruction.py --config configs/apd_reconstruction_pretrain.yaml
python scripts/train_apd_reconstruction.py --config configs/apd_reconstruction_fair.yaml
python scripts/train_apd_grn.py --config configs/apd_grn_stagec.yaml
python scripts/train_apd_grn.py --config configs/apd_grn_stagec.yaml --ablation M5 --run-name M5 --epochs 20
python scripts/train_apd_grn.py --config configs/apd_grn_stagec.yaml --ablation M6 --run-name M6 --epochs 20
python scripts/analyze_diagnostics.py
```

The M1–M4 runs use the same command pattern with their respective `--ablation` and `--run-name` values. The staged checkpoints are `checkpoints/diagnostics/best_segmentor.pt`, `best_apd_reconstruction.pt`, `best_apd_reconstruction_fair.pt`, `best_reconstruction.pt`, `best_segmentation.pt`, and `best_joint.pt`.

Results are collected under `results/diagnostics/`: per-image B0–B4 evaluations at six sigmas, M0–M7 ablations, paired statistics, loss/gradient diagnostics, split audit, and publication panels. `B4_GRN_native_arch_adaptation` uses the GRN project's native generator architecture trained with a separate HC18 synthetic-denoising adaptation; it is not an untouched reproduction of the paper's original training protocol.

In the single-seed validation runs, the reference-only contour segmentor reached Dice 0.269 but Boundary F1 0.751 at 2-pixel tolerance and 0.959 at 5 pixels; the filled-mask segmentor reached Dice 0.969. At sigma 0.20, the shared segmentor's Boundary F1@2 fell from 0.751 on reference images to 0.218 on noisy inputs, then recovered to 0.748 on ordinary APD and 0.749 on APD-GRN. Budget-matched APD restoration reached 35.10 dB PSNR, compared with 34.26 dB for APD-GRN. M5-vs-M6 feedback and M6-vs-M7 PatchGAN comparisons showed no meaningful segmentation gain. Weighted Stage-C generator loss contributions averaged roughly 4% restoration, 73% edge, 22% segmentation, and less than 1% adversarial.

These are validation-only, synthetic-corruption results. Image-level splits are verified, but the metadata lacks patient/examination IDs, so group leakage cannot be assessed. The official test set was not loaded. Paired confidence intervals reflect validation-image sampling; model comparisons and ablations use one training seed and their p-values are exploratory, unadjusted for multiple comparisons.

Repeated-seed studies, external baselines, self-supervised real-ultrasound evaluation, group-aware splits (if metadata becomes available), and official test inference remain necessary before making a final research claim.
