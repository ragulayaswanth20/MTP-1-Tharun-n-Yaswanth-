# HC18 Research Project

This isolated project studies fetal-ultrasound denoising and downstream fetal-head segmentation. It does not modify the existing CETUS, GRN, MedGS, or UltraGS projects.

## Phase 1 and 2 status

- HC-18 audit implemented in `scripts/audit_dataset.py`.
- Deterministic split generation implemented in `scripts/create_splits.py`.
- Configurable image/mask loader implemented in `src/hc18_research/data/dataset.py`.
- Official test images remain in the source archive and are not used by development code.
- Weighted BCE+Dice U-Net segmentation baseline implemented and trained in `experiments/segmentation_baseline/`.
- USDRIFT2D synthetic denoiser implemented and trained in `experiments/usdrift2d_synthetic/`.
- Shared denoising and downstream segmentation evaluators implemented in `scripts/`.
- Measured results are recorded in `reports/final_results.csv` and `reports/final_results.md`.

## Audit

```bash
python scripts/audit_dataset.py \
  --dataset-root "../Ultrasound_Dataset/HC-18_Fetus head dataset" \
  --report-dir reports
```

## Splits

```bash
python scripts/create_splits.py \
  --dataset-root "../Ultrasound_Dataset/HC-18_Fetus head dataset" \
  --output-dir splits \
  --seed 20260920
```

The available metadata does not contain patient identifiers, so the split is image-level. The official test set is archive-referenced only and remains isolated.

## Current results

The first synthetic validation run reached USDRIFT2D PSNR `35.7933`, SSIM `0.9478`, and MAE `0.00871` on 200 validation images. The current preliminary segmentation checkpoint did not show a downstream Dice improvement after denoising. These are validation results only; the test set remains untouched.

Remaining research phases include self-supervised real-ultrasound training, DnCNN/U-Net denoising baselines, ablations, stronger segmentation training, publication visualizations, and final test evaluation.
