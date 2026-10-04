# APD-GRN Diagnostic Report

## Findings

A. **Can the standalone segmentor learn HC18?** Yes. Reference-only contour training reached Dice 0.269, Boundary F1@2 0.751, and HD95 5.62 pixels. The filled-mask experiment reached Dice 0.969 and IoU 0.944.

B. **Does exact contour Dice understate geometric quality?** Yes. Contour Dice is 0.269, while tolerance-aware Boundary F1 rises from 0.567@1px to 0.751@2px and 0.959@5px.

C. **Does sigma=0.20 speckle degrade segmentation?** Yes. With the same fixed reference-trained segmentor, BF1@2 changes from 0.751 on reference images to 0.218 on noisy inputs.

D. **Does ordinary APD recover it?** Yes, nearly to the reference baseline: APD BF1@2 is 0.748; its restoration PSNR is 35.10 dB.

E. **Does APD-GRN outperform ordinary APD?** Not meaningfully in this run. APD-GRN BF1@2 is 0.749, a paired mean difference of +0.0009 (95% bootstrap CI -0.0016 to +0.0035, p=0.6836). APD-GRN restoration PSNR is 34.26 dB versus APD 35.10 dB.

F. **Are segmentation gradients meaningful?** Median weighted generator gradient-norm ratio `||grad_G(L_seg)|| / ||grad_G(L_restore)||` is 0.565 (range 0.120–3.996); gradients are connected and not uniformly negligible.

G. **Does PatchGAN improve segmentation?** M7-vs-M6 Boundary F1@2 paired difference is +0.0001 (95% CI -0.0032 to +0.0032, p=0.9701); this single-seed run does not establish a reliable PatchGAN segmentation gain.

The B4 comparison uses GRN's native generator architecture trained with our separate HC18 synthetic-denoising adaptation; it is not an untouched reproduction of the original paper's training protocol.

H. **Best reconstruction checkpoint:** `best_reconstruction.pt`, selected by validation reconstruction score. **Best downstream segmentation checkpoint:** `best_segmentation.pt`, selected by validation Boundary F1@2.

I. **Current recommendation:** evidence supports keeping APD-Net as the primary despeckling contribution and reporting APD-GRN as an investigated task-guidance extension. The standalone segmentor can learn HC18, but M6 does not provide a practically meaningful gain over M5 and fixed-segmentor APD-GRN does not outperform ordinary APD. Do not claim segmentation improvement from GRN feedback.

The M5-vs-M6 task-feedback paired Boundary F1@2 difference is +0.0002 (95% CI -0.0005 to +0.0010, p=0.8385).

Mean weighted generator-loss fractions across Stage C were restoration 4.3%, edge 73.3%, segmentation 22.2%, and adversarial 0.2%. See `loss_weight_diagnostics.csv` for raw and weighted means.

## Interpretation Limits

- HC18 images are original ultrasound references, not speckle-free ground truth; denoising metrics are synthetic-reference metrics.
- HC18 has thin annotated contours. Boundary F1 with tolerance should accompany exact Dice.
- Split metadata do not identify patient/examination groups. The split is image-level; group leakage cannot be excluded.
- The official test split was not evaluated or used for selection.
- The physical head-circumference estimate is ellipse-based and sensitive to resizing; native annotation contour calibration is substantially better than 256-pixel predicted-contour estimates. Treat model HC-error changes cautiously.
- Ablation and model comparisons are single-seed; bootstrap intervals describe image sampling, not training-seed uncertainty.
