# MTP-1-Tharun-n-Yaswanth-

Research workspace for ultrasound despeckling, segmentation, and 3D ultrasound reconstruction experiments.

## Projects

- `APD-Net/`: anatomy-preserving ultrasound denoising and APD-GRN experiments, including HC18 validation diagnostics.
- `HC18_Research/`: HC18 split generation, segmentation/denoising experiments, and research reports.
- `GRN/`, `MedGS/`, `UltraGS/`: upstream projects tracked as Git submodules.
- `scripts/`, `utils/`, and the root LaTeX files: experiment orchestration and report materials.

The raw datasets, downloaded archives, trained checkpoints, generated output trees, Python caches, and paper PDFs are intentionally excluded by `.gitignore`. Obtain each dataset from its approved source and configure paths locally before running experiments.

## APD-Net diagnostics

The APD-Net project contains the current training and validation instructions in `APD-Net/README.md`. Its HC18/APD-GRN diagnostic outputs are summarized in `APD-Net/results/diagnostics/diagnostic_report.md`.

## Submodules

Initialize the upstream project checkouts after cloning:

```bash
git submodule update --init --recursive
```