import os
import sys
import glob
import numpy as np
from PIL import Image
import csv
import json
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eval_denoising import (
    add_speckle_noise,
    compute_psnr,
    compute_ssim,
    compute_rmse,
    compute_enl,
    compute_cnr,
    compute_epi,
    filter_tv_chambolle,
    filter_bilateral,
    filter_gaussian,
    save_residual_map
)

def benchmark_slice_sequence():
    data_dir = "/home/teaching/MTP_1_Sneha/Denoising/data/cetus_p01"
    orig_dir = os.path.join(data_dir, "original")
    seg_dir = os.path.join(data_dir, "seg")
    
    out_dir = "/home/teaching/MTP_1_Sneha/Denoising/output/benchmark_results"
    vis_dir = os.path.join(out_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)
    
    slice_files = sorted(glob.glob(os.path.join(orig_dir, "*.png")))
    print(f"Running evaluation across {len(slice_files)} CETUS cardiac ultrasound slices...")
    
    methods = ["Noisy (Baseline)", "Gaussian Filter", "Bilateral Filter", "TV-Chambolle Filter"]
    
    results = {m: {"psnr": [], "ssim": [], "rmse": [], "enl": [], "cnr": [], "epi": []} for m in methods}
    
    for idx, fpath in enumerate(slice_files):
        fname = os.path.basename(fpath)
        clean = np.array(Image.open(fpath).convert("L"))
        mask = np.array(Image.open(os.path.join(seg_dir, fname)).convert("L")) > 0
        
        # Inject controlled synthetic speckle noise (sigma = 0.2)
        noisy = add_speckle_noise(clean, sigma=0.2, seed=42 + idx)
        
        # Denoise using candidate classical methods
        denoised_dict = {
            "Noisy (Baseline)": noisy,
            "Gaussian Filter": filter_gaussian(noisy, sigma=1.2),
            "Bilateral Filter": filter_bilateral(noisy, d=7, sigma_color=50.0, sigma_space=50.0),
            "TV-Chambolle Filter": filter_tv_chambolle(noisy, weight=0.15),
        }
        
        # Define foreground (target) and background (cavity/exterior) for CNR
        target_roi = mask
        # If mask is empty in this slice, use synthetic tissue patch
        if np.count_nonzero(mask) > 100:
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            bg_roi = (dilated > 0) & (~mask)
        else:
            h, w = clean.shape
            target_roi = np.zeros_like(clean, dtype=bool)
            bg_roi = np.zeros_like(clean, dtype=bool)
            target_roi[int(h*0.3):int(h*0.5), int(w*0.3):int(w*0.5)] = True
            bg_roi[int(h*0.55):int(h*0.75), int(w*0.55):int(w*0.75)] = True
            
        for m_name, d_img in denoised_dict.items():
            results[m_name]["psnr"].append(compute_psnr(clean, d_img))
            results[m_name]["ssim"].append(compute_ssim(clean, d_img))
            results[m_name]["rmse"].append(compute_rmse(clean, d_img))
            results[m_name]["enl"].append(compute_enl(d_img, target_roi))
            results[m_name]["cnr"].append(compute_cnr(d_img, target_roi, bg_roi))
            results[m_name]["epi"].append(compute_epi(clean, d_img, target_roi))
            
        # Save visual comparison for the central slice (slice 40)
        if idx == 40:
            print(f"Saving visual comparison and residual maps for central slice {fname}...")
            Image.fromarray(clean).save(os.path.join(vis_dir, "slice40_clean.png"))
            Image.fromarray(noisy).save(os.path.join(vis_dir, "slice40_noisy.png"))
            save_residual_map(clean, noisy, os.path.join(vis_dir, "slice40_noisy_residual.png"))
            
            for m_name, d_img in denoised_dict.items():
                if m_name == "Noisy (Baseline)":
                    continue
                safe_name = m_name.lower().replace(" ", "_")
                Image.fromarray(d_img).save(os.path.join(vis_dir, f"slice40_{safe_name}.png"))
                save_residual_map(clean, d_img, os.path.join(vis_dir, f"slice40_{safe_name}_residual.png"))
                
    # Aggregate statistics
    summary = []
    for m_name in methods:
        summary.append({
            "Method": m_name,
            "PSNR (dB) ↑": f"{np.mean(results[m_name]['psnr']):.2f} ± {np.std(results[m_name]['psnr']):.2f}",
            "SSIM ↑": f"{np.mean(results[m_name]['ssim']):.4f} ± {np.std(results[m_name]['ssim']):.4f}",
            "RMSE ↓": f"{np.mean(results[m_name]['rmse']):.2f} ± {np.std(results[m_name]['rmse']):.2f}",
            "ENL (Speckle Smoothness) ↑": f"{np.mean(results[m_name]['enl']):.2f} ± {np.std(results[m_name]['enl']):.2f}",
            "CNR (Tissue Contrast) ↑": f"{np.mean(results[m_name]['cnr']):.2f} ± {np.std(results[m_name]['cnr']):.2f}",
            "EPI (Edge Preservation) ↑": f"{np.mean(results[m_name]['epi']):.3f} ± {np.std(results[m_name]['epi']):.3f}",
        })
        
    # Format and print table
    headers = list(summary[0].keys())
    col_widths = {h: max(len(h), max(len(row[h]) for row in summary)) + 2 for h in headers}
    
    header_str = "".join(h.ljust(col_widths[h]) for h in headers)
    sep_str = "-" * len(header_str)
    
    print("\n" + "="*len(header_str))
    print("      CETUS CARDIAC ULTRASOUND DENOISING BENCHMARK RESULTS")
    print("="*len(header_str))
    print(header_str)
    print(sep_str)
    for row in summary:
        print("".join(row[h].ljust(col_widths[h]) for h in headers))
    print("="*len(header_str) + "\n")
    
    # Save table to CSV and JSON
    csv_path = os.path.join(out_dir, "baseline_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary)
    with open(os.path.join(out_dir, "baseline_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_dir}.")

if __name__ == "__main__":
    benchmark_slice_sequence()
