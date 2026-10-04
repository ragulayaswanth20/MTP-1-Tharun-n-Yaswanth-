import os
import numpy as np
import cv2
from scipy.ndimage import laplace
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim_metric
from PIL import Image

def add_speckle_noise(image: np.ndarray, sigma: float = 0.2, seed: int = 42) -> np.ndarray:
    """
    Inject multiplicative ultrasound speckle noise:
    I_noisy = I + I * noise, where noise ~ N(0, sigma^2).
    """
    rng = np.random.RandomState(seed)
    img_f = image.astype(np.float32) / 255.0
    noise = rng.normal(0, sigma, img_f.shape).astype(np.float32)
    noisy = img_f + img_f * noise
    noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    return noisy

def compute_psnr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    clean_f = clean.astype(np.float32)
    denoised_f = denoised.astype(np.float32)
    mse = np.mean((clean_f - denoised_f) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))

def compute_ssim(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Structural Similarity Index (SSIM)."""
    return float(ssim_metric(clean, denoised, data_range=255.0))

def compute_rmse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Root Mean Square Error."""
    clean_f = clean.astype(np.float32)
    denoised_f = denoised.astype(np.float32)
    return float(np.sqrt(np.mean((clean_f - denoised_f) ** 2)))

def compute_enl(image: np.ndarray, roi_mask: np.ndarray = None) -> float:
    """
    Equivalent Number of Looks (ENL) = mu^2 / sigma^2 in homogeneous region.
    Higher ENL means smoother speckle reduction in uniform tissue.
    """
    img_f = image.astype(np.float32)
    if roi_mask is not None:
        pixels = img_f[roi_mask > 0]
    else:
        # If no explicit mask, take a central homogeneous region with low gradients
        h, w = img_f.shape
        pixels = img_f[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    
    mean = np.mean(pixels)
    var = np.var(pixels)
    if var == 0:
        return float('inf')
    return float((mean ** 2) / var)

def compute_cnr(image: np.ndarray, target_roi: np.ndarray, background_roi: np.ndarray) -> float:
    """
    Contrast-to-Noise Ratio: |mu1 - mu2| / sqrt(var1 + var2).
    Measures organ/tissue contrast against cavity.
    """
    img_f = image.astype(np.float32)
    t_pixels = img_f[target_roi > 0]
    b_pixels = img_f[background_roi > 0]
    
    if len(t_pixels) == 0 or len(b_pixels) == 0:
        return 0.0
        
    mu_t, var_t = np.mean(t_pixels), np.var(t_pixels)
    mu_b, var_b = np.mean(b_pixels), np.var(b_pixels)
    denom = np.sqrt(var_t + var_b)
    if denom == 0:
        return 0.0
    return float(np.abs(mu_t - mu_b) / denom)

def compute_epi(original: np.ndarray, denoised: np.ndarray, roi_mask: np.ndarray = None) -> float:
    """
    Edge Preservation Index (EPI) via correlation of Laplacian high-pass filters.
    Value close to 1.0 indicates anatomical boundaries were preserved without blurring.
    """
    orig_f = original.astype(np.float32)
    denoised_f = denoised.astype(np.float32)
    
    del_orig = laplace(orig_f)
    del_denoised = laplace(denoised_f)
    
    if roi_mask is not None:
        p1 = del_orig[roi_mask > 0]
        p2 = del_denoised[roi_mask > 0]
    else:
        p1 = del_orig.flatten()
        p2 = del_denoised.flatten()
        
    p1 = p1 - np.mean(p1)
    p2 = p2 - np.mean(p2)
    denom = np.sqrt(np.sum(p1 ** 2) * np.sum(p2 ** 2))
    if denom == 0:
        return 1.0
    return float(np.sum(p1 * p2) / denom)

# Classical baseline filters
def filter_tv_chambolle(image: np.ndarray, weight: float = 0.15) -> np.ndarray:
    """Total Variation Chambolle filter (used in MedGS)."""
    img_f = image.astype(np.float32) / 255.0
    denoised_f = denoise_tv_chambolle(img_f, weight=weight)
    return np.clip(denoised_f * 255.0, 0, 255).astype(np.uint8)

def filter_bilateral(image: np.ndarray, d: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """Bilateral edge-preserving filter."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def filter_gaussian(image: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Standard Gaussian smoothing filter."""
    return cv2.GaussianBlur(image, (5, 5), sigma)

def save_residual_map(reference: np.ndarray, denoised: np.ndarray, out_path: str):
    """Saves normalized visual error residual map."""
    diff = np.abs(reference.astype(np.float32) - denoised.astype(np.float32))
    diff_norm = np.clip(diff * 2.5, 0, 255).astype(np.uint8) # scale up slightly for visibility
    # Use COLORMAP_JET to highlight residual intensity
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    cv2.imwrite(out_path, diff_color)
