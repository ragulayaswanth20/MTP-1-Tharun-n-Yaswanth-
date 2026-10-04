"""
GRN Denoising Pipeline for CETUS Cardiac Ultrasound Slices
===========================================================
Trains UNetGenerator+PatchDiscriminator+Segmenter on CETUS slices
with synthetic speckle noise, then runs inference and evaluates.
"""

import os, sys, glob, csv, json, time, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = "/home/teaching/MTP_1_Sneha/Denoising/data/cetus_p01"
ORIG_DIR   = os.path.join(DATA_DIR, "original")
SEG_DIR    = os.path.join(DATA_DIR, "seg")
OUT_DIR    = "/home/teaching/MTP_1_Sneha/Denoising/output/grn_results"
CKPT_DIR   = "/home/teaching/MTP_1_Sneha/Denoising/output/grn_checkpoints"
VIS_DIR    = os.path.join(OUT_DIR, "visuals")

for d in [OUT_DIR, CKPT_DIR, VIS_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eval_denoising import (
    add_speckle_noise, compute_psnr, compute_ssim, compute_rmse,
    compute_enl, compute_cnr, compute_epi, save_residual_map,
)

# ---------------------------------------------------------------------------
# Hyperparameters (defaults; can override via CLI)
# ---------------------------------------------------------------------------
IMAGE_SIZE    = 256
BATCH_SIZE    = 4
NUM_EPOCHS    = 30
LR            = 2e-4
LAMBDA_ADV    = 1.0
LAMBDA_PIXEL  = 100.0
LAMBDA_SEG    = 50.0
SEED          = 42
SPECKLE_SIGMA = 0.2

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CEUSDataset(Dataset):
    def __init__(self, image_dir, seg_dir, image_size=256):
        self.image_dir = image_dir
        self.seg_dir   = seg_dir
        self.files     = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.tfm = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize([0.5],[0.5])])
        self.mtfm = T.Compose([T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname     = os.path.basename(self.files[idx])
        clean_np  = np.array(Image.open(self.files[idx]).convert("L"))
        noisy_np  = add_speckle_noise(clean_np, sigma=SPECKLE_SIGMA, seed=SEED+idx)
        seg_path  = os.path.join(self.seg_dir, fname)
        mask_pil  = Image.open(seg_path).convert("L") if os.path.exists(seg_path) else Image.fromarray(np.zeros_like(clean_np))
        clean_t   = self.tfm(Image.fromarray(clean_np))
        noisy_t   = self.tfm(Image.fromarray(noisy_np))
        mask_t    = (self.mtfm(mask_pil) > 0).long().squeeze(0)
        return clean_t, noisy_t, mask_t, fname

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1,bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,1,1,bias=False), nn.BatchNorm2d(ch))
    def forward(self, x): return x + self.b(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ngf=64):
        super().__init__()
        self.e0 = nn.Sequential(nn.Conv2d(in_ch, ngf, 4,2,1), nn.ReLU(True))
        self.e1 = nn.Sequential(ResBlock(ngf),    nn.Conv2d(ngf,   ngf*2, 4,2,1), nn.BatchNorm2d(ngf*2),  nn.ReLU(True))
        self.e2 = nn.Sequential(ResBlock(ngf*2),  nn.Conv2d(ngf*2, ngf*4, 4,2,1), nn.BatchNorm2d(ngf*4),  nn.ReLU(True))
        self.e3 = nn.Sequential(ResBlock(ngf*4),  nn.Conv2d(ngf*4, ngf*8, 4,2,1), nn.BatchNorm2d(ngf*8),  nn.ReLU(True))
        self.e4 = nn.Sequential(ResBlock(ngf*8),  nn.Conv2d(ngf*8, 512,   4,2,1), nn.BatchNorm2d(512),    nn.ReLU(True))
        self.bt = nn.Sequential(ResBlock(512), ResBlock(512), ResBlock(512))
        self.d4 = nn.Sequential(nn.ConvTranspose2d(512,   ngf*8, 4,2,1), nn.BatchNorm2d(ngf*8), nn.ReLU(True))
        self.d3 = nn.Sequential(nn.ConvTranspose2d(ngf*16,ngf*4, 4,2,1), nn.BatchNorm2d(ngf*4), nn.ReLU(True))
        self.d2 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*2, 4,2,1), nn.BatchNorm2d(ngf*2), nn.ReLU(True))
        self.d1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf,   4,2,1), nn.BatchNorm2d(ngf),   nn.ReLU(True))
        self.d0 = nn.Sequential(nn.ConvTranspose2d(ngf*2, out_ch,4,2,1), nn.Tanh())
    def forward(self, x):
        e0=self.e0(x); e1=self.e1(e0); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        b=self.bt(e4)
        d4=self.d4(b)
        d3=self.d3(torch.cat([d4,e3],1))
        d2=self.d2(torch.cat([d3,e2],1))
        d1=self.d1(torch.cat([d2,e1],1))
        return self.d0(torch.cat([d1,e0],1))

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,ndf,   4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(ndf,  ndf*2, 4,2,1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2,True),
            nn.Conv2d(ndf*2,ndf*4, 4,2,1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2,True),
            nn.Conv2d(ndf*4,ndf*8, 4,1,1), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2,True),
            nn.Conv2d(ndf*8,1,     4,1,1))
    def forward(self, x): return self.net(x)

class SimpleSegmenter(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(True),
            nn.Conv2d(64,32,3,1,1), nn.ReLU(True),
            nn.Conv2d(32,num_classes,1))
    def forward(self, x): return self.net(x)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(device, epochs, batch_size):
    ds     = CEUSDataset(ORIG_DIR, SEG_DIR, IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    S = SimpleSegmenter(2).to(device)

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5,0.999))
    opt_S = optim.Adam(S.parameters(), lr=LR, betas=(0.5,0.999))
    sch_G = optim.lr_scheduler.StepLR(opt_G,10,0.5)
    sch_D = optim.lr_scheduler.StepLR(opt_D,10,0.5)
    sch_S = optim.lr_scheduler.StepLR(opt_S,10,0.5)

    crit_adv = nn.MSELoss()
    crit_px  = nn.L1Loss()
    crit_seg = nn.CrossEntropyLoss()

    best = float("inf")
    print(f"\n{'='*60}")
    print(f"  GRN Training: {epochs} epochs | {len(ds)} slices | {device}")
    print(f"{'='*60}")

    for ep in range(epochs):
        G.train(); D.train(); S.train()
        tg=td=ts=0.0; t0=time.time()

        for clean_t, noisy_t, mask_t, _ in loader:
            clean_t=clean_t.to(device); noisy_t=noisy_t.to(device); mask_t=mask_t.to(device)

            # Discriminator
            den = G(noisy_t).detach()
            r   = D(clean_t); f = D(den)
            ld  = 0.5*(crit_adv(r,torch.ones_like(r))+crit_adv(f,torch.zeros_like(f)))
            opt_D.zero_grad(); ld.backward(); opt_D.step()

            # Generator
            den     = G(noisy_t)
            fv      = D(den)
            lg      = (LAMBDA_ADV*crit_adv(fv,torch.ones_like(fv))
                      +LAMBDA_PIXEL*crit_px(den,clean_t)
                      +LAMBDA_SEG*crit_seg(S(den),mask_t))
            opt_G.zero_grad(); lg.backward(); opt_G.step()

            # Segmenter
            with torch.no_grad(): den2=G(noisy_t)
            ls = crit_seg(S(den2.detach()),mask_t)
            opt_S.zero_grad(); ls.backward(); opt_S.step()

            tg+=lg.item(); td+=ld.item(); ts+=ls.item()

        sch_G.step(); sch_D.step(); sch_S.step()
        n=len(loader)
        print(f"  Ep [{ep+1:3d}/{epochs}]  G={tg/n:.3f}  D={td/n:.3f}  S={ts/n:.3f}  ({time.time()-t0:.1f}s)")
        if tg/n < best:
            best=tg/n
            torch.save(G.state_dict(), os.path.join(CKPT_DIR,"best_G.pth"))
            torch.save(D.state_dict(), os.path.join(CKPT_DIR,"best_D.pth"))
            torch.save(S.state_dict(), os.path.join(CKPT_DIR,"best_S.pth"))

    print(f"\nTraining done. Best G-loss: {best:.4f}  |  Checkpoints: {CKPT_DIR}\n")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def denoise_all(device):
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(os.path.join(CKPT_DIR,"best_G.pth"), map_location=device))
    G.eval()

    tfm = T.Compose([T.Resize((IMAGE_SIZE,IMAGE_SIZE)), T.ToTensor(), T.Normalize([0.5],[0.5])])
    den_dir = os.path.join(OUT_DIR,"denoised"); os.makedirs(den_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(ORIG_DIR,"*.png")))
    print(f"Running inference on {len(files)} slices...")
    with torch.no_grad():
        for idx, fp in enumerate(files):
            fname    = os.path.basename(fp)
            clean_np = np.array(Image.open(fp).convert("L"))
            noisy_np = add_speckle_noise(clean_np, sigma=SPECKLE_SIGMA, seed=SEED+idx)
            x = tfm(Image.fromarray(noisy_np)).unsqueeze(0).to(device)
            y = G(x).squeeze().cpu().numpy()
            y = np.clip((y*0.5+0.5)*255,0,255).astype(np.uint8)
            h,w = clean_np.shape
            Image.fromarray(y).resize((w,h),Image.BILINEAR).save(os.path.join(den_dir,fname))
    print(f"Denoised slices → {den_dir}")
    return den_dir

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(den_dir):
    import cv2
    files = sorted(glob.glob(os.path.join(ORIG_DIR,"*.png")))
    m = {k:[] for k in ["psnr","ssim","rmse","enl","cnr","epi"]}

    for idx, fp in enumerate(files):
        fname    = os.path.basename(fp)
        clean_np = np.array(Image.open(fp).convert("L"))
        noisy_np = add_speckle_noise(clean_np, sigma=SPECKLE_SIGMA, seed=SEED+idx)
        mask_np  = np.array(Image.open(os.path.join(SEG_DIR,fname)).convert("L")) > 0
        d_img    = np.array(Image.open(os.path.join(den_dir,fname)).convert("L"))
        if d_img.shape != clean_np.shape:
            d_img = np.array(Image.fromarray(d_img).resize((clean_np.shape[1],clean_np.shape[0]),Image.BILINEAR))

        if np.count_nonzero(mask_np) > 100:
            kern   = np.ones((15,15),np.uint8)
            dil    = cv2.dilate(mask_np.astype(np.uint8), kern, iterations=1)
            bg_roi = (dil>0) & (~mask_np)
            tg_roi = mask_np
        else:
            h,w    = clean_np.shape
            tg_roi = np.zeros_like(clean_np, dtype=bool)
            bg_roi = np.zeros_like(clean_np, dtype=bool)
            tg_roi[int(h*0.3):int(h*0.5), int(w*0.3):int(w*0.5)] = True
            bg_roi[int(h*0.55):int(h*0.75),int(w*0.55):int(w*0.75)] = True

        m["psnr"].append(compute_psnr(clean_np, d_img))
        m["ssim"].append(compute_ssim(clean_np, d_img))
        m["rmse"].append(compute_rmse(clean_np, d_img))
        m["enl"].append(compute_enl(d_img, tg_roi))
        m["cnr"].append(compute_cnr(d_img, tg_roi, bg_roi))
        m["epi"].append(compute_epi(clean_np, d_img, tg_roi))

        if idx == 40:
            Image.fromarray(clean_np).save(os.path.join(VIS_DIR,"slice40_clean.png"))
            Image.fromarray(noisy_np).save(os.path.join(VIS_DIR,"slice40_noisy.png"))
            Image.fromarray(d_img).save(os.path.join(VIS_DIR,"slice40_grn_denoised.png"))
            save_residual_map(clean_np, noisy_np, os.path.join(VIS_DIR,"slice40_noisy_residual.png"))
            save_residual_map(clean_np, d_img,    os.path.join(VIS_DIR,"slice40_grn_residual.png"))

    result = {
        "Method":                        "GRN (UNet+GAN)",
        "PSNR (dB) \u2191":              f"{np.mean(m['psnr']):.2f} \u00b1 {np.std(m['psnr']):.2f}",
        "SSIM \u2191":                   f"{np.mean(m['ssim']):.4f} \u00b1 {np.std(m['ssim']):.4f}",
        "RMSE \u2193":                   f"{np.mean(m['rmse']):.2f} \u00b1 {np.std(m['rmse']):.2f}",
        "ENL (Speckle Smoothness) \u2191": f"{np.mean(m['enl']):.2f} \u00b1 {np.std(m['enl']):.2f}",
        "CNR (Tissue Contrast) \u2191":    f"{np.mean(m['cnr']):.2f} \u00b1 {np.std(m['cnr']):.2f}",
        "EPI (Edge Preservation) \u2191":  f"{np.mean(m['epi']):.3f} \u00b1 {np.std(m['epi']):.3f}",
    }

    print("\n"+"="*65)
    print("      GRN DENOISING RESULTS — CETUS CARDIAC ULTRASOUND")
    print("="*65)
    for k,v in result.items():
        print(f"  {k:<40} {v}")
    print("="*65+"\n")

    with open(os.path.join(OUT_DIR,"grn_metrics.json"),"w") as f: json.dump(result,f,indent=2)
    with open(os.path.join(OUT_DIR,"grn_metrics.csv"), "w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(result.keys())); w.writeheader(); w.writerow(result)
    print(f"Results saved to {OUT_DIR}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    args = p.parse_args()

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.skip_train:
        train(device, args.epochs, args.batch_size)

    den_dir = denoise_all(device)
    evaluate(den_dir)
