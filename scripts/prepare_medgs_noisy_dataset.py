import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eval_denoising import add_speckle_noise


SOURCE_DIR = "/home/teaching/MTP_1_Sneha/Denoising/data/cetus_p01"
TARGET_DIR = "/home/teaching/MTP_1_Sneha/Denoising/data/cetus_p01_medgs_noisy"
SPECKLE_SIGMA = 0.2
SEED = 42


def main():
    source_original = os.path.join(SOURCE_DIR, "original")
    target_original = os.path.join(TARGET_DIR, "original")
    os.makedirs(target_original, exist_ok=True)

    for filename in sorted(os.listdir(source_original)):
        if not filename.endswith(".png"):
            continue
        source_path = os.path.join(source_original, filename)
        clean = np.array(Image.open(source_path).convert("L"))
        noisy = add_speckle_noise(clean, sigma=SPECKLE_SIGMA, seed=SEED + int(filename[:-4]))
        Image.fromarray(noisy).save(os.path.join(target_original, filename))

    shutil.copy2(
        os.path.join(SOURCE_DIR, "transforms_train.json"),
        os.path.join(TARGET_DIR, "transforms_train.json"),
    )
    print(f"Prepared {len(os.listdir(target_original))} noisy frames in {TARGET_DIR}")


if __name__ == "__main__":
    main()
