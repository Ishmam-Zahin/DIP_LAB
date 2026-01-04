# src/utils.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim_metric
import pandas as pd

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image {path} not found")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def mse(original, compressed):
    return np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)

def psnr(original, compressed):
    mse_val = mse(original, compressed)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

# FIXED SSIM – works on any image size ≥ 7×7
def ssim(original, compressed):
    min_dim = min(original.shape[0], original.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)  # odd & ≤ image size
    try:
        return ssim_metric(
            original, compressed,
            win_size=win_size,
            channel_axis=2,          # NEW scikit-image requirement
            data_range=255
        )
    except Exception as e:
        print(f"SSIM failed (using win_size={win_size}): {e}")
        return 0.0  # fallback

def plot_comparison(original, compressed_list, titles, save_path):
    plt.figure(figsize=(20, 10))
    n = len(compressed_list) + 1
    plt.subplot(2, n, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')

    for i, (comp_img, title) in enumerate(zip(compressed_list, titles), 2):
        plt.subplot(2, n, i)
        plt.imshow(comp_img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()