# src/rgb332_quantization.py
import numpy as np
from .utils import save_image

def compress_rgb332(image, save_path=None):
    # 3 bits R, 3 bits G, 2 bits B
    r = (image[..., 0] >> 5) << 5
    g = (image[..., 1] >> 5) << 5
    b = (image[..., 2] >> 6) << 6
    quantized = np.stack([r, g, b], axis=-1).astype(np.uint8)
    if save_path:
        save_image(quantized, save_path)
    ratio = (image.size * 3) / (image.size)  # 24 → 8 bit
    return quantized, ratio