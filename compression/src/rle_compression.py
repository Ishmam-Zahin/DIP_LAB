# src/rle_compression.py
import numpy as np
from .utils import save_image

def rle_encode(img):
    # Simple RLE on flattened array (lossless only for images with runs)
    flat = img.flatten()
    encoded = []
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == flat[i-1] and count < 255:
            count += 1
        else:
            encoded.extend([flat[i-1], count])
            count = 1
    encoded.extend([flat[-1], count])
    return np.array(encoded, dtype=np.uint8)

def rle_decode(encoded, shape):
    decoded = []
    for i in range(0, len(encoded), 2):
        decoded.extend([encoded[i]] * encoded[i+1])
    return np.array(decoded, dtype=np.uint8).reshape(shape)

def compress_rle(original_img, save_path=None):
    encoded = rle_encode(original_img)
    reconstructed = rle_decode(encoded, original_img.shape)
    if save_path:
        save_image(reconstructed, save_path)
    ratio = (original_img.size * original_img.itemsize) / len(encoded)
    return reconstructed, ratio