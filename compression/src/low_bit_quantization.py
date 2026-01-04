# src/low_bit_quantization.py
import numpy as np
from .utils import save_image

def reduce_to_n_bits(image, bits, save_path=None):
    shift = 8 - bits
    quantized = (image >> shift) << shift
    quantized = quantized.astype(np.uint8)
    if save_path:
        save_image(quantized, save_path)
    ratio = 24.0 / bits
    return quantized, ratio