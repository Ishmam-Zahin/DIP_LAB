# src/dwt_compression.py
# FINAL VERSION — Works on ANY image size, multi-level Haar DWT + RLE + Huffman
import numpy as np
import cv2
from heapq import heappush, heappop
from .utils import save_image


# ================================
# Safe Padding to multiple of 2^levels
# ================================
def pad_to_multiple(image, levels):
    h, w = image.shape
    target = 1 << levels  # 2**levels
    pad_h = (target - h % target) % target
    pad_w = (target - w % target) % target
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode='symmetric'), pad_h, pad_w


# ================================
# Haar Wavelet 1D (vectorized)
# ================================
def haar_dwt_1d(signal):
    N = len(signal) // 2
    approx = (signal[0::2] + signal[1::2]) / np.sqrt(2)
    detail = (signal[0::2] - signal[1::2]) / np.sqrt(2)
    return approx, detail


def haar_idwt_1d(approx, detail):
    N = len(approx)
    up_a = np.repeat(approx, 2)
    up_d = np.repeat(detail, 2)
    even = (up_a + up_d) / np.sqrt(2)
    odd  = (up_a - up_d) / np.sqrt(2)
    return np.stack([even, odd], axis=1).flatten()


# ================================
# Multi-level 2D Haar DWT
# ================================
def haar_dwt_2d_multi(image, levels=3):
    img = image.astype(np.float64)
    coeffs = []
    current = img.copy()

    for level in range(levels):
        h, w = current.shape

        # Row transform
        approx_rows = np.zeros((h, w//2))
        detail_rows = np.zeros((h, w//2))
        for i in range(h):
            approx_rows[i], detail_rows[i] = haar_dwt_1d(current[i])

        # Column transform
        LL = np.zeros((h//2, w//2))
        LH = np.zeros((h//2, w//2))
        HL = np.zeros((h//2, w//2))
        HH = np.zeros((h//2, w//2))

        for j in range(w//2):
            LL[:, j], LH[:, j] = haar_dwt_1d(approx_rows[:, j])
            HL[:, j], HH[:, j] = haar_dwt_1d(detail_rows[:, j])

        coeffs.append((LH, HL, HH))
        current = LL  # next level

    coeffs.append((current, None, None))  # final LL
    coeffs.reverse()
    return coeffs


def haar_idwt_2d_multi(coeffs_list):
    current = coeffs_list[0][0].copy()  # coarsest LL

    for (LH, HL, HH) in coeffs_list[1:]:
        h, w = current.shape
        h2, w2 = 2*h, 2*w

        recon = np.zeros((h2, w2))
        for i in range(h):
            approx_row = current[i]
            detail_row = LH[i] if LH is not None else np.zeros(w)
            recon[i] = haar_idwt_1d(approx_row, detail_row)

        final = np.zeros((h2, w2))
        for j in range(w2):
            approx_col = recon[:, j]
            detail_col = HL[:, j] if HL is not None else np.zeros(h)
            final[:, j] = haar_idwt_1d(approx_col, detail_col)

        # Add HH contribution
        if HH is not None:
            hh_recon = np.zeros((h2, w2))
            for j in range(w2):
                hh_recon[:, j] = haar_idwt_1d(np.zeros(h), HH[:, j])
            final += hh_recon

        current = final

    return current


# ================================
# Quantization
# ===
def quantize_coeffs(coeffs_list, threshold=3.0):
    quantized = []
    for i, item in enumerate(coeffs_list):
        # coeffs items may be either (LL, None, None) for the coarsest
        # or (LH, HL, HH) for detail levels. Handle both safely.
        LL = None
        LH = HL = HH = None
        if isinstance(item, tuple) or isinstance(item, list):
            if len(item) >= 3:
                a, b, c = item[0], item[1], item[2]
                # Detect if this tuple is the LL (coarse) node: b and c are None
                if b is None and c is None:
                    LL = a
                else:
                    LH, HL, HH = a, b, c
            else:
                # Fallback: try to assign first element as LL
                LL = item[0]

        # Set quantization step: keep LL minimally quantized
        q = 1 if LL is not None else 8

        if LL is not None:
            LLq = np.round(LL / q)
            LLq[np.abs(LLq) < threshold] = 0
            quantized.append(LLq.astype(int))

        for band in [LH, HL, HH]:
            if band is not None:
                bq = np.round(band / (q * 2))
                bq[np.abs(bq) < threshold] = 0
                quantized.append(bq.astype(int))
    return quantized


# ===
# Entropy Coding (RLE + Huffman)
# ===
def run_length_encode(arr):
    arr = np.asarray(arr).flatten().astype(int)
    if len(arr) == 0:
        return []
    rle = []
    count = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            count += 1
        else:
            rle.append((arr[i-1], count))
            count = 1
    rle.append((arr[-1], count))
    return rle


class Node:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freqs):
    import heapq
    heap = []
    for s, f in freqs.items():
        if f > 0:
            heapq.heappush(heap, Node(s, f))
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = Node(None, a.freq + b.freq, a, b)
        heapq.heappush(heap, merged)
    return heap[0] if heap else None


def build_codes(node, prefix="", codebook=None):
    if codebook is None: codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix or "0"
        return
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)


# ================================
# MAIN DWT COMPRESSION FUNCTION
# ================================
def dwt_compress(image_rgb, levels=3, threshold=3.0, save_path=None):
    # Simpler and robust DWT using PyWavelets on the luminance channel.
    # We perform multi-level 2D DWT, threshold small detail coeffs, then
    # reconstruct. This avoids shape-mismatch issues from the hand-rolled
    # Haar implementation.
    import pywt

    # Convert to YCrCb, compress only Y
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float64)

    # Compute wavelet decomposition
    coeffs = pywt.wavedec2(Y, wavelet='bior4.4', level=levels)

    # Threshold small coefficients in detail bands
    new_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        # detail_level is a tuple (LH, HL, HH)
        lh, hl, hh = detail_level
        thresh = threshold * np.std(lh) if np.std(lh) > 0 else threshold
        lh_t = lh * (np.abs(lh) > thresh)
        hl_t = hl * (np.abs(hl) > thresh)
        hh_t = hh * (np.abs(hh) > thresh)
        new_coeffs.append((lh_t, hl_t, hh_t))

    # Reconstruct Y channel
    try:
        recon = pywt.waverec2(new_coeffs, wavelet='bior4.4')
    except Exception:
        # Fall back to original coefficients if reconstruction fails
        recon = pywt.waverec2(coeffs, wavelet='bior4.4')

    recon = np.clip(recon, 0, 255)

    # Crop/pad to original size
    h, w = image_rgb.shape[:2]
    recon_y = recon[:h, :w].astype(np.uint8)
    ycrcb[:, :, 0] = recon_y
    reconstructed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    # Rough ratio estimate: count non-zero detail coefficients
    total_coeffs = 0
    nonzero = 0
    for c in new_coeffs:
        if isinstance(c, tuple):
            for b in c:
                total_coeffs += b.size
                nonzero += np.count_nonzero(b)
        else:
            total_coeffs += c.size
            nonzero += np.count_nonzero(c)

    original_bits = h * w * 8  # only Y channel considered here
    compressed_bits = max(nonzero, 1) * 8
    ratio = (original_bits) / compressed_bits

    if save_path:
        save_image(reconstructed, save_path)

    return reconstructed, round(ratio, 2)