# src/dct_compression.py
# Full JPEG-like DCT compression (based on your great implementation)
import cv2
import numpy as np
import math
from heapq import heappush, heappop
from .utils import save_image

# ==========================================================
# DCT, IDCT, Zigzag, RLE, Huffman (exactly your code)
# ==========================================================
def dct_1d(vector):
    N = len(vector)
    result = np.zeros(N)
    for u in range(N):
        alpha = math.sqrt(1/N) if u == 0 else math.sqrt(2/N)
        sum_val = sum(vector[x] * math.cos(((2*x + 1) * u * math.pi) / (2 * N)) for x in range(N))
        result[u] = alpha * sum_val
    return result

def dct_2d(block):
    temp = np.apply_along_axis(dct_1d, 0, block)
    return np.apply_along_axis(dct_1d, 1, temp)

def idct_1d(vector):
    N = len(vector)
    result = np.zeros(N)
    for x in range(N):
        sum_val = sum((math.sqrt(1/N) if u == 0 else math.sqrt(2/N)) * vector[u] *
                      math.cos(((2*x + 1) * u * math.pi) / (2 * N)) for u in range(N))
        result[x] = sum_val
    return result

def idct_2d(block):
    temp = np.apply_along_axis(idct_1d, 0, block)
    return np.apply_along_axis(idct_1d, 1, temp)

# Quantization table (standard JPEG luminance)
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

zigzag_indices = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]

def zigzag(block):
    return np.array([block[i,j] for i,j in zigzag_indices])

def inverse_zigzag(arr):
    block = np.zeros((8,8))
    for idx, (i,j) in enumerate(zigzag_indices):
        block[i,j] = arr[idx]
    return block

def run_length_encode(arr):
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
    heap = []
    for symbol, freq in freqs.items():
        heappush(heap, Node(symbol, freq))
    while len(heap) > 1:
        n1 = heappop(heap)
        n2 = heappop(heap)
        heappush(heap, Node(None, n1.freq + n2.freq, n1, n2))
    return heap[0]

def build_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix or '0'
        return codebook
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)
    return codebook

# ==========================================================
# Main compression function (RGB input → returns reconstructed RGB + real ratio)
# ==========================================================
def dct_compress(image_rgb, quality=50, save_path=None):
    # Scale quantization table with quality
    scale = 50 / quality if quality < 50 else 200 - 2 * quality
    Q_scaled = np.clip(Q * scale, 1, 255)

    # Convert to YCrCb and only compress Y channel (like real JPEG)
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    Y = ycrcb[:,:,0].astype(np.float32)
    h, w = Y.shape

    # Padding
    ph = ((h + 7) // 8) * 8
    pw = ((w + 7) // 8) * 8
    padded = np.pad(Y, ((0, ph-h), (0, pw-w)), mode='constant')

    # DCT + Quantization + Zigzag
    quantized_blocks = []
    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            block = padded[i:i+8, j:j+8] - 128
            dct_block = dct_2d(block)
            quant = np.round(dct_block / Q_scaled)
            zz = zigzag(quant)
            quantized_blocks.append(zz.astype(int))

    # RLE + Huffman
    all_coeffs = np.concatenate(quantized_blocks)
    rle = run_length_encode(all_coeffs)
    freqs = {}
    for val, count in rle:
        freqs[val] = freqs.get(val, 0) + count
    tree = build_huffman_tree(freqs)
    codes = build_codes(tree)

    bitstream = ''.join(codes[val] * count for val, count in rle)

    # === Decompression (to get reconstructed image) ===
    inv_codes = {v: k for k, v in codes.items()}
    curr = ""
    decoded = []
    for bit in bitstream:
        curr += bit
        if curr in inv_codes:
            decoded.append(inv_codes[curr])
            curr = ""
    decoded = np.array(decoded, dtype=int)

    # Reconstruct Y channel
    recon_y = np.zeros((ph, pw))
    idx = 0
    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            block = inverse_zigzag(decoded[idx:idx+64])
            dequant = block * Q_scaled
            idct_block = idct_2d(dequant) + 128
            recon_y[i:i+8, j:j+8] = np.clip(idct_block, 0, 255)
            idx += 64

    recon_y = recon_y[:h, :w].astype(np.uint8)
    ycrcb[:, :, 0] = recon_y
    reconstructed_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    # Real compression ratio (only Y channel is compressed, Cr/Cb unchanged)
    original_bits = h * w * 24
    compressed_bits = len(bitstream) + (h * w * 16)  # +16 bits/pixel for Cr+Cb
    ratio = original_bits / max(compressed_bits, 1)

    if save_path:
        save_image(reconstructed_rgb, save_path)

    return reconstructed_rgb, round(ratio, 2)