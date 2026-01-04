# src/palette_quantization.py
import numpy as np
from collections import deque
from .utils import save_image

class MedianCutQuantizer:
    def __init__(self, n_colors=256):
        self.n_colors = n_colors

    def quantize(self, image):
        h, w, _ = image.shape
        pixels = image.reshape(-1, 3)

        # Create initial bucket
        bucket = [(0, 255, 0, 255, 0, 255, pixels.tolist())]  # r,g,b ranges + pixel list

        queue = deque(bucket)
        while len(queue) < self.n_colors:
            bucket = queue.popleft()
            r_min, r_max, g_min, g_max, b_min, b_max, pixel_list = bucket

            r_range = r_max - r_min
            g_range = g_max - g_min
            b_range = b_max - b_min

            # Split along longest dimension
            if r_range >= g_range and r_range >= b_range:
                pixel_list.sort(key=lambda p: p[0])
            elif g_range >= b_range:
                pixel_list.sort(key=lambda p: p[1])
            else:
                pixel_list.sort(key=lambda p: p[2])

            mid = len(pixel_list) // 2
            # Bucket 1
            bucket1_pixels = pixel_list[:mid]
            bucket1_r = [p[0] for p in bucket1_pixels]
            bucket1_g = [p[1] for p in bucket1_pixels]
            bucket1_b = [p[2] for p in bucket1_pixels]
            queue.append((min(bucket1_r), max(bucket1_r),
                          min(bucket1_g), max(bucket1_g),
                          min(bucket1_b), max(bucket1_b), bucket1_pixels))
            # Bucket 2
            bucket2_pixels = pixel_list[mid:]
            bucket2_r = [p[0] for p in bucket2_pixels]
            bucket2_g = [p[1] for p in bucket2_pixels]
            bucket2_b = [p[2] for p in bucket2_pixels]
            queue.append((min(bucket2_r), max(bucket2_r),
                          min(bucket2_g), max(bucket2_g),
                          min(bucket2_b), max(bucket2_b), bucket2_pixels))

        # Create palette
        palette = []
        for b in queue:
            pixels = np.array(b[6])
            color = np.mean(pixels, axis=0).astype(int)
            palette.append(color)
        palette = np.array(palette)

        # Map pixels to closest palette entry.
        # For large images computing a full (num_pixels x palette_size x 3)
        # distance array would OOM. Use a KD-tree with chunked queries if
        # SciPy is available; otherwise fall back to a small chunk-size
        # vectorized mapping.
        flat_pixels = image.reshape(-1, 3).astype(np.float32)
        palette_arr = palette.astype(np.float32)

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(palette_arr)
            # Query in reasonably large chunks to limit memory/time
            chunk_size = 1_000_000
            idx_chunks = []
            for i in range(0, flat_pixels.shape[0], chunk_size):
                pts = flat_pixels[i:i+chunk_size]
                _, idx = tree.query(pts, k=1)
                idx_chunks.append(idx)
            indices = np.concatenate(idx_chunks)
        except Exception:
            # Fallback: chunked vectorized distance computation with small chunks
            chunk_size = 30000
            idx_list = []
            for i in range(0, flat_pixels.shape[0], chunk_size):
                pts = flat_pixels[i:i+chunk_size]
                # distances shape: (chunk, palette_size)
                d = np.linalg.norm(pts[:, None, :] - palette_arr[None, :, :], axis=2)
                idx_list.append(np.argmin(d, axis=1))
            indices = np.concatenate(idx_list)

        quantized_flat = palette_arr[indices].astype(np.uint8)
        quantized = quantized_flat.reshape(image.shape)

        return quantized, palette

def compress_palette(image, save_path=None):
    quantizer = MedianCutQuantizer(256)
    quantized, palette = quantizer.quantize(image)
    # Ensure uint8 dtype for OpenCV saving and downstream metrics
    quantized = np.asarray(quantized, dtype=np.uint8)
    palette = np.asarray(palette, dtype=np.uint8)
    if save_path:
        save_image(quantized, save_path)
    original_bytes = image.size * 3
    compressed_bytes = image.shape[0] * image.shape[1] * 1 + len(palette) * 3  # index map + palette
    ratio = original_bytes / compressed_bytes
    return quantized, ratio