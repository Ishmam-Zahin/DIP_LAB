import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil

def compute_hist_mapping(tile):
    flat = tile.flatten()
    hist = np.bincount(flat, minlength=256).astype(np.int64)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min() if cdf.sum() > 0 else 0
    denom = tile.size - cdf_min
    if denom <= 0:
        return np.arange(256, dtype=np.uint8)
    mapping = np.floor(255.0 * (cdf - cdf_min) / denom).clip(0,255).astype(np.uint8)
    return mapping

def tile_grid_indices(h, w, s):
    th, tw = h // s, w // s
    heights = [th] * s
    widths = [tw] * s
    for i in range(h - th * s):
        heights[s - 1 - i] += 1
    for j in range(w - tw * s):
        widths[s - 1 - j] += 1
    row_starts = [0]
    for ht in heights[:-1]:
        row_starts.append(row_starts[-1] + ht)
    col_starts = [0]
    for wd in widths[:-1]:
        col_starts.append(col_starts[-1] + wd)
    return heights, widths, row_starts, col_starts

def global_hist_equalize(img):
    return cv2.equalizeHist(img)

def ahe_naive(img, s):
    h, w = img.shape
    heights, widths, row_starts, col_starts = tile_grid_indices(h, w, s)
    out = np.empty_like(img)
    for i, ys in enumerate(row_starts):
        for j, xs in enumerate(col_starts):
            ht = heights[i]
            wd = widths[j]
            tile = img[ys:ys+ht, xs:xs+wd]
            out[ys:ys+ht, xs:xs+wd] = cv2.equalizeHist(tile)
    return out

def ahe_bilinear(img, s):
    h, w = img.shape
    heights, widths, row_starts, col_starts = tile_grid_indices(h, w, s)
    mappings = [[None]*s for _ in range(s)]
    row_ends = [row_starts[i] + heights[i] for i in range(s)]
    col_ends = [col_starts[j] + widths[j] for j in range(s)]
    for i in range(s):
        for j in range(s):
            ys = row_starts[i]; ye = row_ends[i]
            xs = col_starts[j]; xe = col_ends[j]
            mappings[i][j] = compute_hist_mapping(img[ys:ye, xs:xe])

    out = np.empty_like(img, dtype=np.uint8)
    row_idx_for_r = np.empty(h, dtype=np.int32)
    for i in range(s):
        row_idx_for_r[row_starts[i]:row_ends[i]] = i
    col_idx_for_c = np.empty(w, dtype=np.int32)
    for j in range(s):
        col_idx_for_c[col_starts[j]:col_ends[j]] = j

    dy_for_r = np.empty(h, dtype=np.float32)
    for r in range(h):
        i = row_idx_for_r[r]
        dy_for_r[r] = (r - row_starts[i]) / max(1.0, heights[i] - 1.0)
    dx_for_c = np.empty(w, dtype=np.float32)
    for c in range(w):
        j = col_idx_for_c[c]
        dx_for_c[c] = (c - col_starts[j]) / max(1.0, widths[j] - 1.0)

    for r in range(h):
        i = int(row_idx_for_r[r])
        dy = float(dy_for_r[r])
        i0 = i; i1 = min(i+1, s-1)
        wy0, wy1 = 1-dy, dy
        for c in range(w):
            j = int(col_idx_for_c[c])
            dx = float(dx_for_c[c])
            j0 = j; j1 = min(j+1, s-1)
            wx0, wx1 = 1-dx, dx
            w00, w01, w10, w11 = wy0*wx0, wy0*wx1, wy1*wx0, wy1*wx1
            v = int(img[r, c])
            val = (w00*mappings[i0][j0][v] + w01*mappings[i0][j1][v] +
                   w10*mappings[i1][j0][v] + w11*mappings[i1][j1][v])
            out[r, c] = int(round(val))
    return out

def clahe_with_clip(img, s, clip):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(s, s))
    return clahe.apply(img)

def make_composite(results, out_path='composite_gray.png', max_cols=3):
    names = list(results.keys())
    imgs = [results[n] for n in names]
    h, w = imgs[0].shape
    rows = int(np.ceil(len(imgs)/max_cols))
    fig, axes = plt.subplots(rows, max_cols, figsize=(4*max_cols, 3*rows))
    axes = axes.flatten()

    for ax in axes:
        ax.axis('off')

    for i, (name, im) in enumerate(zip(names, imgs)):
        axes[i].imshow(im, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(name, fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved composite to {out_path}")


def main():
    in_file = '/home/zahin/Desktop/DIP_LAB/images/img5.jpg'
    s = 8
    clip_values = [2.0, 4.0, 8.0]
    os.makedirs('outputs', exist_ok=True)
    img = cv2.imread(in_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Place a grayscale or color image as input.jpg")

    results = {'Original': img}
    results['Global HE'] = global_hist_equalize(img)
    results['AHE Naive'] = ahe_naive(img, s)
    results['AHE Bilinear'] = ahe_bilinear(img, s)
    for clip in clip_values:
        results[f'CLAHE clip={clip}'] = clahe_with_clip(img, s, clip)

    make_composite(results, out_path='outputs/composite_gray.png')

if __name__ == '__main__':
    main()
