# main.py (DEBUGGED VERSION)
import os
import pandas as pd
from pathlib import Path
from src.utils import load_image, mse, psnr, ssim, plot_comparison
from src.rle_compression import compress_rle
from src.palette_quantization import compress_palette
from src.rgb332_quantization import compress_rgb332
from src.dct_compression import dct_compress
from src.dwt_compression import dwt_compress
from src.low_bit_quantization import reduce_to_n_bits

# Paths
IMAGE_DIR = Path("images")
RESULT_DIR = Path("results/compressed_images")
PLOT_DIR = Path("results/plots")
CSV_PATH = Path("results/metrics_all.csv")

# Create directories
for dir_path in [IMAGE_DIR, RESULT_DIR, PLOT_DIR, Path("results")]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"IMAGE_DIR: {IMAGE_DIR.absolute()}")

# Find images (case-insensitive .jpg/.png)
image_files = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.JPG")) + \
              list(IMAGE_DIR.glob("*.jpeg")) + list(IMAGE_DIR.glob("*.JPEG")) + \
              list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.PNG"))

print(f"Found {len(image_files)} images: {[f.name for f in image_files]}")
# If no images turminate this script
if not image_files:
    exit(1)


records = []

for img_path in image_files:
    print(f"\n--- Processing {img_path.name} ---")
    try:
        original = load_image(str(img_path))
        print(f"Loaded image shape: {original.shape}")
    except Exception as e:
        print(f"ERROR loading {img_path.name}: {e}")
        continue

    methods = []  # For plotting

    # 1. RLE
    print("Running RLE...")
    rec_rle, ratio_rle = compress_rle(original, RESULT_DIR / f"{img_path.stem}_rle.png")
    mse_rle = mse(original, rec_rle)
    psnr_rle = psnr(original, rec_rle)
    ssim_rle = ssim(original, rec_rle)
    records.append({
        "Image": img_path.name, "Method": "RLE",
        "Ratio": f"{ratio_rle:.2f}", "MSE": f"{mse_rle:.2f}",
        "PSNR": f"{psnr_rle:.2f}", "SSIM": f"{ssim_rle:.4f}"
    })
    methods.append((rec_rle, "RLE"))
    print(f"RLE: Ratio={ratio_rle:.2f}, MSE={mse_rle:.2f}, PSNR={psnr_rle:.2f}, SSIM={ssim_rle:.4f}")

    # 2. Palette 256
    print("Running Palette 256...")
    rec_pal, ratio_pal = compress_palette(original, RESULT_DIR / f"{img_path.stem}_palette.png")
    mse_pal = mse(original, rec_pal)
    psnr_pal = psnr(original, rec_pal)
    ssim_pal = ssim(original, rec_pal)
    records.append({
        "Image": img_path.name, "Method": "Palette 256",
        "Ratio": f"{ratio_pal:.2f}", "MSE": f"{mse_pal:.2f}",
        "PSNR": f"{psnr_pal:.2f}", "SSIM": f"{ssim_pal:.4f}"
    })
    methods.append((rec_pal, "Palette 256"))
    print(f"Palette: Ratio={ratio_pal:.2f}, MSE={mse_pal:.2f}, PSNR={psnr_pal:.2f}, SSIM={ssim_pal:.4f}")

    # 3. RGB332
    print("Running RGB332...")
    rec_332, _ = compress_rgb332(original, RESULT_DIR / f"{img_path.stem}_rgb332.png")
    mse_332 = mse(original, rec_332)
    psnr_332 = psnr(original, rec_332)
    ssim_332 = ssim(original, rec_332)
    records.append({
        "Image": img_path.name, "Method": "RGB332",
        "Ratio": "3.00", "MSE": f"{mse_332:.2f}",
        "PSNR": f"{psnr_332:.2f}", "SSIM": f"{ssim_332:.4f}"
    })
    methods.append((rec_332, "RGB332"))
    print(f"RGB332: Ratio=3.00, MSE={mse_332:.2f}, PSNR={psnr_332:.2f}, SSIM={ssim_332:.4f}")

    # 4. DCT 
    print("Running DCT (Full JPEG-like)...")
    rec_dct, ratio_dct = dct_compress(original, quality=50, save_path=RESULT_DIR / f"{img_path.stem}_dct_full.jpg")
    methods.append((rec_dct, "DCT (Full)"))  # For plotting

    mse_dct = mse(original, rec_dct)
    psnr_dct = psnr(original, rec_dct)
    ssim_dct = ssim(original, rec_dct)

    records.append({
        "Image": img_path.name, "Method": "DCT",
        "Ratio": f"{ratio_dct:.2f}", "MSE": f"{mse_dct:.2f}",
        "PSNR": f"{psnr_dct:.2f}", "SSIM": f"{ssim_dct:.4f}"
    })

    print(f"DCT: Ratio={ratio_dct:.2f}, MSE={mse_dct:.2f}, PSNR={psnr_dct:.2f}, SSIM={ssim_dct:.4f}")
    
    # 5. DWT
    print("Running DWT...")
    rec_dwt, ratio_dwt = dwt_compress(original, levels=3, threshold=3.0, save_path=RESULT_DIR / f"{img_path.stem}_dwt_full.png")
    mse_dwt = mse(original, rec_dwt)
    psnr_dwt = psnr(original, rec_dwt)
    ssim_dwt = ssim(original, rec_dwt)
    records.append({
        "Image": img_path.name, "Method": "DWT",
        "Ratio": f"{ratio_dwt:.2f}", "MSE": f"{mse_dwt:.2f}",
        "PSNR": f"{psnr_dwt:.2f}", "SSIM": f"{ssim_dwt:.4f}"
    })
    methods.append((rec_dwt, "DWT"))
    print(f"DWT: Ratio={ratio_dwt:.2f}, MSE={mse_dwt:.2f}, PSNR={psnr_dwt:.2f}, SSIM={ssim_dwt:.4f}")

    # 6. Low bit depths
    for bits in [8, 5, 4, 2, 1]:
        print(f"Running {bits}-bit...")
        rec_low, _ = reduce_to_n_bits(original, bits, RESULT_DIR / f"{img_path.stem}_{bits}bit.png")
        mse_low = mse(original, rec_low)
        psnr_low = psnr(original, rec_low)
        ssim_low = ssim(original, rec_low)
        records.append({
            "Image": img_path.name, "Method": f"{bits}-bit",
            "Ratio": f"{24/bits:.2f}", "MSE": f"{mse_low:.2f}",
            "PSNR": f"{psnr_low:.2f}", "SSIM": f"{ssim_low:.4f}"
        })
        methods.append((rec_low, f"{bits}-bit"))
        print(f"{bits}-bit: Ratio={24/bits:.2f}, MSE={mse_low:.2f}, PSNR={psnr_low:.2f}, SSIM={ssim_low:.4f}")

    # Save comparison plot (limit to first 5 methods for large plots)
    imgs = [original] + [m[0] for m in methods[:4]]  # Original + first 4
    titles = ["Original"] + [m[1] for m in methods[:4]]
    plot_comparison(original, [m[0] for m in methods[:4]], [m[1] for m in methods[:4]], PLOT_DIR / f"{img_path.stem}_comparison.png")
    print(f"Saved plot: {PLOT_DIR / f'{img_path.stem}_comparison.png'}")

print(f"\nTotal records added: {len(records)}")


# Save final CSV
if records:
    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nCSV saved! Rows: {len(df)}")
    print(df.head().to_string())  # Preview
else:
    print("WARNING: No records to save! CSV remains empty.")