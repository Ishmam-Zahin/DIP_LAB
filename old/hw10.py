import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Custom morphological functions
# ------------------------------

def custom_erode(img, kernel):
    pad_h, pad_w = kernel.shape[0]//2, kernel.shape[1]//2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.min(region[kernel == 1])
    return output

def custom_dilate(img, kernel):
    pad_h, pad_w = kernel.shape[0]//2, kernel.shape[1]//2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.max(region[kernel == 1])
    return output

def custom_open(img, kernel):
    return custom_dilate(custom_erode(img, kernel), kernel)

def custom_close(img, kernel):
    return custom_erode(custom_dilate(img, kernel), kernel)

def custom_tophat(img, kernel):
    return cv2.subtract(img, custom_open(img, kernel))

def custom_blackhat(img, kernel):
    return cv2.subtract(custom_close(img, kernel), img)


# ------------------------------
# Structuring elements
# ------------------------------

def diamond_kernel(size=5):
    assert size % 2 == 1, "Kernel size must be odd"
    k = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    for i in range(size):
        for j in range(size):
            if abs(i - mid) + abs(j - mid) <= mid:
                k[i, j] = 1
    return k

kernel_types = {
    "Rectangular": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    "Elliptical": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    "Cross": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
    "Diamond": diamond_kernel(5)
}

# ------------------------------
# Morphological operations
# ------------------------------

builtin_ops = {
    "Erosion": lambda i, k: cv2.erode(i, k),
    "Dilation": lambda i, k: cv2.dilate(i, k),
    "Opening": lambda i, k: cv2.morphologyEx(i, cv2.MORPH_OPEN, k),
    "Closing": lambda i, k: cv2.morphologyEx(i, cv2.MORPH_CLOSE, k),
    "Top-hat": lambda i, k: cv2.morphologyEx(i, cv2.MORPH_TOPHAT, k),
    "Black-hat": lambda i, k: cv2.morphologyEx(i, cv2.MORPH_BLACKHAT, k)
}

custom_ops = {
    "Erosion": custom_erode,
    "Dilation": custom_dilate,
    "Opening": custom_open,
    "Closing": custom_close,
    "Top-hat": custom_tophat,
    "Black-hat": custom_blackhat
}

# ------------------------------
# Load image
# ------------------------------

img = cv2.imread("../images/img6.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Please place '../images/img.jpg' in the correct path.")

# ------------------------------
# Plot both OpenCV and Custom side-by-side (clearly labeled)
# ------------------------------

num_ops = len(builtin_ops)
num_kernels = len(kernel_types)

# Make larger figure for readability
fig, axes = plt.subplots(num_ops * 2, num_kernels, figsize=(16, 8))
plt.suptitle("Morphological Transformations: OpenCV vs Custom Implementation", fontsize=20, y=0.93, weight="bold")

for row_idx, (op_name, builtin_func) in enumerate(builtin_ops.items()):
    custom_func = custom_ops[op_name]
    
    for col_idx, (k_name, k) in enumerate(kernel_types.items()):
        # --- OpenCV version ---
        built_result = builtin_func(img, k)
        ax = axes[row_idx * 2][col_idx]
        ax.imshow(built_result, cmap="gray")
        if col_idx == 0:
            ax.set_ylabel(f"{op_name}\n(OpenCV)", fontsize=12, weight="bold", color="navy")
        ax.set_title(k_name, fontsize=11)
        ax.axis("off")
        
        # --- Custom version ---
        cust_result = custom_func(img, k)
        ax = axes[row_idx * 2 + 1][col_idx]
        ax.imshow(cust_result, cmap="gray")
        if col_idx == 0:
            ax.set_ylabel(f"{op_name}\n(Custom)", fontsize=12, weight="bold", color="darkred")
        ax.axis("off")

# Add visual divider lines between operations
for i in range(1, num_ops):
    y_pos = (i * 2) / (num_ops * 2)
    fig.text(0.5, 1 - y_pos, " ", ha="center", va="center",
             backgroundcolor="lightgray", alpha=0.5, transform=fig.transFigure)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("../images/morph_compare_labeled.png", dpi=300)
# plt.show()

print("✅ Saved figure as '../images/morph_compare_labeled.png' with readable labels and clear grouping.")
