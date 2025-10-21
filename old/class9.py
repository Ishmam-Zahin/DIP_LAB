import cv2
import numpy as np
import matplotlib.pyplot as plt


def cal_hist_equalize(img):
    hist = np.zeros(256, dtype=float)
    for row in img:
        for val in row:
            hist[val] += 1
    pdf = hist / hist.sum()
    cdf = np.cumsum(pdf)
    return hist, cdf


def closest_value(val, img_ref_cdf):
    # Return the index (intensity) where ref CDF is closest to val
    idx = np.argmin(np.abs(img_ref_cdf - val))
    return idx


def main():
    img_src = cv2.imread('../images/img5.jpg', cv2.IMREAD_GRAYSCALE)
    img_ref = cv2.imread('../images/img6.jpg', cv2.IMREAD_GRAYSCALE)

    img_src_hist, img_src_cdf = cal_hist_equalize(img_src)
    img_ref_hist, img_ref_cdf = cal_hist_equalize(img_ref)

    # Build mapping LUT (0-255 → 0-255)
    mapping = np.zeros(256, dtype=np.uint8)
    for i, val in enumerate(img_src_cdf):
        mapping[i] = closest_value(val, img_ref_cdf)

    # Apply mapping on source image
    img_new = mapping[img_src]

    # Show results
    plt.subplot(1, 3, 1)
    plt.imshow(img_src, cmap='gray')
    plt.title("Source")

    plt.subplot(1, 3, 2)
    plt.imshow(img_ref, cmap='gray')
    plt.title("Reference")

    plt.subplot(1, 3, 3)
    plt.imshow(img_new, cmap='gray')
    plt.title("Matched")

    plt.show()


if __name__ == '__main__':
    main()
