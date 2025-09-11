import cv2
import numpy as np
import matplotlib.pyplot as plt

def cal_hist_equalize(img):
    clahe_hist = np.zeros(256, dtype=float)
    for row in img:
        for val in row:
            clahe_hist[val] += 1
    clahe_pdf = clahe_hist / clahe_hist.sum()
    clahe_cdf = np.cumsum(clahe_pdf)

    return (clahe_hist, clahe_cdf)


image1 = cv2.imread('../images/img.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('../images/img5.jpg', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('../images/img6.jpg', cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img1 = clahe.apply(image1)
clahe_img2 = clahe.apply(image2)
clahe_img3 = clahe.apply(image3)

hist1, cdf1 = cal_hist_equalize(image1)
hist2, cdf2 = cal_hist_equalize(image2)
hist3, cdf3 = cal_hist_equalize(image3)

x = np.arange(0, 256)

plt.figure(figsize=(16, 9))

plt.subplot(3, 3, 1)
plt.imshow(image1, cmap='gray')

plt.subplot(3, 3, 2)
plt.imshow(image2, cmap='gray')

plt.subplot(3, 3, 3)
plt.imshow(image3, cmap='gray')

plt.subplot(3, 3, 4)
plt.imshow(clahe_img1, cmap='gray')

plt.subplot(3, 3, 5)
plt.imshow(clahe_img2, cmap='gray')

plt.subplot(3, 3, 6)
plt.imshow(clahe_img3, cmap='gray')

plt.subplot(3, 3, 7)
plt.plot(x, hist1, label='Histogram')
plt.plot(x, cdf1 * hist1.sum(), label='CDF')
plt.legend()

plt.subplot(3, 3, 8)
plt.plot(x, hist2, label='Histogram')
plt.plot(x, cdf2 * hist2.sum(), label='CDF')
plt.legend()

plt.subplot(3, 3, 9)
plt.plot(x, hist3, label='Histogram')
plt.plot(x, cdf3 * hist3.sum(), label='CDF')
plt.legend()


plt.tight_layout()
# plt.savefig('../images/hw7_result.png')
plt.show()
