import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../images/img5.jpg', cv2.IMREAD_GRAYSCALE)

x = np.arange(0, 256)
hist = np.zeros(256, dtype=float)

for row in image:
    for val in row:
        hist[val] += 1

pdf = hist / hist.sum()
cdf = np.cumsum(pdf)
cdf_normalized = np.round(cdf * 255).astype(np.uint8)

equalized_image = cdf_normalized[image]

equalized_hist = np.zeros(256, dtype=float)
for row in equalized_image:
    for val in row:
        equalized_hist[val] += 1
equalized_pdf = equalized_hist / equalized_hist.sum()
equalized_cdf = np.cumsum(equalized_pdf)

opencv_equalized = cv2.equalizeHist(image)

opencv_hist = np.zeros(256, dtype=float)
for row in opencv_equalized:
    for val in row:
        opencv_hist[val] += 1
opencv_pdf = opencv_hist / opencv_hist.sum()
opencv_cdf = np.cumsum(opencv_pdf)

plt.figure(figsize=(16, 9))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Custom Equalized Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(opencv_equalized, cmap='gray')
plt.title('OpenCV Equalized Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.plot(x, hist, label='Histogram')
plt.plot(x, cdf * hist.sum(), label='CDF')
plt.title('Original Histogram & CDF')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(x, equalized_hist, label='Histogram')
plt.plot(x, equalized_cdf * equalized_hist.sum(), label='CDF')
plt.title('Custom Equalized Histogram & CDF')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(x, opencv_hist, label='Histogram')
plt.plot(x, opencv_cdf * opencv_hist.sum(), label='CDF')
plt.title('OpenCV Equalized Histogram & CDF')
plt.legend()

plt.tight_layout()
plt.savefig('../images/hw7_result.png')
# plt.show()
