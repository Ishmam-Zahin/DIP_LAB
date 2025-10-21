import matplotlib.pyplot as plt
import cv2
import math
import numpy as np

plt.figure(figsize=(16, 9))

kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32) / 9.0

kernel2 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)

kernel3 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]], dtype=np.float32) / 9.0

kernel4 = np.array([[-1, -1, 1],
                    [1, 0, 1],
                    [2, 1, -2]], dtype=np.float32) / 9.0


img = cv2.imread('../images/img5.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.cvtColor(img, cv2.GRAY)
img_filter1 = cv2.filter2D(img, -1, kernel)
img_filter2 = cv2.filter2D(img, -1, kernel2)
img_filter3 = cv2.filter2D(img, -1, kernel3)

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_filter1, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_filter2, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_filter3, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
