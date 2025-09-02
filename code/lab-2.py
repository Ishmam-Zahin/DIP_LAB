import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))


img = cv2.imread('../images/img6.jpg', cv2.IMREAD_GRAYSCALE)
original = img.copy()
img[:100, :100] = 0


plt.subplot(1, 2, 1)
plt.imshow(original, cmap='gray')
plt.title('original image')

plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.title('modified image')


plt.tight_layout()
plt.show()