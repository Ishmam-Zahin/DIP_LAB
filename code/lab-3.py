import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))

img = cv2.imread('../images/img5.jpg', cv2.IMREAD_GRAYSCALE)


x = np.arange(0, 256)
y = np.zeros(256, dtype=float)

for row in img:
    for val in row:
        y[val] += 1



plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('image')

plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title('histogram')
plt.grid(True)



plt.tight_layout()
plt.show()