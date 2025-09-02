import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))

image = cv2.imread('../images/img.jpg', cv2.IMREAD_GRAYSCALE)


x = np.arange(0, 256)
y = np.zeros(256, dtype=float)

for row in image:
    for val in row:
        y[val] += 1


pdf = y / y.sum()
cdf = np.cumsum(pdf)
print(pdf)

plt.plot(x, y)
plt.plot(x, cdf)


plt.show()