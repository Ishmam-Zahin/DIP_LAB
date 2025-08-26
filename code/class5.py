import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 9))

image = cv2.imread('../images/img5.jpg', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32) / 9.0

mean = 0
std = 10
print(std)
gaussianNoise = np.random.normal(mean, std, image.shape).astype(np.float32)
gaussianNoise = np.reshape(gaussianNoise, image.shape)
noisyImage = image.astype(np.float32) + gaussianNoise
noisyImage = np.clip(noisyImage, 0, 255).astype(np.uint8)

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='grey')
plt.title('original')

plt.subplot(2, 2, 2)
plt.imshow(noisyImage, cmap='grey')
plt.title('noisy')

originalFiltered = cv2.filter2D(image, -1, kernel)
noisyFiltered = cv2.filter2D(noisyImage, -1, kernel)

plt.subplot(2, 2, 3)
plt.imshow(originalFiltered, cmap='grey')
plt.title('original filtered')

plt.subplot(2, 2, 4)
plt.imshow(noisyFiltered, cmap='grey')
plt.title('noisy filtered')

plt.show()


