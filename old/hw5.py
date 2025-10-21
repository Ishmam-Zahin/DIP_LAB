import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../images/img6.jpg", cv2.IMREAD_GRAYSCALE)

avg_kernel = np.ones((3, 3), np.float32) / 9
smoothImg = cv2.filter2D(img, -1, avg_kernel)
sobelX = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobelY = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]], dtype=np.float32)

sobelXImg = cv2.filter2D(img, -1, sobelX)
sobelYImg = cv2.filter2D(img, -1, sobelY)

prewittX = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

prewittY = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]], dtype=np.float32)

prewittXImg = cv2.filter2D(img, -1, prewittX)
prewittYimg = cv2.filter2D(img, -1, prewittY)

lap = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=np.float32)

lapImg = cv2.filter2D(img, -1, lap)

titles = [
    "Original", "Average (3x3)",
    "Sobel X", "Sobel Y",
    "Prewitt X", "Prewitt Y",
    "Laplacian"
]
images = [
    img, smoothImg,
    sobelXImg, sobelYImg,
    prewittXImg, prewittYimg,
    lapImg
]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
# plt.show()
plt.savefig('kernal.png')
