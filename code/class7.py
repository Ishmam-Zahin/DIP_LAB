import cv2
import matplotlib.pyplot as plt

image = cv2.imread("../images/img6.jpg", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(image, threshold1=100, threshold2=200)

plt.figure(figsize=(16, 9))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.tight_layout()
plt.savefig('../images/hw8.png')
# plt.show()
