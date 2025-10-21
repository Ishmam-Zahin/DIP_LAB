import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))

img1 = cv2.imread('../images/img3.jpg', cv2.IMREAD_GRAYSCALE)
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img1 = cv2.cvtColor(img1, cv2.COLOR_RG)
img2 = cv2.imread('../images/img4.jpg', cv2.IMREAD_GRAYSCALE)

if(img1.shape != img2.shape):
    print('shape do not match!')


img = img1 + img2

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title('image 1')

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.title('image 2')

plt.subplot(1, 3, 3)
plt.imshow(img, cmap='grey')
plt.title('merged image')


plt.tight_layout()
plt.show()
