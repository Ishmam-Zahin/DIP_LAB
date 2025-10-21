import matplotlib.pyplot as plt
import cv2
import math
import numpy as np

plt.figure(figsize=(15, 7))

c = 1
g = [.1, .3, .7, 1, 3, 5, 10]

def func1(r):
    return (c * (r ** g))

def func2(r):
    return (c * math.log2(1 + r))


img_path = '/home/zahin/DIP/images/img.jpg'
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(2, 4, 1)
plt.imshow(img_rgb, cmap='grey')
plt.title(label='original')

img_normalized = img_rgb / 255.0
transformed_imgs = []
# transformed_clipped = np.clip(transformed * 255, 0, 255).astype(np.uint8)

for x in g:
    transformed_imgs.append((c * (img_normalized ** x)))


for i, transformed_img in enumerate(transformed_imgs, 2):
    plt.subplot(2, 4, i)
    plt.imshow(transformed_img, cmap='grey')
    plt.title(label=f'gamma: {g[i - 2]}')


plt.tight_layout()
plt.show()