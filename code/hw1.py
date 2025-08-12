import matplotlib.pyplot as plt
import cv2
import numpy as np

img = np.ones((300, 300, 3), dtype=np.uint8)
shades = [0, 50, 100, 150, 255]
channels = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 1], [0, 2, 2], [1, 2, 2], [0, 1, 2]]
img_sets = []

for channel in channels:
    for shade in shades:
        shade_cannel = np.array([0, 0, 0])
        shade_cannel[channel[0]] = shade
        shade_cannel[channel[1]] = shade
        shade_cannel[channel[2]] = shade
        shade_img = img * shade_cannel
        img_sets.append(shade_img)

col = 5
row = int((len(img_sets) / col))

plt.figure(figsize=(15, 7))

for i, img in enumerate(img_sets):
    plt.subplot(row, col, i+1)
    plt.imshow(img)


plt.tight_layout()
plt.show()
# plt.savefig('shade.png')