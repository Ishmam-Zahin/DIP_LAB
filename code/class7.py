import matplotlib.pyplot as plt
import cv2

img_path = '../images/img.jpg'

img = cv2.imread(img_path, 0)
print(img.shape)

processed_img = 255 - img
save_img_path = '../images/processed.jpg'
cv2.imwrite(save_img_path, processed_img)

