import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    img = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/peppers.png', cv2.IMREAD_GRAYSCALE)
    img_transformed = cv2.Canny(img, 100, 200)

    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap = 'gray')
    plt.axis(False)
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(img_transformed, cmap = 'gray')
    plt.axis(False)
    plt.title('canny transform')


    plt.tight_layout()
    plt.savefig('./outputs/q8.png')





if __name__ == "__main__":
    main()