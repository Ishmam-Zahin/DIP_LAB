import cv2
import numpy as np
import matplotlib.pyplot as plt
from zahinDIP import zahin_dip as zahin


def main():
    img = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/dark.png', cv2.IMREAD_GRAYSCALE)
    hist0 = zahin.calc_hist(img)
    img_equilized = zahin.hist_equaliz(img)
    hist1 = zahin.calc_hist(img_equilized)
    img_equilized2 = cv2.equalizeHist(img)
    hist2 = zahin.calc_hist(img_equilized2)
    plt.figure(figsize=(12, 9))
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap = 'gray')
    plt.title('original')
    plt.subplot(2, 3, 2)
    plt.imshow(img_equilized, cmap = 'gray')
    plt.title('equilized custom')
    plt.subplot(2, 3, 3)
    plt.imshow(img_equilized2, cmap = 'gray')
    plt.title('equilized builtin')
    plt.subplot(2, 3, 4)
    plt.bar(np.arange(len(hist0)), hist0)
    plt.title('original hist')
    plt.subplot(2, 3, 5)
    plt.bar(np.arange(len(hist1)), hist1)
    plt.title('equilized hist custom')
    plt.subplot(2, 3, 6)
    plt.bar(np.arange(len(hist2)), hist2)
    plt.title('equilized hist builtin')

    plt.tight_layout()
    plt.savefig('./outputs/q7.png')




if __name__ == '__main__':
    main()