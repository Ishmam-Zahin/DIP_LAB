import cv2
import numpy as np
import matplotlib.pyplot as plt
from zahinDIP import zahin_dip as zahin


def main():
    img = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/peppers.png', cv2.IMREAD_GRAYSCALE)
    img_ahe = zahin.adaptive_hist_equaliz(img, block_size = 50)
    img_clahe_1 = zahin.adaptive_hist_clahe(img, block_size = 50, clip = 10)
    img_clahe_2 = zahin.adaptive_hist_clahe(img, block_size = 50, clip = 50)
    hist_ahe = zahin.calc_hist(img_ahe)
    hist = zahin.calc_hist(img)
    hist_calhe_1 = zahin.calc_hist(img_clahe_1)
    hist_calhe_2 = zahin.calc_hist(img_clahe_2)

    plt.figure(figsize=(12, 7))
    r = 2
    c = 4

    plt.subplot(r, c, 1)
    plt.imshow(img, cmap = 'gray')
    plt.axis(False)
    plt.title('original')
    plt.subplot(r, c, 2)
    plt.imshow(img_ahe, cmap = 'gray')
    plt.axis(False)
    plt.title('AHE')
    plt.subplot(r, c, 3)
    plt.imshow(img_clahe_1, cmap = 'gray')
    plt.axis(False)
    plt.title('CLAHE clipped = 10')
    plt.subplot(r, c, 4)
    plt.imshow(img_clahe_2, cmap = 'gray')
    plt.axis(False)
    plt.title('CLAHE clipped = 50')

    plt.subplot(r, c, 5)
    plt.bar(np.arange(len(hist)), hist)
    plt.title('original hist')
    plt.subplot(r, c, 6)
    plt.bar(np.arange(len(hist_ahe)), hist_ahe)
    plt.title('AHE hist')
    plt.subplot(r, c, 7)
    plt.bar(np.arange(len(hist_calhe_1)), hist_calhe_1)
    plt.title('CLAHE 10 hist')
    plt.subplot(r, c, 8)
    plt.bar(np.arange(len(hist_calhe_2)), hist_calhe_2)
    plt.title('CLAHE 50 hist')

    
    plt.tight_layout()
    plt.savefig('./outputs/q11.png')




if __name__ == '__main__':
    main()