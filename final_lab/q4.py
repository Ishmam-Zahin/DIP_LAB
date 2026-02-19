from zahinDIP import zahin_dip as zahin
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    img = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/peppers.png', cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize = (12, 9))
    total_row = 2
    total_col = 4

    plt.subplot(total_row, total_col, 1)
    plt.imshow(img, cmap = 'gray')
    plt.title('Original Image')
    plt.axis(False)

    img_threshold_1 = zahin.step_func(img, 80)
    plt.subplot(total_row, total_col, 2)
    plt.imshow(img_threshold_1, cmap = 'gray')
    plt.title('Threshold 80')
    plt.axis(False)
    img_threshold_2 = zahin.step_func(img, 150)
    plt.subplot(total_row, total_col, 3)
    plt.imshow(img_threshold_2, cmap = 'gray')
    plt.title('Threshold 150')
    plt.axis(False)
    img_threshold_3 = zahin.step_func(img, 220)
    plt.subplot(total_row, total_col, 4)
    plt.imshow(img_threshold_3, cmap = 'gray')
    plt.title('Threshold 220')
    plt.axis(False)

    img_log = zahin.log_transform(img)
    counts = zahin.calc_hist(img_log)
    plt.subplot(total_row, total_col, 5)
    plt.imshow(img_log, cmap = 'gray')
    plt.title('Log Transform')
    plt.axis(False)
    plt.subplot(total_row, total_col, 6)
    plt.bar(np.arange(len(counts)), counts)
    plt.title('Log Transform Hist')
    plt.axis(True)

    img_gamma = zahin.gamma_correction(img, 3.5)
    counts = zahin.calc_hist(img_gamma)
    plt.subplot(total_row, total_col, 7)
    plt.imshow(img_gamma, cmap = 'gray')
    plt.title('Gamma Correction')
    plt.axis(False)
    plt.subplot(total_row, total_col, 8)
    plt.bar(np.arange(len(counts)), counts)
    plt.title('Gamma Correction Hist')
    plt.axis(True)


    plt.tight_layout()
    plt.savefig('./outputs/q4.png')

    




if __name__ == "__main__":
    main()



