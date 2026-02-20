import cv2
import matplotlib.pyplot as plt
import numpy as np
from zahinDIP import zahin_dip as zahin



def main():
    img = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/pickacoo.png', cv2.IMREAD_GRAYSCALE)
    dft, dft_shift, magnitude = zahin.apply_fft(img)
    gauss_low = zahin.gaussian_lowpass_filter(img)
    img_gauss_low = zahin.apply_filter(dft_shift, gauss_low)
    gauss_high = zahin.gaussian_bandpass_filter(img)
    img_gauss_high = zahin.apply_filter(dft_shift, gauss_high)
    gauss_band = zahin.gaussian_bandpass_filter(img)
    img_gauss_band = zahin.apply_filter(dft_shift, gauss_band)

    butter_low = zahin.butterworth_lowpass_filter(img)
    butter_high = zahin.butterworth_highpass_filter(img)
    butter_band = zahin.butterworth_bandpass_filter(img)
    img_butter_low = zahin.apply_filter(dft_shift, butter_low)
    img_butter_high = zahin.apply_filter(dft_shift, butter_high)
    img_butter_band = zahin.apply_filter(dft_shift, butter_band)

    butter_c3 = zahin.butterworth_lowpass_filter(img, n = 3)
    butter_c4 = zahin.butterworth_lowpass_filter(img, n = 4)
    img_butter_c3 = zahin.apply_filter(dft_shift, butter_c3)
    img_butter_c4 = zahin.apply_filter(dft_shift, butter_c4)

    plt.figure(figsize=(12, 7))
    r = 3
    c = 3
    plt.subplot(r, c, 1)
    plt.imshow(img, cmap = 'gray')
    plt.axis(False)
    plt.title('original')

    plt.subplot(r, c, 2)
    plt.imshow(img_gauss_low, cmap = 'gray')
    plt.axis(False)
    plt.title('gauss low')

    plt.subplot(r, c, 3)
    plt.imshow(img_gauss_high, cmap = 'gray')
    plt.axis(False)
    plt.title('gauss high')

    plt.subplot(r, c, 4)
    plt.imshow(img_gauss_band, cmap = 'gray')
    plt.axis(False)
    plt.title('gauss band')

    plt.subplot(r, c, 5)
    plt.imshow(img_butter_low, cmap = 'gray')
    plt.axis(False)
    plt.title('butter low')

    plt.subplot(r, c, 6)
    plt.imshow(img_butter_high, cmap = 'gray')
    plt.axis(False)
    plt.title('butter high')

    plt.subplot(r, c, 7)
    plt.imshow(img_butter_band, cmap = 'gray')
    plt.axis(False)
    plt.title('butter band')

    plt.subplot(r, c, 8)
    plt.imshow(img_butter_c3, cmap = 'gray')
    plt.axis(False)
    plt.title('butter n = 3')

    plt.subplot(r, c, 9)
    plt.imshow(img_butter_c4, cmap = 'gray')
    plt.axis(False)
    plt.title('butter n = 4')

    plt.tight_layout()
    plt.savefig('./outputs/q15.png')



if __name__ == '__main__':
    main()