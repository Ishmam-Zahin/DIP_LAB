import cv2
import matplotlib.pyplot as plt
import numpy as np
from zahinDIP import zahin_dip as zahin



def main():
    img1 = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/peppers.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/pickacoo.png', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/Reference.png', cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/dark.png', cv2.IMREAD_GRAYSCALE)
    img5 = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/dark_normal.png', cv2.IMREAD_GRAYSCALE)
    
    imgs = {
        'imgae1': {
            'normal': img1,
            'low': zahin.toggle_contrast(img1, alpha = 0.5),
            'high': zahin.toggle_contrast(img1, alpha = 1.5)
        },
        'imgae2': {
            'normal': img2,
            'low': zahin.toggle_contrast(img2, alpha = 0.5),
            'high': zahin.toggle_contrast(img2, alpha = 1.5)
        },
        'imgae3': {
            'normal': img3,
            'low': zahin.toggle_contrast(img3, alpha = 0.5),
            'high': zahin.toggle_contrast(img3, alpha = 1.5)
        },
        'imgae4': {
            'normal': img4,
            'low': zahin.toggle_contrast(img4, alpha = 0.5),
            'high': zahin.toggle_contrast(img4, alpha = 1.5)
        },
        'imgae5': {
            'normal': img5,
            'low': zahin.toggle_contrast(img5, alpha = 0.5),
            'high': zahin.toggle_contrast(img5, alpha = 1.5)
        }
    }

    for (name, images) in imgs.items():
        plt.figure(figsize=(12, 7))
        r = 3
        c = 5
        for index, (contrast, img) in enumerate(images.items()):
            filter_low = zahin.ideal_lowpass_filter(img, 30)
            filter_high = zahin.ideal_highpass_filter(img, 80)
            filter_band = zahin.ideal_bandpass_filter(img, 30, 80)
            dft, dft_shift, magnitude = zahin.apply_fft(img)
            img_filtered_low = zahin.apply_filter(dft_shift, filter_low)
            img_filtered_high = zahin.apply_filter(dft_shift, filter_high)
            img_filtered_band = zahin.apply_filter(dft_shift, filter_band)
            offset = index * 5
            plt.subplot(r, c, (offset + 1))
            plt.imshow(img, cmap = 'gray')
            plt.axis(False)
            plt.title(f'{name} {contrast}')
            plt.subplot(r, c, (offset + 2))
            plt.imshow(magnitude, cmap = 'gray')
            plt.axis(False)
            plt.title(f'frequency domain')
            plt.subplot(r, c, (offset + 3))
            plt.imshow(img_filtered_low, cmap = 'gray')
            plt.axis(False)
            plt.title(f'filter low')
            plt.subplot(r, c, (offset + 4))
            plt.imshow(img_filtered_high, cmap = 'gray')
            plt.axis(False)
            plt.title(f'filter high')
            plt.subplot(r, c, (offset + 5))
            plt.imshow(img_filtered_band, cmap = 'gray')
            plt.axis(False)
            plt.title(f'filter band')

        plt.suptitle(f'{name}')
        plt.tight_layout()
        plt.savefig(f'./outputs/q13_{name}.png')
        print(f'finished {name}')
        



if __name__ == '__main__':
    main()