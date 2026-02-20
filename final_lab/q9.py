import cv2
import numpy as np
import matplotlib.pyplot as plt
from zahinDIP import zahin_dip as zahin


def main():
    img_source = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/dark.png', cv2.IMREAD_GRAYSCALE)
    img_target = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/Reference.png', cv2.IMREAD_GRAYSCALE)

    source = {
        'normal': img_source,
        # 'low': zahin.low_contrast(img_source, low=50, high=100),
        'low': zahin.toggle_contrast(img_source, alpha = 0.6),
        # 'high': zahin.high_contrast(img_source)
        'high': zahin.toggle_contrast(img_source, alpha = 1.6),
    }

    target = {
        'normal': img_target,
        # 'low': zahin.low_contrast(img_target, low = 50, high = 100),
        'low': zahin.toggle_contrast(img_target, alpha = 0.6),
        # 'high': zahin.high_contrast(img_target)
        'high': zahin.toggle_contrast(img_target, alpha = 1.6),
    }

    for index_s, (type_s, img_s) in enumerate(source.items()):
        for index_t, (type_t, img_t) in enumerate(target.items()):
            hist_s = zahin.calc_hist(img_s)
            hist_t = zahin.calc_hist(img_t)
            matched_v1 = zahin.hist_matchv1(img_s, img_t)
            hist_v1 = zahin.calc_hist(matched_v1)
            matched_v2 = zahin.hist_matchv2(img_s, img_t)
            hist_v2 = zahin.calc_hist(matched_v2)
            plt.figure(figsize=(12, 7))
            t = 2
            c = 4
            plt.subplot(t, c, 1)
            plt.imshow(img_s, cmap = 'gray', vmin = 0, vmax = 255)
            plt.axis(False)
            plt.title(f'source {type_s}')
            plt.subplot(t, c, 2)
            plt.imshow(img_t, cmap = 'gray', vmin = 0, vmax = 255)
            plt.axis(False)
            plt.title(f'target {type_t}')
            plt.subplot(t, c, 3)
            plt.imshow(matched_v1, cmap = 'gray', vmin = 0, vmax = 255)
            plt.axis(False)
            plt.title(f'matched v1')
            plt.subplot(t, c, 4)
            plt.imshow(matched_v2, cmap = 'gray', vmin = 0, vmax = 255)
            plt.axis(False)
            plt.title(f'matched v2')
            
            plt.subplot(t, c, 5)
            plt.bar(np.arange(len(hist_s)), hist_s)
            plt.title('hist source')
            plt.subplot(t, c, 6)
            plt.bar(np.arange(len(hist_t)), hist_t)
            plt.title('hist target')
            plt.subplot(t, c, 7)
            plt.bar(np.arange(len(hist_v1)), hist_v1)
            plt.title('hist v1')
            plt.subplot(t, c, 8)
            plt.bar(np.arange(len(hist_v2)), hist_v2)
            plt.title('hist v2')

            plt.tight_layout()
            plt.savefig(f'./outputs/q9_{type_s}_{type_t}.png')

            print(f'finished source: {type_s}, target: {type_t}')




if __name__ == "__main__":
    main()