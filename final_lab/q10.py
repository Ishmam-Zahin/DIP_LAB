import cv2
import numpy as np
import matplotlib.pyplot as plt
from zahinDIP import zahin_dip as zahin


def main():
    img = cv2.imread('/home/zahin/Desktop/DIP_LAB/final_lab/images/peppers.png', cv2.IMREAD_GRAYSCALE)
    kernels = {
        'rectangle': zahin.get_rect_se(),
        'ellipse': zahin.get_ellipse_se(),
        'cross': zahin.get_cross_se(),
        'diamond': zahin.get_diamond_se()
    }
    ops = {
        'erosion': [zahin.ero_or_dial, cv2.erode],
        'dialation': [zahin.ero_or_dial, cv2.dilate],
        'opening': [zahin.morph_open, cv2.morphologyEx],
        'closing': [zahin.morph_close, cv2.morphologyEx],
        'tophat': [zahin.morph_tohat, cv2.morphologyEx],
        'blackhat': [zahin.morph_blackhat, cv2.morphologyEx],
    }

    for (name, op) in ops.items():
        plt.figure(figsize=(12, 7))
        r = 3
        c = 4
        plt.subplot(r, c, 1)
        plt.imshow(img, cmap = 'gray')
        plt.axis(False)
        plt.title('original')
        plt.suptitle(name)
        ct = 1
        for index, (name_kernel, kernel) in enumerate(kernels.items(), 1):
            if name == 'erosion':
                img_transformed = op[0](img, kernel, type_morph = 'ero')
                img_transformed2 = op[1](img, kernel)
            elif name == 'dialation':
                img_transformed = op[0](img, kernel, type_morph = 'dial')
                img_transformed2 = op[1](img, kernel)
            else:
                img_transformed = op[0](img, kernel)
                if name == 'opening':
                    tmp = cv2.MORPH_OPEN
                elif name == 'closing':
                    tmp = cv2.MORPH_CLOSE
                elif name == 'tophat':
                    tmp = cv2.MORPH_TOPHAT
                else:
                    tmp = cv2.MORPH_BLACKHAT
                img_transformed2 = op[1](img, tmp, kernel)
            ct += 1
            plt.subplot(r, c, ct)
            plt.imshow(img_transformed, cmap = 'gray')
            plt.axis(False)
            plt.title(f'{name_kernel} (custom)')
            ct += 1
            plt.subplot(r, c, ct)
            plt.imshow(img_transformed2, cmap = 'gray')
            plt.axis(False)
            plt.title(f'{name_kernel} (builtin)')
        
        plt.tight_layout()
        plt.savefig(f'./outputs/q10_{name}.png')
        print(f'finished: {name}')



if __name__ == "__main__":
    main()