import cv2
import numpy as np
import matplotlib.pyplot as plt
from zahinDIP import zahin_dip as zahin


def main():
    img = cv2.imread('/home/zahin/Desktop/DIP_LAB/images/img6.jpg', cv2.IMREAD_GRAYSCALE)
    kernel_names = ['Smooth', 'Sobel X', 'Sobel Y', 'Prewitt X', 'Prewitt Y', 'Laplace']
    kernels = []
    kernel_smooth = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype = np.float32)
    kernel_smooth /= np.sum(kernel_smooth)
    kernels.append(kernel_smooth)
    kernel_sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    kernels.append(kernel_sobel_x)
    kernel_sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)
    kernels.append(kernel_sobel_y)
    kernel_prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)
    kernels.append(kernel_prewitt_x)
    kernel_prewitt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)
    kernels.append(kernel_prewitt_y)
    kernel_laplace = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    kernels.append(kernel_laplace)
    kernels = np.array(kernels, dtype = np.float32)

    plt.figure(figsize=(12, 7))
    total_row = 2
    total_col = 3

    for index, (name, kernel) in enumerate(zip(kernel_names, kernels)):
        img_transformed = cv2.filter2D(img, ddepth = -1, kernel = kernel)
        plt.subplot(total_row, total_col, index + 1)
        plt.imshow(img_transformed, cmap = 'gray')
        plt.title(name)
        plt.axis(False)
        


    plt.tight_layout()
    plt.savefig('./outputs/q5.png')





if __name__ == '__main__':
    main()