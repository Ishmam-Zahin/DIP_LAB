import cv2
import numpy as np
import matplotlib.pyplot as plt
from zahinDIP import zahin_dip as zahin


def main():
    img = cv2.imread('/home/zahin/Desktop/DIP_LAB/images/img6.jpg', cv2.IMREAD_GRAYSCALE)
    kernels = {
        'Average': np.ones((3,3), dtype=np.float32)/9,
        'Sobel_X': np.array([[-1,0,1], [-2,0,2], [-1,0,1]], dtype=np.float32),
        'Sobel_Y': np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),
        'Prewitt_X': np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32),
        'Prewitt_Y': np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32),
        'Scharr_X': np.array([[-3,0,3],[-10,0,10],[-3,0,3]], dtype=np.float32),
        'Scharr_Y': np.array([[-3,-10,-3],[0,0,0],[3,10,3]], dtype=np.float32),
        'Laplace': np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32),
        'Custom1': np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32),
        'Custom2': np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32),
        'Custom3': np.array([[2,0,-2],[1,0,-1],[0,0,0]], dtype=np.float32),
        'Custom4': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    }

    plt.figure(figsize=(16, 9))
    total_row = 3
    total_col = 4

    for index, (name, kernel) in enumerate(kernels.items()):
        img_transformed = zahin.conv2D(img, kernel, 3)
        plt.subplot(total_row, total_col, index + 1)
        plt.imshow(img_transformed, cmap = 'gray')
        plt.title(name)
        plt.axis(False)
        


    plt.tight_layout()
    plt.savefig('./outputs/q6.png')





if __name__ == '__main__':
    main()
