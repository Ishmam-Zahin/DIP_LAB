import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_filter2D(img, kernel, mode='same'):

    img_h, img_w = img.shape
    k_h, k_w = kernel.shape
    
    kernel = np.flipud(np.fliplr(kernel))
    
    if mode == 'same':
        pad_h = k_h // 2
        pad_w = k_w // 2
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        out_h, out_w = img_h, img_w
    elif mode == 'valid':
        padded = img
        out_h, out_w = img_h - k_h + 1, img_w - k_w + 1
    else:
        raise ValueError("mode should be 'same' or 'valid'")
    
    output = np.zeros((out_h, out_w), dtype=np.float32)
    
    for i in range(out_h):
        for j in range(out_w):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
    
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

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

img_path = "../images/img6.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(16,9))
plt.subplot(3,5,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

for idx, (name, kernel) in enumerate(kernels.items(), start=2):
    filtered = custom_filter2D(img, kernel, mode='same')
    plt.subplot(3,5,idx)
    plt.imshow(filtered, cmap='gray')
    plt.title(name)
    plt.axis('off')

plt.tight_layout()
plt.savefig('../images/hw6_result.png')
# plt.show()
