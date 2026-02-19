import numpy as np
import numpy.typing as npt

def calc_hist(img: npt.NDArray[np.uint8], max_value: int = 255)->npt.NDArray[np.int32]:
    counts = np.zeros((max_value), dtype = np.int32)

    for row in img:
        for pixel in row:
            counts[pixel] += 1
    
    return counts

def log_transform(img: npt.NDArray[np.uint8], max_value: int = 255, c = None)->npt.NDArray[np.uint8]:
    c_given = False if c is None else True
    img = img.astype(np.float32)
    if not c_given:
        c = max_value / np.log(1 + max_value)
    img_transformed = (c * np.log(1 + img))
    if c_given:
        max_transformed = np.max(img_transformed)
        min_transformed = np.min(img_transformed)
        img_transformed = (img_transformed - min_transformed) / (max_transformed - min_transformed)
        img_transformed = img_transformed * float(max_value)
    img_transformed = np.clip(np.round(img_transformed), 0, max_value)
    img_transformed = img_transformed.astype(np.uint8)
    return img_transformed


def gamma_correction(img: npt.NDArray[np.uint8], gamma: float, max_value: int = 255, c: int = 1)->npt.NDArray[np.uint8]:
    img_transformed = img / float(max_value)
    img_transformed = np.pow(img_transformed, gamma)
    img_transformed *= c
    img_transformed *= float(max_value)
    img_transformed = np.clip(np.round(img_transformed), 0, max_value)
    img_transformed = img_transformed.astype(np.uint8)
    return img_transformed

def step_func(img: npt.NDArray[np.uint8], threshold: int, max_value: int = 255):
    img_transformed = np.where((img > threshold), max_value, 0)
    img_transformed = img_transformed.astype(np.uint8)
    return img_transformed


def conv2D(img: npt.NDArray[np.uint8], kernel: npt.NDArray[np.float32], kernel_size: int, max_value: int = 255, input_normalize: bool = True, same: bool = True, output_normalize: bool = True)->npt.NDArray:
    img = img.astype(np.float32)
    if input_normalize:
        img = img / float(max_value)
    row = img.shape[0]
    col = img.shape[1]
    row_transformed = row - kernel_size + 1
    col_transformed = col - kernel_size + 1
    pad_width = (kernel_size - 1) // 2
    img_padded = np.array(img)
    if same:
        row_transformed = row
        col_transformed = col
        img_padded = np.pad(img_padded, pad_width, mode = 'constant', constant_values = 0)
    img_transformed = np.zeros((row_transformed, col_transformed), dtype = np.float32)
    for i in range(row_transformed):
        for j in range(col_transformed):
            patch = img_padded[i:(i + kernel_size), j:(j + kernel_size)]
            img_transformed[i][j] = np.sum((patch * kernel))
    if output_normalize:
        # max_transformed = np.max(img_transformed)
        # min_transformed = np.min(img_transformed)
        # img_transformed = ((img_transformed - min_transformed) / (max_transformed - min_transformed))
        img_transformed *= float(max_value)
        img_transformed = np.clip(np.round(img_transformed), 0, max_value)
        img_transformed = img_transformed.astype(np.uint8)
    return img_transformed