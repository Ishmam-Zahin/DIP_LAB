from typing import Tuple

import numpy as np
import numpy.typing as npt
import cv2

import numpy as np
import numpy.typing as npt

def toggle_contrast(
    img: npt.NDArray[np.uint8],
    alpha: float = 1.0,
    beta: float = 0.0,
    max_value: int = 255
) -> npt.NDArray[np.uint8]:
    out = img.astype(np.float32)
    out = alpha * out + beta
    out = np.abs(out)
    out = np.clip(np.round(out), 0, max_value)
    return out.astype(np.uint8)


def high_contrast(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    return cv2.equalizeHist(img).astype(np.uint8)

def low_contrast(img: npt.NDArray[np.uint8], low: int = 50, high: int = 200) -> npt.NDArray[np.uint8]:
    return cv2.normalize(img, None, low, high, cv2.NORM_MINMAX)

def calc_hist(img: npt.NDArray[np.uint8], max_value: int = 255)->npt.NDArray[np.int32]:
    counts = np.zeros((max_value + 1), dtype = np.int32)

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


def _calc_pdf(counts: npt.NDArray)->np.ndarray[np.float32]:
    total = float(np.sum(counts))
    pdf = counts / total
    return pdf

def _calc_cdf(pdf: npt.NDArray[np.float32])->npt.NDArray[np.float32]:
    cdf = np.cumsum(pdf)
    return cdf

def hist_equaliz(img: npt.NDArray[np.uint8], max_value: int = 255, output_image: bool = True)->npt.NDArray[np.uint8]:
    counts = calc_hist(img)
    pdf = _calc_pdf(counts)
    cdf = _calc_cdf(pdf)
    nz = np.nonzero(cdf)[0]
    cdf_min = cdf[nz[0]]
    if (1 - cdf_min) <= 0:
        raise ValueError('less than zero')
    equalized = (((cdf - cdf_min) / (1 - cdf_min)) * float(max_value))
    equalized = np.clip(np.round(equalized), 0, max_value)
    equalized = equalized.astype(np.uint8)
    if not output_image:
        return equalized
    img_transformed = equalized[img]
    return img_transformed

def hist_matchv1(img: npt.NDArray[np.uint8], img_target: npt.NDArray[np.uint8], max_value: int = 255)->npt.NDArray[np.uint8]:
    hist_source = calc_hist(img)
    hist_target = calc_hist(img_target)
    cdf_source = _calc_cdf(_calc_pdf(hist_source))
    cdf_target = _calc_cdf(_calc_pdf(hist_target))
    matched = np.zeros((len(cdf_source)), dtype = np.uint8)
    for index in range(len(matched)):
        differences = np.abs(cdf_target - cdf_source[index])
        matched[index] = np.argmin(differences).astype(np.uint8)
    img_transformed = matched[img]
    return img_transformed


def hist_matchv2(img: npt.NDArray[np.uint8], img_target: npt.NDArray[np.uint8], max_value: int = 255)->npt.NDArray[np.uint8]:
    equilized_source = hist_equaliz(img, output_image = False).astype(np.int32)
    equilized_target = hist_equaliz(img_target, output_image = False).astype(np.int32)
    matched = np.zeros((len(equilized_source)), dtype = np.uint8)
    for index in range(len(matched)):
        differences = np.abs(equilized_target - equilized_source[index])
        matched[index] = np.argmin(differences).astype(np.uint8)
    img_transformed = matched[img]
    return img_transformed




def adaptive_hist_equaliz(img: npt.NDArray[np.uint8], max_value: int = 255, block_size: int = 8)->npt.NDArray[np.uint8]:
    row = img.shape[0]
    col = img.shape[1]
    i = 0
    img_transformed = np.zeros_like(img, dtype = np.uint8)
    while i < row:
        j = 0
        while j < col:
            i_m = min((i + block_size), row)
            j_m = min((j + block_size), col)
            img_block = img[i:i_m, j:j_m]
            img_block_transformed = hist_equaliz(img_block)
            img_transformed[i:i_m, j:j_m] = img_block_transformed
            j += block_size
        i += block_size
    return img_transformed

def hist_clahe(img: npt.NDArray[np.uint8], max_value: int = 255, clip: float = 2)->npt.NDArray[np.uint8]:
    counts = calc_hist(img)
    counts_cliped = np.array(counts, dtype = np.float32)
    while True:
        counts_cliped = np.minimum(counts_cliped, clip)
        excessives = np.maximum((counts_cliped - clip), 0)
        tmp = excessives.sum()
        if tmp < (max_value + 1):
            break
        exc_amount = tmp / float(max_value + 1)
        counts_cliped = counts_cliped + exc_amount
    pdf = _calc_pdf(counts_cliped)
    cdf = _calc_cdf(pdf)
    nz = np.nonzero(cdf)[0]
    cdf_min = cdf[nz[0]]
    if (1 - cdf_min) <= 0:
        raise ValueError('less than zero')
    equalized = (((cdf - cdf_min) / (1 - cdf_min)) * float(max_value))
    equalized = np.clip(np.round(equalized), 0, max_value)
    equalized = equalized.astype(np.uint8)
    img_transformed = equalized[img]
    return img_transformed


def adaptive_hist_clahe(img: npt.NDArray[np.uint8], max_value: int = 255, block_size: int = 8, clip: float = 2)->npt.NDArray[np.uint8]:
    row = img.shape[0]
    col = img.shape[1]
    i = 0
    img_transformed = np.zeros_like(img, dtype = np.uint8)
    while i < row:
        j = 0
        while j < col:
            i_m = min((i + block_size), row)
            j_m = min((j + block_size), col)
            img_block = img[i:i_m, j:j_m]
            img_block_transformed = hist_clahe(img_block, clip=clip)
            img_transformed[i:i_m, j:j_m] = img_block_transformed
            j += block_size
        i += block_size
    return img_transformed



def get_rect_se(kernel_size: int = 5)->npt.NDArray[np.uint8]:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)).astype(np.uint8)


def get_ellipse_se(kernel_size: int = 5)->npt.NDArray[np.uint8]:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)).astype(np.uint8)

def get_cross_se(kernel_size: int = 5)->npt.NDArray[np.uint8]:
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size)).astype(np.uint8)

def get_diamond_se(kernel_size: int = 5)->npt.NDArray[np.uint8]:
    cx = kernel_size // 2
    cy = kernel_size // 2
    radius = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    for y in range(kernel_size):
        for x in range(kernel_size):
            if abs(x - cx) + abs(y - cy) <= radius:
                kernel[y][x] = 1
    return kernel

def ero_or_dial(img: npt.NDArray[np.uint8], kernel: npt.NDArray[np.uint8], max_value: int = 255, type_morph: str = 'ero')->npt.NDArray[np.uint8]:
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    img_padded = np.pad(img, pad_width = pad_size, mode = 'constant', constant_values = max_value if type_morph == 'ero' else 0)
    img_transformed = np.zeros_like(img, dtype = np.uint8)

    for i in range(img_transformed.shape[0]):
        for j in range(img_transformed.shape[1]):
            patch = img_padded[i:(i+kernel_size), j:(j+kernel_size)]
            img_transformed[i][j] = np.min(patch[kernel == 1]) if type_morph == 'ero' else np.max(patch[kernel == 1])
    return img_transformed

def morph_open(img: npt.NDArray[np.uint8], kernel: npt.NDArray[np.uint8])->npt.NDArray[np.uint8]:
    return ero_or_dial(ero_or_dial(img, kernel, type_morph='ero'), kernel, type_morph = 'dial')

def morph_close(img: npt.NDArray[np.uint8], kernel: npt.NDArray[np.uint8])->npt.NDArray[np.uint8]:
    return ero_or_dial(ero_or_dial(img, kernel, type_morph='dial'), kernel, type_morph = 'ero')

def morph_tohat(img: npt.NDArray[np.uint8], kernel: npt.NDArray[np.uint8])->npt.NDArray[np.uint8]:
    return cv2.subtract(img, morph_open(img, kernel))

def morph_blackhat(img: npt.NDArray[np.uint8], kernel: npt.NDArray[np.uint8])->npt.NDArray[np.uint8]:
    return cv2.subtract(morph_close(img, kernel), img)



def ideal_lowpass_filter(img: npt.NDArray[np.uint8], radius: int)->npt.NDArray[np.uint8]:
    row = img.shape[0]
    col = img.shape[1]
    crow = row // 2
    ccol = col // 2
    x = np.arange(row)
    y = np.arange(col)
    y, x = np.meshgrid(y, x)
    distances = np.sqrt((x - crow)**2 + (y - ccol)**2)
    filter = (distances <= radius).astype(np.uint8)
    return filter

def ideal_highpass_filter(img: npt.NDArray[np.uint8], radius: int)->npt.NDArray[np.uint8]:
    filter = (1 - ideal_lowpass_filter(img, radius)).astype(np.uint8)
    return filter

def ideal_bandpass_filter(img: npt.NDArray[np.uint8], radius_low: int, radius_high: int)->npt.NDArray[np.uint8]:
    row = img.shape[0]
    col = img.shape[1]
    crow = row // 2
    ccol = col // 2
    x = np.arange(row)
    y = np.arange(col)
    y, x = np.meshgrid(y, x)
    distances = np.sqrt((x - crow)**2 + (y - ccol)**2)
    filter = ((distances >= radius_low) & (distances <= radius_high)).astype(np.uint8)
    return filter


def apply_fft(img: npt.NDArray[np.uint8])->Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    img = img.astype(np.float32)
    dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return dft, dft_shift, magnitude

def apply_filter(dft_shift: npt.NDArray, filter: npt.NDArray)->npt.NDArray:
    filter = filter[:, :, np.newaxis]
    filter_shift = dft_shift * filter
    filter_ishift = np.fft.ifftshift(filter_shift)
    img_back = cv2.idft(filter_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    return img_back


def butterworth_lowpass_filter(img: npt.NDArray[np.uint8], d0: int = 40, n: int = 2) -> npt.NDArray[np.float32]:
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u - crow, v - ccol, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    return 1 / (1 + (D / d0) ** (2 * n))

def butterworth_highpass_filter(img: npt.NDArray[np.uint8], d0: int = 40, n: int = 2) -> npt.NDArray[np.float32]:
    return 1 - butterworth_lowpass_filter(img, d0, n)

def gaussian_lowpass_filter(img: npt.NDArray[np.uint8], d0: int = 40) -> npt.NDArray[np.float32]:
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u - crow, v - ccol, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    return np.exp(-(D**2) / (2 * (d0**2)))

def gaussian_highpass_filter(img: npt.NDArray[np.uint8], d0: int = 40) -> npt.NDArray[np.float32]:
    return 1 - gaussian_lowpass_filter(img, d0)


def butterworth_bandpass_filter(img: npt.NDArray[np.uint8], d_low: int = 30, d_high: int = 80, n: int = 2) -> npt.NDArray[np.float32]:
    return butterworth_lowpass_filter(img, d_high, n) - butterworth_lowpass_filter(img, d_low, n)

def gaussian_bandpass_filter(img: npt.NDArray[np.uint8], d_low: int = 30, d_high: int = 80) -> npt.NDArray[np.float32]:
    return gaussian_lowpass_filter(img, d_high) - gaussian_lowpass_filter(img, d_low)
