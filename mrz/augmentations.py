import numpy as np
import cv2


def gauss_noise(image: np.ndarray, mean: float = 0, sigma: float = 20.0) -> np.ndarray:
    gauss = np.random.normal(mean, sigma, image.shape)
    return ((image + gauss).clip(0, 255)).astype(np.uint8)


def salt_pepper_noise(image: np.ndarray, s_vs_p: float = 0.5, amount: float = 0.01) -> np.ndarray:
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out.astype(np.uint8)


def speckle_noise(image: np.ndarray, force: float = 0.15) -> np.ndarray:
    gauss = np.random.randn(*image.shape) * force
    return ((image + image * gauss).clip(0, 255)).astype(np.uint8)


def gauss_blur(image: np.ndarray, size: int) -> np.ndarray:
    kernel_size = size + 1 if size % 2 == 0 else size
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), size / 3.0)
