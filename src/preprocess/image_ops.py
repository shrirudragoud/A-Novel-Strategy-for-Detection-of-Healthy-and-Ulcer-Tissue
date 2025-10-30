from typing import Literal, Tuple

import numpy as np
from skimage import color, filters


ColorSpace = Literal["HSV", "YCbCr"]


def to_colorspace(rgb_uint8: np.ndarray, target: ColorSpace) -> np.ndarray:
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    if target == "HSV":
        hsv = color.rgb2hsv(rgb_float)
        return hsv.astype(np.float32)
    if target == "YCbCr":
        # skimage uses YCbCr via rgb2ycbcr in 0-255 range
        ycbcr = color.rgb2ycbcr(rgb_float)
        return ycbcr.astype(np.float32)
    raise ValueError(f"Unsupported colorspace: {target}")


def median_denoise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    # skimage.filters.median expects 2D or 3D; use disk kernel implicitly by size
    # For simplicity, apply channel-wise if 3D
    if image.ndim == 2:
        return filters.median(image, footprint=np.ones((kernel_size, kernel_size), dtype=bool))
    if image.ndim == 3:
        denoised = np.empty_like(image)
        for c in range(image.shape[2]):
            denoised[..., c] = filters.median(
                image[..., c], footprint=np.ones((kernel_size, kernel_size), dtype=bool)
            )
        return denoised
    raise ValueError("median_denoise expects 2D or 3D image array")


def extract_luminance(image_in_cs: np.ndarray, cs: ColorSpace) -> np.ndarray:
    if cs == "HSV":
        # V channel in [0,1]
        v = image_in_cs[..., 2]
        return v.astype(np.float32)
    if cs == "YCbCr":
        # Y channel in [0,255]
        y = image_in_cs[..., 0]
        # normalize to [0,1]
        return (y / 255.0).astype(np.float32)
    raise ValueError(f"Unsupported colorspace: {cs}")


def preprocess_luminance(
    rgb_uint8: np.ndarray,
    colorspace: ColorSpace = "HSV",
    kernel_size: int = 3,
) -> np.ndarray:
    cs_img = to_colorspace(rgb_uint8, colorspace)
    denoised = median_denoise(cs_img, kernel_size=kernel_size)
    lum = extract_luminance(denoised, colorspace)
    return lum  # float32 in [0,1]


