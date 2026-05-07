import pathlib
from typing import cast

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import tukey


def read_gray(path: pathlib.Path) -> NDArray[np.uint8] | None:
    """
    Read an image as grayscale.

    Parameters:
        path: The path to the image.

    Returns:
        The image.
    """
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is not None:
        return cast(NDArray[np.uint8], image)
    else:
        return None


def read_rgb(path: pathlib.Path) -> NDArray[np.uint8] | None:
    """
    Read an image as grayscale.

    Parameters:
        path: The path to the image.

    Returns:
        The image.
    """
    image = cv.imread(str(path), cv.IMREAD_COLOR_RGB)
    if image is not None:
        return cast(NDArray[np.uint8], image)
    else:
        return None


def normalized(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """
    Create a normalized (mean subtracted image) from the grayscale input.

    Parameters:
        image: Grayscale image.

    Returns:
        The normalized image.
    """
    f = image.astype(np.float64)
    return f - np.mean(f)


def resized(image: NDArray[np.uint8], shape: tuple[int, int]) -> NDArray[np.uint8]:
    """
    Resize an image if necessary.

    Parameters:
        image: The input image.
        shape: The requested shape.


    Returns:
        The image in the requested size.
    """
    if image.shape[:2] != shape:
        h, w = shape
        return cast(
            NDArray[np.uint8],
            cv.resize(image, dsize=(w, h), interpolation=cv.INTER_LINEAR),
        )
    else:
        return image


def tukey_window(shape: tuple[int, int], alpha: float = 0.25) -> NDArray[np.float64]:
    """
    Create a Tukey window.

    Parameters:
        shape: Tuple height, width.
        alpha: Tukey parameter.

    Returns:
        The Hamming window.
    """
    h, w = shape
    return np.outer(tukey(h, alpha=alpha), tukey(w, alpha=alpha))
