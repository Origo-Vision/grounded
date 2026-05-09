from typing import Any, cast

import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike, NDArray

import grounded.math.matrix as matrix


def rotate(image: NDArray[np.uint8], theta: float) -> NDArray[np.uint8]:
    """
    Rotate an image by theta degrees.

    Parameters:
        image: Input image.
        theta: The rotation angle (in degrees).

    Returns:
        The warped image.
    """

    h, w = image.shape
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    M = matrix.translate((cx, cy)) @ matrix.rotate(theta) @ matrix.translate((-cx, -cy))
    return cast(NDArray[np.uint8], cv.warpAffine(image, M=M[:2], dsize=(w, h)))


def translate(image: NDArray[np.uint8], xy: ArrayLike) -> NDArray[np.uint8]:
    """
    Translate an image

    Parameters:
        image: Input image.
        xy: The image translations

    Returns:
        The warped image.
    """

    h, w = image.shape

    M = matrix.translate(xy)
    return cast(NDArray[np.uint8], cv.warpAffine(image, M=M[:2], dsize=(w, h)))
