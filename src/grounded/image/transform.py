from typing import Any, cast

import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike, NDArray

import optivect.core.math.matrix as matrix


def normalized_grayscale(image: NDArray[Any]) -> NDArray[np.float64]:
    """
    Normalize an image to a grayscale (float32) in range [0 1].

    Parameters:
        image: The input image.

    Returns:
        The normalized image.s
    """
    if len(image.shape) == 3 and image.dtype == np.uint8:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    elif len(image.shape) == 2 and image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    else:
        assert len(image.shape) == 2
        assert image.dtype == np.float64

    return image


def polar(
    image: NDArray[np.float64], polar_height: int, polar_width: int
) -> NDArray[np.float64]:
    """
    Perform image warp to log polar shape.

    Parameters:
        image: The image (single channel, float32 is expected.)
        polar_height: The expected height of the polar image.
        polar_width: The expected width of the polar image.

    Returns:
        The log polar image.
    """
    h, w = image.shape
    return cast(
        NDArray[np.float64],
        cv.warpPolar(
            image,
            dsize=(polar_width, polar_height),
            center=(w / 2.0, h / 2.0),
            maxRadius=min(w / 2.0, h / 2.0),
            flags=cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS,
        ),
    )


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
