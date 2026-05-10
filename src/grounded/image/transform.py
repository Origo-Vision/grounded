from typing import Any, cast

import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike, NDArray

import grounded.math.matrix as matrix


def warp_affine(
    image: NDArray[np.uint8], theta: float, xt: float, yt: float
) -> NDArray[np.uint8]:
    """
    Perform an affine transform of the image given the parameters.

    Parameters:
        theta: The rotation around image center, in degrees.
        xt: X direction translation.
        yt: Y direction translation.

    Returns:
        The transformed image.
    """
    h, w = image.shape[:2]

    M = matrix.affine(theta=theta, xt=xt, yt=yt, cx=(w - 1) * 0.5, cy=(h - 1) * 0.5)
    return cast(NDArray[np.uint8], cv.warpAffine(image, M=M[:2], dsize=(w, h)))
