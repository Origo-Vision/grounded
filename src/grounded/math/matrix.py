import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def translate(xt: float, yt: float) -> NDArray[np.float64]:
    """
    Create a translate matrix.

    Parameters:
        xy: The vector of translation.

    Returns:
        The translate matrix.
    """
    return np.array([[1.0, 0.0, xt], [0.0, 1.0, yt], [0.0, 0.0, 1.0]])


def rotate(theta: float) -> NDArray[np.float64]:
    """
    Create a rotation matrix.

    Parameters:
        theta: The rotation angle in degrees.

    Returns:
        The rotation matrix.
    """
    theta = math.radians(theta)
    c = math.cos(theta)
    s = math.sin(theta)

    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def affine(
    theta: float, xt: float, yt: float, cx: float, cy: float
) -> NDArray[np.float64]:
    return (
        translate(xt=cx + xt, yt=cy + yt)
        @ rotate(theta=theta)
        @ translate(xt=-cx, yt=-cy)
    )


def rotate_translate(
    theta: float, xy: ArrayLike, size: ArrayLike
) -> NDArray[np.float64]:
    center = np.array(size) / 2.0

    return translate(center + np.array(xy)) @ rotate(theta) @ translate(-center)


def translate_rotate(xy: ArrayLike, theta: float) -> NDArray[np.float64]:
    """
    Create an affine matrix with translation and rotation (rotation is applied first).

    Parameters:
        xy: The vector of translation.
        theta: The rotation angle in degrees.

    Returns:
        The rotation matrix.
    """
    return translate(xy) @ rotate(theta)


def decomp_affine(M: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """
    Decompose an affine matrix into translation and rotation.

    Parameters:
        M: The affine matrix.

    Returns:
        Tuple with translation (xy), and rotation in degrees.
    """
    if not (M.shape == (3, 3) or M.shape == (2, 3)):
        raise ValueError("Affine matrix expected to be 3x3 or 2x3")

    theta = -math.atan2(M[0, 1], M[0, 0])

    return M[:2, 2], math.degrees(theta)
