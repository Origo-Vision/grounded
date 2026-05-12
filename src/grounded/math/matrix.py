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
    """
    Create an affine matrix, for the given parameters.

    theta: The rotation around image center, in degrees.
        xt: X direction translation.
        yt: Y direction translation.
        cx: X coordinate to rotate around.
        cy: Y coordinate to rotate around.

    Returns:
        The affine matrix.
    """
    return (
        translate(xt=cx + xt, yt=cy + yt)
        @ rotate(theta=theta)
        @ translate(xt=-cx, yt=-cy)
    )


def decomp_affine(
    M: NDArray[np.float64], cx: float, cy: float
) -> tuple[NDArray[np.float64], float]:
    """
    Decompose an affine matrix into translation and rotation.

    Parameters:
        M: The affine matrix.
        cx: Center of image.
        cy: Center of image.

    Returns:
        Tuple with translation vector (xy), and rotation in degrees.
    """
    if not (M.shape == (3, 3) or M.shape == (2, 3)):
        raise ValueError("Affine matrix expected to be 3x3 or 2x3")

    theta = math.atan2(M[1, 0], M[0, 0])
    t = M[:2, 2]
    t -= (np.eye(2) - M[:2, :2]) @ (cx, cy)

    return t, math.degrees(theta)
