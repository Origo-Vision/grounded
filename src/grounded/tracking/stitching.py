import math
from typing import cast

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

import grounded.math.matrix as matrix
from grounded.tracking.frame import Frame


def stitch_frames(frames: list[Frame]) -> NDArray[np.uint8] | None:
    """
    Stitch together a list of frames into a map.

    Parameters:
        frames: A list if frames, must be >= two frames, and the first is assumed
        to be the reference.

    Returns:
        The stitched map, or None if failed.
    """
    # There must be at least two frames to perform the stitching.
    if len(frames) < 2:
        return None

    # All other frames will be related to the first frame. Build a list
    # of homographies relating each frame to the first.
    H0 = frames[0]._H  # Assume first frame's H is I.
    Hs = [H0]

    for frame in frames[1:]:
        # First frame => frame.
        H = np.linalg.inv(H0) @ frame._H

        # frame => first frame.
        Hs.append(cast(NDArray[np.float64], np.linalg.inv(H)))

    # Map the frame's image corners into the first frames image space, to get the
    # dimensions for the canvas.
    w, h = frames[0]._original.shape[::-1]
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [w - 1.0, 0.0, 1.0],
            [0.0, h - 1.0, 1.0],
            [w - 1.0, h - 1.0, 1.0],
        ]
    ).T

    min = np.array([np.inf, np.inf])
    max = np.array([-np.inf, -np.inf])

    for H in Hs:
        mapped_corners = H @ corners
        mapped_corners /= mapped_corners[-1]
        mapped_corners = mapped_corners[:2].T

        min = np.minimum(np.min(mapped_corners, axis=0), min)
        max = np.maximum(np.max(mapped_corners, axis=0), max)

    xmin, ymin = min
    xmax, ymax = max
    canvas_size = (math.ceil(ymax - ymin), math.ceil(xmax - xmin))

    # Warp all images into the canvas.
    T = matrix.translate(xt=-xmin, yt=-ymin)

    mask = np.ones((h, w), dtype=np.float32)
    accum = np.zeros(canvas_size, dtype=np.float32)
    weights = np.zeros(canvas_size, dtype=np.float32)

    for H, frame in zip(Hs, frames):
        warped_image = cv.warpPerspective(
            frame._original, M=T @ H, dsize=canvas_size[::-1]
        )
        warped_mask = cv.warpPerspective(mask, M=T @ H, dsize=canvas_size[::-1])
        accum += warped_image * warped_mask
        weights += warped_mask

    # Average blending in overlaps.
    result = accum / np.maximum(weights, 1e-07)

    return result.astype(np.uint8)
