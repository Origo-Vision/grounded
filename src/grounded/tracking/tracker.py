from __future__ import annotations

import math
from typing import cast

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

import grounded.image.utils as image_utils
from grounded.tracking.frame import Frame


class Tracker:
    """
    Visual tracker class. Works on square images, where sides shall be
    power of two.
    """

    def __init__(self: Tracker, size: int, debug: bool = False) -> None:
        """
        Construct the tracker.

        Parameters:
            size: The side of the square image (shall be power of two).
            debug: Debug flag. When set, a lot more data is stored into frames.
        """
        if not math.log2(size).is_integer():
            raise ValueError("Size must be power of two")

        self._image_height = size
        self._image_width = size
        self._polar_height = 360
        self._polar_width = size // 2
        self._polar_center = (self._image_width / 2, self._image_height / 2)
        self._polar_max_radius = size / 2.0
        self._rmin = 5
        self._rmax = 5

        self._debug = debug
        self._image_window = image_utils.tukey_window(
            (self._image_height, self._image_width)
        )

    def new_frame(self: Tracker, image: NDArray[np.uint8]) -> Frame:
        """
        Create a new Frame object.

        Parameters:
            image: The input input (assumes reshaped, grayscale image).

        Returns:
            The Frame object.
        """
        if (
            image.shape != (self._image_height, self._image_width)
            and image.dtype != np.uint8
        ):
            raise ValueError("Expecting an 8-bit grayscale, of the specified shape")

        normalized_filtered_image = image_utils.normalized(image) * self._image_window
        spectrum = self._create_spectrum(normalized_filtered_image)
        polar_spectrum_fft = np.fft.rfft2(self._polar_warp(spectrum))

        frame = Frame(
            image=image,
            normalized_filtered_image=normalized_filtered_image,
            polar_spectrum_fft=polar_spectrum_fft,
            spectrum=spectrum if self._debug else None,
        )

        return frame

    def _create_spectrum(
        self: Tracker, normalized_filtered_image: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For now, not using rfft. Use full transform for simplicity.
        spectrum = np.fft.fftshift(np.fft.fft2(normalized_filtered_image))
        return np.log1p(np.abs(spectrum))

    def _polar_warp(
        self: Tracker, spectrum: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        polar = cv.warpPolar(
            spectrum,
            dsize=(self._polar_width, self._polar_height),
            center=self._polar_center,
            maxRadius=self._polar_max_radius,
            flags=cv.WARP_POLAR_LINEAR | cv.WARP_FILL_OUTLIERS | cv.INTER_CUBIC,
        )

        # Band pass filtering on the polar image.
        polar[:, : self._rmin] = 0.0
        polar[:, -self._rmax :] = 0.0

        return cast(NDArray[np.float64], polar)
