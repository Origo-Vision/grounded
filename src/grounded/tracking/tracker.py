from __future__ import annotations

import math
from typing import cast

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

import grounded.image.heatmap as heatmap
import grounded.image.transform as transform
import grounded.image.utils as image_utils
import grounded.math.utils as math_utils
from grounded.tracking.frame import Frame


class Tracker:
    """
    Visual tracker class. Works on square images, where sides shall be
    power of two. Assuming to work in a homographic scene.
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
        self._polar_window = image_utils.tukey_window(
            (self._polar_height, self._polar_width)
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
        polar_spectrum_fft = np.fft.rfft2(
            self._polar_warp(spectrum) * self._polar_window
        )

        frame = Frame(
            image=image,
            normalized_filtered_image=normalized_filtered_image,
            polar_spectrum_fft=polar_spectrum_fft,
        )

        if self._debug:
            frame.set_spectrum(spectrum)

        return frame

    def track_frame(self: Tracker, ref: Frame, qry: Frame) -> None:
        """
        Track the query frame relative to the reference frame.

        Parameters:
            ref: The reference frame.
            qry: The query frame.
        """
        # Find the global rotation.
        theta, psr = self._find_global_rotation(ref, qry)
        print(f"theta={theta:.2f}, psr={psr:.2f}")

        # Warp the query image to neutralize the rotation.
        global_rotation_warped_image = transform.rotate(qry._image, theta=-theta)

        if self._debug:
            qry.set_global_rotation_warped_image(global_rotation_warped_image)

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

    def _find_global_rotation(
        self: Tracker, ref: Frame, qry: Frame
    ) -> tuple[float, float]:
        corr_map, t, psr = self._correlate(
            ref_fft=ref._polar_spectrum_fft, qry_fft=qry._polar_spectrum_fft
        )

        _, yt = t
        theta = math_utils.normalize_degrees(yt * (2.0 / self._polar_height) * 180.0)

        if self._debug:
            qry.set_global_rotation_corr(np.clip(corr_map, 0.0, 1.0))

        return theta, psr

    def _correlate(
        self: Tracker, ref_fft: NDArray[np.complex128], qry_fft: NDArray[np.complex128]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        cps = qry_fft * np.conj(ref_fft)

        cps[0].real = 0.0
        cps[0].imag = 0.0
        cps[1:] /= np.abs(cps[1:]) + 1e-15

        corr_map = np.fft.fftshift(np.fft.irfft2(cps))
        xy, _ = heatmap.peak_location(heatmap=corr_map)

        h, w = corr_map.shape
        t = xy - (w / 2.0, h / 2.0)

        psr = heatmap.peak_sidelobe_ratio(heatmap=corr_map, xy=xy)

        return corr_map, t, psr
