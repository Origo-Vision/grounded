from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Frame:
    next_id = 1

    def __init__(
        self: Frame,
        image: NDArray[np.uint8],
        image_fft: NDArray[np.complex128],
        polar_spectrum_fft: NDArray[np.complex128],
    ) -> None:
        self._id = Frame.next_id
        Frame.next_id += 1

        self._H = np.eye(3)

        self._image = image
        self._image_fft = image_fft
        self._polar_spectrum_fft = polar_spectrum_fft

        # Optional fields, set to a value when the tracker is in debug mode.
        self._spectrum: NDArray[np.float64] | None = None
        self._coarse_rotation_corr: NDArray[np.float64] | None = None
        self._coarse_translation_corr: NDArray[np.float64] | None = None
        self._coarse_warped_image: NDArray[np.uint8] | None = None
        self._fine_corr: NDArray[np.float64] | None = None
        self._fine_warped_image: NDArray[np.uint8] | None = None

    def id(self: Frame) -> int:
        return self._id

    def set_spectrum(self: Frame, spectrum: NDArray[np.float64]) -> None:
        self._spectrum = spectrum

    def set_coarse_rotation_corr(
        self: Frame, coarse_rotation_corr: NDArray[np.float64]
    ) -> None:
        self._coarse_rotation_corr = coarse_rotation_corr

    def set_coarse_translation_corr(
        self: Frame, coarse_translation_corr: NDArray[np.float64]
    ) -> None:
        self._coarse_translation_corr = coarse_translation_corr

    def set_coarse_warped_image(
        self: Frame, coarse_warped_image: NDArray[np.uint8]
    ) -> None:
        self._coarse_warped_image = coarse_warped_image

    def set_fine_corr(self: Frame, fine_corr: NDArray[np.float64]) -> None:
        self._fine_corr = fine_corr

    def set_fine_warped_image(
        self: Frame, fine_warped_image: NDArray[np.uint8]
    ) -> None:
        self._fine_warped_image = fine_warped_image
