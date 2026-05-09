from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Frame:
    next_id = 1

    def __init__(
        self: Frame,
        image: NDArray[np.uint8],
        normalized_filtered_image: NDArray[np.float64],
        polar_spectrum_fft: NDArray[np.complex128],
    ) -> None:
        self._id = Frame.next_id
        Frame.next_id += 1

        self._image = image
        self._normalized_filtered_image = normalized_filtered_image
        self._polar_spectrum_fft = polar_spectrum_fft

        # Optional fields, set to a value when tracker is in debug mode.
        self._spectrum: NDArray[np.float64] | None = None
        self._global_rotation_corr: NDArray[np.float64] | None = None
        self._global_rotation_warped_image: NDArray[np.uint8] | None = None

    def set_spectrum(self: Frame, spectrum: NDArray[np.float64]) -> None:
        self._spectrum = spectrum

    def set_global_rotation_corr(
        self: Frame, global_rotation_corr: NDArray[np.float64]
    ) -> None:
        self._global_rotation_corr = global_rotation_corr

    def set_global_rotation_warped_image(
        self: Frame, global_rotation_warped_image: NDArray[np.uint8]
    ) -> None:
        self._global_rotation_warped_image = global_rotation_warped_image
