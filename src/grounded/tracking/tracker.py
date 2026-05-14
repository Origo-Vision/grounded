from __future__ import annotations

import math
from typing import Any, cast

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

import grounded.image.heatmap as heatmap
import grounded.image.transform as transform
import grounded.image.utils as image_utils
import grounded.math.matrix as matrix
import grounded.math.utils as math_utils
from grounded.tracking.frame import Frame


class Tracker:
    """
    Visual tracker class. Works on square images, where sides shall be
    power of two. Assuming to work in a homographic scene.
    """

    def __init__(
        self: Tracker, size: int, kcc: bool = False, debug: bool = False
    ) -> None:
        """
        Construct the tracker.

        Parameters:
            size: The side of the square image (shall be power of two).
            debug: Debug flag. When set, a lot more data is stored into frames.
        """
        if not math.log2(size).is_integer():
            raise ValueError("Size must be power of two")

        # General settings.
        self._image_height = size
        self._image_width = size
        self._polar_height = 360
        self._polar_width = size // 2
        self._polar_center = (self._image_width / 2, self._image_height / 2)
        self._polar_max_radius = size / 2.0

        # Polar filtering settings.
        self._rmin = 5
        self._rmax = 5

        # KCC specific settings.
        self._kernel_offset = 0.1
        self._kernel_power = 3.0
        self._kernel_lambda = 0.1

        # Runtime flags.
        self._kcc = kcc
        self._debug = debug

        # Coarse windowing.
        self._image_window = image_utils.tukey_window(
            (self._image_height, self._image_width)
        )
        self._polar_window = image_utils.tukey_window(
            (self._polar_height, self._polar_width)
        )

        # Patch settings and windowing.
        self._num_patches = 4
        self._patch_height = self._image_height // self._num_patches
        self._patch_width = self._image_width // self._num_patches

        self._patch_window = image_utils.tukey_window(
            (self._patch_height, self._patch_width)
        )

        # Targets for KCC.
        image_target = np.zeros(
            (self._image_height, self._image_width), dtype=np.float32
        )
        image_target[self._image_height // 2, self._image_width // 2] = 1.0
        self._image_target_fft = np.fft.rfft2(image_target)

        polar_target = np.zeros(
            (self._polar_height, self._polar_width), dtype=np.float32
        )
        polar_target[self._polar_height // 2, self._polar_width // 2] = 1.0
        self._polar_target_fft = np.fft.rfft2(polar_target)

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
        image_fft = np.fft.rfft2(normalized_filtered_image)
        spectrum = self._create_spectrum(normalized_filtered_image)
        polar_spectrum_fft = np.fft.rfft2(
            self._polar_warp(spectrum) * self._polar_window
        )

        frame = Frame(
            image=image,
            image_fft=image_fft,
            polar_spectrum_fft=polar_spectrum_fft,
        )

        if self._debug:
            frame.set_spectrum(spectrum)

        return frame

    def track_frame(
        self: Tracker, ref: Frame, qry: Frame
    ) -> tuple[NDArray[np.float64], float]:
        """
        Track the query frame relative to the reference frame.

        Parameters:
            ref: The reference frame.
            qry: The query frame.

        Returns:
            Tuple affine forward matrix, and psr from coarse registration.
        """
        coarse = (
            self._registration_kcc(
                ref=ref,
                qry_image=qry._image,
                qry_polar_spectrum_fft=qry._polar_spectrum_fft,
            )
            if self._kcc
            else self._registration_fmt(
                ref=ref,
                qry_image=qry._image,
                qry_polar_spectrum_fft=qry._polar_spectrum_fft,
            )
        )

        coarse_warped = coarse["warped"]

        normalized_filtered_image = image_utils.normalized(
            coarse_warped * self._image_window
        )
        spectrum = self._create_spectrum(normalized_filtered_image)
        polar_spectrum_fft = np.fft.rfft2(
            self._polar_warp(spectrum) * self._polar_window
        )

        fine = (
            self._registration_kcc(
                ref=ref,
                qry_image=coarse_warped,
                qry_polar_spectrum_fft=polar_spectrum_fft,
            )
            if self._kcc
            else self._registration_fmt(
                ref=ref,
                qry_image=coarse_warped,
                qry_polar_spectrum_fft=polar_spectrum_fft,
            )
        )

        A = fine["affine"] @ coarse["affine"]
        qry._H = A @ ref._H

        psr = (coarse["psr"] + fine["psr"]) / 2.0

        if self._debug:
            qry.set_coarse_rotation_corr(np.clip(coarse["rotation_corr"], 0.0, 1.0))
            qry.set_coarse_translation_corr(
                np.clip(coarse["translation_corr"], 0.0, 1.0)
            )
            qry.set_coarse_warped_image(coarse_warped)
            qry.set_fine_rotation_corr(np.clip(fine["rotation_corr"], 0.0, 1.0))
            qry.set_fine_translation_corr(np.clip(fine["translation_corr"], 0.0, 1.0))
            qry.set_fine_warped_image(fine["warped"])

        return A, psr


    def _registration_fmt(
        self: Tracker,
        ref: Frame,
        qry_image: NDArray[np.uint8],
        qry_polar_spectrum_fft: NDArray[np.complex128],
    ) -> dict[str, Any]:
        # Find the global rotation.
        rotation_corr, rotation_offset, rotation_psr = self._correlate(
            ref_fft=ref._polar_spectrum_fft, qry_fft=qry_polar_spectrum_fft
        )

        _, yt = rotation_offset
        theta = math_utils.normalize_degrees(yt * (2.0 / self._polar_height) * 180.0)
        #print(f"fmt theta={theta:.2f}, psr={rotation_psr:.2f}")

        # Rectify the query image with regards to the rotation.
        rotated = transform.warp_affine(qry_image, theta=-theta, xt=0.0, yt=0.0)

        # Find the global translation using the rectified image.
        translation_corr, translation_offset, translation_psr = self._correlate(
            ref_fft=ref._image_fft,
            qry_fft=np.fft.rfft2(image_utils.normalized(rotated) * self._image_window),
        )

        # The translation offset vector is rotated R(-theta) @ t. Rotate with theta
        # to get the true translation.
        xt, yt, _ = matrix.rotate(theta) @ np.append(translation_offset, 1.0)
        #print(f"fmt xt={xt:.2f}, yt={yt:.2f}, psr={translation_psr:.2f}")

        # Create the forward (ref => qry) affine matrix.
        M = matrix.affine(
            theta=theta,
            xt=xt,
            yt=yt,
            cx=(self._image_width - 1) * 0.5,
            cy=(self._image_height - 1) * 0.5,
        )

        # Get the reverse (qry => ref) affine matrix for warping.
        Minv = np.linalg.inv(M)
        warped = cast(
            NDArray[np.uint8],
            cv.warpAffine(
                qry_image, M=Minv[:2], dsize=(self._image_width, self._image_height)
            ),
        )

        return {
            "affine": M,
            "warped": warped,
            "psr": (rotation_psr + translation_psr) / 2,
            "rotation_corr": rotation_corr,
            "translation_corr": translation_corr,
        }

    def _registration_kcc(
        self: Tracker,
        ref: Frame,
        qry_image: NDArray[np.uint8],
        qry_polar_spectrum_fft: NDArray[np.complex128],
    ) -> dict[str, Any]:
        # Find the global rotation.
        rotation_offset, rotation_psr, rotation_corr = (
            self._calculate_kernel_correlation(
                fft=qry_polar_spectrum_fft,
                ref_fft=ref._polar_spectrum_fft,
                target_fft=self._polar_target_fft,
            )
        )

        _, yt = rotation_offset
        theta = math_utils.normalize_degrees(yt * (2.0 / self._polar_height) * 180.0)
        #print(f"kcc theta={theta:.2f}, psr={rotation_psr:.2f}")

        # Rectify the query image with regards to the rotation.
        rotated = transform.warp_affine(qry_image, theta=-theta, xt=0.0, yt=0.0)

        # Find the global translation using the rectified image.
        translation_offset, translation_psr, translation_corr = (
            self._calculate_kernel_correlation(
                fft=np.fft.rfft2(image_utils.normalized(rotated) * self._image_window),
                ref_fft=ref._image_fft,
                target_fft=self._image_target_fft,
            )
        )

        # The translation offset vector is rotated R(-theta) @ t. Rotate with theta
        # to get the true translation.
        xt, yt, _ = matrix.rotate(theta) @ np.append(translation_offset, 1.0)
        #print(f"kcc xt={xt:.2f}, yt={yt:.2f}, psr={translation_psr:.2f}")

        # Create the forward (ref => qry) affine matrix.
        M = matrix.affine(
            theta=theta,
            xt=xt,
            yt=yt,
            cx=(self._image_width - 1) * 0.5,
            cy=(self._image_height - 1) * 0.5,
        )

        # Get the reverse (qry => ref) affine matrix for warping.
        Minv = np.linalg.inv(M)
        warped = cast(
            NDArray[np.uint8],
            cv.warpAffine(
                qry_image, M=Minv[:2], dsize=(self._image_width, self._image_height)
            ),
        )

        return {
            "affine": M,
            "warped": warped,
            "psr": (rotation_psr + translation_psr) / 2,
            "rotation_corr": rotation_corr,
            "translation_corr": translation_corr,
        }

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

    def _correlate(
        self: Tracker, ref_fft: NDArray[np.complex128], qry_fft: NDArray[np.complex128]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        cps = qry_fft * np.conj(ref_fft)

        cps[0].real = 0.0
        cps[0].imag = 0.0
        cps[1:] /= np.abs(cps[1:]) + 1e-15

        corr_map = np.fft.fftshift(np.fft.irfft2(cps))
        if np.max(corr_map) < 1e-07:
            raise ValueError("Zero valued correlation map")

        xy, _ = heatmap.peak_location(heatmap=corr_map)

        h, w = corr_map.shape
        offset = xy - (w / 2.0, h / 2.0)

        psr = heatmap.peak_sidelobe_ratio(heatmap=corr_map, xy=xy)

        return corr_map, offset, psr

    def _calculate_kernel_correlation(
        self: Tracker,
        fft: NDArray[np.complex128],
        ref_fft: NDArray[np.complex128],
        target_fft: NDArray[np.complex128],
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
        assert fft.shape == ref_fft.shape
        assert fft.shape == target_fft.shape

        kzz = self._calculate_kernel(xf=ref_fft)
        kxz = self._calculate_kernel(xf=fft, zf=ref_fft)

        H = target_fft / (kzz + self._kernel_lambda)
        G = H * kxz

        g = np.fft.irfft2(G)
        xy, response = heatmap.peak_location(heatmap=g)
        psr = heatmap.peak_sidelobe_ratio(heatmap=g, xy=xy)

        # Infer the image shape from the FFT.
        h, w = fft.shape
        w = (w - 1) * 2

        translation = xy - [w / 2.0, h / 2.0]

        return translation, psr, g

    def _calculate_kernel(
        self: Tracker,
        xf: NDArray[np.complex128],
        zf: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        zfc = np.conj(xf) if zf is None else np.conj(zf)
        xzf = xf * zfc
        xz = np.fft.irfft2(xzf)

        kernel = np.pow(xz + self._kernel_offset, self._kernel_power)
        kernel /= np.max(np.abs(kernel))

        return np.fft.rfft2(kernel)
