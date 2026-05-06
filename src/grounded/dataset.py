from __future__ import annotations

import pathlib
from typing import cast

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

import grounded.image.utils as image_utils


class Dataset:
    """
    Dataset utility class. Handles datasets where inside the specified directory find:
    - The file "image_names.txt", containing a list of image files, sorted in order.
    - The directory "rgb", containing the image files.
    """

    def __init__(self: Dataset, datadir: pathlib.Path, shape: tuple[int, int]) -> None:
        """
        Construct the dataset.

        Parameters:
            datadir: The data directory.
            shape: The requested shape of returned images.
        """

        if not datadir.is_dir():
            raise ValueError(f"Error: '{datadir}' is not a directory")

        image_names_path = datadir / "image_names.txt"
        if not image_names_path.is_file():
            raise ValueError(f"Error: file '{image_names_path}' is not found")

        rgb_path = datadir / "rgb"
        if not rgb_path.is_dir():
            raise ValueError(f"Error: directory '{rgb_path}' is not found")

        self._image_names = []
        with open(image_names_path, "r") as f:
            done = False
            while not done:
                name = f.readline().strip("\n")
                if name != "":
                    path = rgb_path / name
                    self._image_names.append(path)

                    if not path.is_file():
                        raise ValueError(f"Error: file '{path}' does not exist")
                else:
                    done = True

        self._shape = shape
        self._current = 0

    def __len__(self: Dataset) -> int:
        return len(self._image_names)

    def __iter__(self: Dataset) -> Dataset:
        self._current = 0
        return self

    def __next__(self: Dataset) -> NDArray[np.uint8]:
        if self._current < len(self):
            index = self._current
            self._current += 1
            return self._read_image(index)
        else:
            raise StopIteration

    def __getitem__(self: Dataset, index: int) -> NDArray[np.uint8]:
        return self._read_image(index)

    def _read_image(self: Dataset, index: int) -> NDArray[np.uint8]:
        if index < 0 or index >= len(self):
            raise IndexError("Error: Index is outside the range of the dataset")

        image = image_utils.read_gray(self._image_names[index])
        if image is None:
            raise FileExistsError(
                f"Error: Failed to read image with index {index} from dataset"
            )

        if image.shape != self._shape:
            h, w = self._shape
            return cast(
                NDArray[np.uint8],
                cv.resize(image, dsize=(w, h), interpolation=cv.INTER_LINEAR),
            )
        else:
            return image
