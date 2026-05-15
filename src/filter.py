import argparse
import pathlib
import sys

from grounded.dataset import Dataset
import grounded.image.utils as image_utils

from matplotlib import pyplot as plt


def main(options: argparse.Namespace) -> int:
    dataset = Dataset(options.datadir, shape=(options.size, options.size))
    mask = image_utils.bandpass_mask(
        shape=(options.size, options.size), low=options.low, high=options.high
    )

    ref_image = image_utils.normalized(dataset[options.reference])
    qry_image = image_utils.normalized(dataset[options.query])

    ref_image_filtered = image_utils.bandpass_filtered(
        ref_image, low=options.low, high=options.high
    )
    qry_image_filtered = image_utils.bandpass_filtered(
        qry_image, low=options.low, high=options.high
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.title("Bandpass filter mask")

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(ref_image, cmap="gray")
    plt.axis("off")
    plt.title("Ref original")

    plt.subplot(2, 2, 2)
    plt.imshow(ref_image_filtered, cmap="gray")
    plt.axis("off")
    plt.title("Ref filtered")

    plt.subplot(2, 2, 3)
    plt.imshow(qry_image, cmap="gray")
    plt.axis("off")
    plt.title("Qry original")

    plt.subplot(2, 2, 4)
    plt.imshow(qry_image_filtered, cmap="gray")
    plt.axis("off")
    plt.title("Qry filtered")

    plt.tight_layout()

    plt.show()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datadir", type=pathlib.Path, help="Path to dataset directory")
    parser.add_argument(
        "--reference",
        type=int,
        required=True,
        help="The reference image index from the dataset",
    )
    parser.add_argument(
        "--query",
        type=int,
        required=True,
        help="The query index from the dataset",
    )
    parser.add_argument(
        "--size", type=int, choices=(128, 256, 512), default=256, help="The image size"
    )
    parser.add_argument("--low", type=float, default=0.05, help="The low cutoff")
    parser.add_argument("--high", type=float, default=0.3, help="The high cutoff")
    options = parser.parse_args()

    sys.exit(main(options))
