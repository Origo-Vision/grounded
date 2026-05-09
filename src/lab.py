import argparse
import pathlib
import sys

import cv2 as cv
from matplotlib import pyplot as plt

from grounded.dataset import Dataset
import grounded.image.transform as transform
from grounded.tracking.tracker import Tracker


def main(options: argparse.Namespace) -> int:
    dataset = Dataset(options.datadir, shape=(options.size, options.size))
    tracker = Tracker(size=options.size, debug=True)

    # Get the reference image.
    ref_image = dataset[options.reference]

    # Get the query image.
    if options.query is not None:
        qry_image = dataset[options.query]
    else:
        qry_image = transform.translate(ref_image, xy=(options.xt, options.yt))
        qry_image = transform.rotate(qry_image, theta=options.theta)

    # Get the frames.
    ref = tracker.new_frame(image=ref_image)
    qry = tracker.new_frame(image=qry_image)

    plt.figure(figsize=(20, 12))

    # Reference images.
    plt.subplot(2, 3, 1)
    plt.imshow(ref._image, cmap="gray")
    plt.axis("off")
    plt.title("Ref image")

    plt.subplot(2, 3, 2)
    plt.imshow(ref._normalized_filtered_image, cmap="gray")
    plt.axis("off")
    plt.title("Ref filtered")

    plt.subplot(2, 3, 3)
    plt.imshow(ref._spectrum, cmap="gray")
    plt.axis("off")
    plt.title("Ref spectrum")

    # Query images.
    plt.subplot(2, 3, 4)
    plt.imshow(qry._image, cmap="gray")
    plt.axis("off")
    plt.title("Qry image")

    plt.subplot(2, 3, 5)
    plt.imshow(qry._normalized_filtered_image, cmap="gray")
    plt.axis("off")
    plt.title("Qry filtered")

    plt.subplot(2, 3, 6)
    plt.imshow(qry._spectrum, cmap="gray")
    plt.axis("off")
    plt.title("Qry spectrum")

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
        default=0,
        help="The reference image index from the dataset",
    )
    parser.add_argument(
        "--query",
        type=int,
        default=None,
        help="The (optional) query index from the dataset",
    )
    parser.add_argument(
        "--size", choices=(256, 512), default=256, help="The image size"
    )
    parser.add_argument(
        "--theta", type=float, default=0.0, help="Rotation angle (degrees)"
    )
    parser.add_argument("--xt", type=float, default=0.0, help="Translation in x")
    parser.add_argument("--yt", type=float, default=0.0, help="Translation in y")

    options = parser.parse_args()

    try:
        sys.exit(main(options))
    except Exception as e:
        print(f"{e}")
        sys.exit(1)
