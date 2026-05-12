import argparse
import pathlib
import sys

import cv2 as cv
from matplotlib import pyplot as plt

from grounded.dataset import Dataset
import grounded.image.transform as transform
import grounded.math.matrix as matrix
import grounded.tracking.stitching as stitching
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
        qry_image = transform.warp_affine(
            ref_image, theta=options.theta, xt=options.xt, yt=options.yt
        )

    # Get the frames.
    ref = tracker.new_frame(image=ref_image)
    qry = tracker.new_frame(image=qry_image)

    # Track the query relative to the reference.
    A, psr = tracker.track_frame(ref=ref, qry=qry)
    xy, theta = matrix.decomp_affine(
        M=A, cx=(options.size - 1) * 0.5, cy=(options.size - 1) * 0.5
    )
    print(
        f"> theta={theta:.2f}{chr(176)}, xt={xy[0]:.2f}px, yt={xy[1]:.2f}px, psr={psr:.2f}"
    )

    # Stitch the frames to a map
    map = stitching.stitch_frames(frames=[ref, qry])

    if map is not None:
        plt.figure(figsize=(20, 12))
        plt.imshow(map, cmap="gray")
        plt.axis("off")
        plt.title("Stitched map")

    # Debug images.
    plt.figure(figsize=(20, 12))

    # Reference images.
    plt.subplot(4, 3, 1)
    plt.imshow(ref._image, cmap="gray")
    plt.axis("off")
    plt.title("Ref image")

    plt.subplot(4, 3, 2)
    plt.imshow(ref._spectrum, cmap="gray")
    plt.axis("off")
    plt.title("Ref spectrum")

    # Query images.
    plt.subplot(4, 3, 4)
    plt.imshow(qry._image, cmap="gray")
    plt.axis("off")
    plt.title("Qry image")

    plt.subplot(4, 3, 5)
    plt.imshow(qry._spectrum, cmap="gray")
    plt.axis("off")
    plt.title("Qry spectrum")

    # Coarse registration images.
    plt.subplot(4, 3, 7)
    plt.imshow(qry._coarse_warped_image, cmap="gray")
    plt.axis("off")
    plt.title("Coarse warped image (qry => ref)")

    plt.subplot(4, 3, 8)
    plt.imshow(qry._coarse_rotation_corr, cmap="gray")
    plt.axis("off")
    plt.title("Coarse rotation corr")

    plt.subplot(4, 3, 9)
    plt.imshow(qry._coarse_translation_corr, cmap="gray")
    plt.axis("off")
    plt.title("Coarse translation corr")

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

    sys.exit(main(options))

    # try:
    #    sys.exit(main(options))
    # except Exception as e:
    #    print(f"{e}")
    #    sys.exit(1)
