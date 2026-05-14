import argparse
import pathlib
import sys

from matplotlib import pyplot as plt
import numpy as np

from grounded.dataset import Dataset
import grounded.math.matrix as matrix
from grounded.tracking.tracker import Tracker
import grounded.tracking.stitching as stitching


def main(options: argparse.Namespace) -> int:
    dataset = Dataset(options.datadir, shape=(options.size, options.size))
    tracker = Tracker(size=options.size, kcc=options.kcc, debug=True)

    start = options.start
    end = (
        min(len(dataset), start + options.num_items)
        if options.num_items is not None
        else len(dataset)
    )

    keyframes = []
    for i in range(start, end):
        frame = tracker.new_frame(image=dataset[i])

        if keyframes == []:
            # First frame is always designated as keyframe.
            keyframes.append(frame)
            continue

        print(f"Frame #{frame.id()} tracking keyframe #{keyframes[-1].id()}")

        A, psr = tracker.track_frame(ref=keyframes[-1], qry=frame)
        xy, theta = matrix.decomp_affine(
            M=A, cx=(options.size - 1) * 0.5, cy=(options.size - 1) * 0.5
        )
        print(
            f" theta={theta:.2f}{chr(176)}, xt={xy[0]:.2f}px, yt={xy[1]:.2f}px, psr={psr:.2f}"
        )

        # Check if we should promote this frame to a keyframe.
        dist = np.linalg.norm(xy)
        if (
            psr < options.thr_psr
            or theta > options.thr_theta
            or dist >= options.size * options.thr_translate
        ):
            print(f" frame #{frame.id()} is promoted to keyframe")
            keyframes.append(frame)

    print(f"Stitching {len(keyframes)} frames")
    map = stitching.stitch_frames(frames=keyframes)
    if map is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(map, cmap="gray")
        plt.title("Stitched Track Map")
        plt.axis("off")
        plt.show()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datadir", type=pathlib.Path, help="Path to dataset directory")
    parser.add_argument(
        "--size", choices=[256, 512], default=256, help="The image size"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="First image in the dataset"
    )
    parser.add_argument(
        "--num-items", type=int, default=None, help="The number of items to track"
    )
    parser.add_argument("--kcc", action="store_true", help="Use the KCC based tracker")
    parser.add_argument("--thr-psr", type=float, default=6.0, help="PSR threshold")
    parser.add_argument(
        "--thr-translate",
        type=float,
        default=0.1,
        help="Translation threshold (percentage of image)",
    )
    parser.add_argument(
        "--thr-theta", type=float, default=10.0, help="Rotation threshold"
    )

    options = parser.parse_args()

    sys.exit(main(options))
