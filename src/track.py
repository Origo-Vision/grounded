import argparse
import pathlib
import sys

from matplotlib import pyplot as plt

from grounded.dataset import Dataset
import grounded.math.matrix as matrix
from grounded.tracking.tracker import Tracker
import grounded.tracking.stitching as stitching


def main(options: argparse.Namespace) -> int:
    image_size = 256
    dataset = Dataset(options.datadir, shape=(image_size, image_size))
    tracker = Tracker(size=image_size, debug=False)

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
            M=A, cx=(image_size - 1) * 0.5, cy=(image_size - 1) * 0.5
        )
        print(
            f" theta={theta:.2f}{chr(176)}, xt={xy[0]:.2f}px, yt={xy[1]:.2f}px, psr={psr:.2f}"
        )

        keyframes.append(frame)

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
        "--start", type=int, default=0, help="First image in the dataset"
    )
    parser.add_argument(
        "--num-items", type=int, default=None, help="The number of items to track"
    )

    options = parser.parse_args()

    sys.exit(main(options))
