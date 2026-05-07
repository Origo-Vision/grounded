import argparse
import pathlib
import sys

from matplotlib import pyplot as plt

from grounded.dataset import Dataset
import grounded.image.utils as image_utils


def main(options: argparse.Namespace) -> int:
    dataset = Dataset(options.datadir, shape=(256, 256))
    image = image_utils.normalized(dataset[10])
    window = image_utils.tukey_window(shape=(256, 256))

    plt.imshow(image * window, cmap="gray")
    plt.show()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datadir", type=pathlib.Path, help="Path to dataset directory")
    options = parser.parse_args()

    try:
        sys.exit(main(options))
    except Exception as e:
        print(f"{e}")
        sys.exit(1)
