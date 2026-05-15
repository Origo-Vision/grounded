import argparse
import pathlib
import sys

import cv2 as cv


def main(options: argparse.Namespace) -> int:
    if not options.videopath.is_file():
        print("Error: The video path does not exist")
        return 1
    
    capture = cv.VideoCapture(str(options.videopath))
    if not capture.isOpened():
        print("Error: Failed to open the video")
        return 1

    if options.datadir.exists() and not options.datadir.is_dir():
        print("Error: The data directory path exists, but is not a directory")
        return 1
    
    options.datadir.mkdir(parents=True, exist_ok=True)

    n = 0
    while True:
        status, image = capture.read()
        if not status:
            print("Info: Video has ended")
            break

        path = options.datadir / f"scene_{n:05d}.png"
        cv.imwrite(str(path), image)

        print(f"Info: import {path}")

        n += 1

    capture.release()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "videopath", type=pathlib.Path, help="Path to the video to be imported"
    )
    parser.add_argument(
        "--datadir",
        type=pathlib.Path,
        required=True,
        help="Path to destination data directory",
    )
    options = parser.parse_args()
    sys.exit(main(options))
