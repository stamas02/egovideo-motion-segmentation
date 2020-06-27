import argparse
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm

from src.frame_generator import FrameGenerator
from src.video import get_video_info

k_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDCClassifier on vide")
    parser.add_argument(
        "--video-file", "-v", type=str, help="Path to the video file to be processed"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Path to the log folder where the result is saved.",
    )
    return parser.parse_args()


def get_displacements(image1, image2):
    """ Calculate the displacement between "good" features 
    given two consecuitive grayscale images.

    Parameters
    ----------
    iamge1: numpy array,
        First image

    iamge1: numpy array,
        First image
        
    Returns
    -------
        - a set of 2D coordinates representing the features location in the first.
        - a set of 2D coordinates representing the matching features 
            location in the second image.
    """
    p1 = cv2.goodFeaturesToTrack(image1, mask=None, **feature_params)
    if p1 is None:
        return None, None

    p2, st, err = cv2.calcOpticalFlowPyrLK(image1, image2, p1, None, **k_params)
    if p2 is None:
        return None, None

    if not np.any(st == 1):
        return None, None

    origin = p1[st == 1].reshape(-1, 2)
    dispacement = p2[st == 1].reshape(-1, 2)
    return origin, dispacement


def main():
    args = parse_args()
    _, _, fps, _, h, w = get_video_info(args.video_file)
    fg = FrameGenerator(args.video_file, show_video_info=True, use_rgb=False)

    frame_iterator = iter(fg)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(join(args.output_dir, "op_flow.mp4"), fourcc, fps, (w, h))
    p_frame = next(frame_iterator)
    for frame in tqdm(
        frame_iterator, desc="playing video", unit="frame", total=len(fg) - 1
    ):
        draw_frame = np.array(frame)
        gray_p_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        origins, displacements = get_displacements(gray_p_frame, gray_frame)
        p_frame = frame
        if origins is None or displacements is None:
            out.write(draw_frame)
            continue
        # Draw vectors.
        for origin, displacement, in zip(origins, displacements):
            cv2.line(
                draw_frame, tuple(origin), tuple(displacement), (255, 0, 0), 2,
            )
            cv2.circle(draw_frame, tuple(origin), 1, (0, 255, 0))
        out.write(draw_frame)

    out.release()


if __name__ == "__main__":
    main()
