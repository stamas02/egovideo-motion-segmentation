import numpy as np
import os
import argparse
from tqdm import tqdm
import textwrap
from src.video import get_video_info
from src.frame_generator import FrameGenerator
from src.grid_optical_flow import get_grid_flow, get_grid_centres
import pandas as pd
import logging
from report_segmentation import render_report
import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent(
                                         '''classify video frames stationary or moving based on optical flow'''))
    parser.add_argument('--video', '-v', type=str, help="path to the videofile")
    parser.add_argument('--threshold', '-t', type=float, help="The threshold parameter for segmentation")
    parser.add_argument("--output-dir", "-o", type=str,
                        help="Folder where the segmentation and plots are saved")
    parser.add_argument(
        "--grid-size",
        "-g",
        nargs="+",
        type=int,
        help="A touple representing the ncols and nrows of the grid.",
    )

    args = parser.parse_args()
    return args


def segment(video, output_dir, grid_size, threshold):
    _, _, fps, _, h, w = get_video_info(video)
    fg = FrameGenerator(video, show_video_info=True, use_rgb=False)

    # center and normalize grid centres
    frame_iterator = iter(fg)
    p_frame = next(frame_iterator)
    p_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
    segmentation = pd.DataFrame()

    logging.info("Segment video {}".format(video))
    out_file = os.path.join(output_dir, "segmented.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_file, fourcc, 25.0, (200,200))

    total_optical_flow = None
    for i, frame in tqdm(enumerate(frame_iterator), desc="playing video", unit="frame", total=len(fg) - 1):
        original_frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        optical_flow = get_grid_flow(p_frame, frame, grid_size)
        if total_optical_flow is None:
            total_optical_flow = optical_flow
        else:
            total_optical_flow += optical_flow
        magnitudes = np.sqrt(np.sum(np.power(total_optical_flow, 2), axis=2))
        mean_magnitude = np.mean(magnitudes)
        segmentation = segmentation.append(
            {"Name": "magnitude",
             "Magnitude": mean_magnitude,
             "Frame": i,
             "Threshold": threshold,
             "Class": "stationary" if mean_magnitude < threshold else "moving"},
            ignore_index=True)
        if mean_magnitude > threshold:
            total_optical_flow = None
        p_frame = frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        out_frame = cv2.resize(original_frame, (200,200))
        out_frame = cv2.putText(out_frame, str(mean_magnitude), (10, 20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(out_frame)
    writer.release()
    segmentation.to_pickle(os.path.join(output_dir, "segmentation.pickle"))
    logging.info("Segment Done!")
    render_report(os.path.join(output_dir, "segmentation.pickle"), output_dir)


if __name__ == "__main__":
    args = parseargs()
    segment(**args.__dict__)
