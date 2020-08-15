import numpy as np
import os
import argparse
from tqdm import tqdm
import textwrap
from src.video import get_video_info
from src.frame_generator import FrameGeneratorVideo, FrameGeneratorImageSequence
from src.grid_optical_flow import get_grid_flow, get_grid_centres
import logging
import cv2
import pickle

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

OUTPUT_FRAME_SIZE = (400, 400)


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent(
                                         '''classify video frames stationary or moving based on optical flow'''))
    parser.add_argument('--video', '-v', type=str, help="path to the videofile")
    parser.add_argument("--output-dir", "-o", type=str,
                        help="Folder where the segmentation and plots are saved")
    parser.add_argument(
        "--grid-size",
        "-g",
        nargs="+",
        type=int,
        help="A touple representing the nrows and ncols of the grid.",
    )
    parser.add_argument("--video_type", type=str, default="video", choices=["video", "image_sequence"],
                        help="Folder where the segmentation and plots are saved")

    args = parser.parse_args()
    return args


def calculate_optical_flow(video,video_type, grid_size, output_dir):

    if video_type == "video":
        fg = FrameGeneratorVideo(video, show_video_info=True, use_rgb=False)
    elif video_type == "image_sequence":
        fg = FrameGeneratorImageSequence(video, use_rgb=False)

    # get the first frame
    frame_iterator = iter(fg)
    p_frame = next(frame_iterator)
    p_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)

    # The optical flow belonging to the 1st frame is 0 in its magnitude.
    optical_flow_data = [[],[]]

    logging.info("Calculation optical flow".format(video))

    out_video_file = os.path.join(output_dir, "optical_flow.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_video_file, fourcc, 25.0, OUTPUT_FRAME_SIZE)

    for frame in tqdm(frame_iterator, desc="playing video", unit="frame", total=len(fg) - 1):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        origins, displacements = get_grid_flow(p_frame, gray_frame, grid_size[0], grid_size[1])
        optical_flow_data[0].append(origins)
        optical_flow_data[1].append(displacements)
        p_frame = gray_frame
        if output_dir is not None:
            out_frame = np.array(frame)
            for v0, v1 in zip(np.reshape(origins, (-1, 2)), np.reshape(displacements, (-1, 2))):
                out_frame = cv2.line(out_frame, tuple(v0), tuple(v1), (0, 255, 0), thickness=5)
                out_frame = cv2.circle(out_frame, tuple(v0), 5, (0, 0, 255),-1)
            out_frame = cv2.resize(out_frame, OUTPUT_FRAME_SIZE)
            writer.write(out_frame)
    optical_flow_data[0].insert(0, optical_flow_data[0][0])
    optical_flow_data[1].insert(0, optical_flow_data[1][0])
    optical_flow_data = np.array(optical_flow_data)
    writer.release()
    with open(os.path.join(output_dir, "optical_flow.npy"), 'wb') as handle:
        pickle.dump(optical_flow_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Optical flow computation Done!")


if __name__ == "__main__":
    args = parseargs()
    calculate_optical_flow(**args.__dict__)
