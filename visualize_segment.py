import os
import argparse
from tqdm import tqdm
import textwrap
from src.video import get_video_info
from src.frame_generator import FrameGenerator
import logging
from report_segmentation import render_report
import cv2
import pickle
from src.segmnet import segment_motion, segment_transition
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
OUTPUT_FRAME_SIZE = (400, 400)


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent(
                                         '''segment video based on optical flow'''))
    parser.add_argument('--optical-flow-file', '-f', type=str, help="path to the video file")
    parser.add_argument('--video_file', '-v', type=str, help="path to the video file")
    parser.add_argument('--transition-threshold', type=float, help="The threshold parameter for transition segmentation")
    parser.add_argument('--motion-threshold', type=float, help="The threshold parameter for motion segmentation")
    parser.add_argument("--output-dir", "-o", type=str,
                        help="Folder where the segmentation and plots are saved")

    args = parser.parse_args()
    return args


def do_segmentation(video_file, optical_flow_file, transition_threshold, motion_threshold, output_dir):
    with open(optical_flow_file, 'rb') as handle:
        optical_flow = pickle.load(handle)

    _, _, fps, _, h, w = get_video_info(video_file)
    fg = FrameGenerator(video_file, show_video_info=True, use_rgb=False)

    out_video_file = os.path.join(output_dir, "segmented.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_video_file, fourcc, 25.0, OUTPUT_FRAME_SIZE)

    frame_cnt = 0
    motion_segment_cnt = 0
    segmentation = pd.DataFrame()
    for is_transition, is_new_segment, frame in tqdm(zip(segment_transition(optical_flow, transition_threshold),
                                                         segment_motion(optical_flow, motion_threshold),
                                                         fg),
                                                     total=len(fg),
                                                     unit="Frame"):
        frame_cnt += 1
        segmentation = segmentation.append(
            {"Is transition": is_transition,
             "Is new segment": is_new_segment,
             "Frame": frame_cnt}, ignore_index=True)
        if is_new_segment:
            motion_segment_cnt += 1
        transition_text = "Transition:{}".format(str(is_transition))
        motion_segment_text = "sgmt:{}".format(str(motion_segment_cnt))
        frame = cv2.resize(frame, OUTPUT_FRAME_SIZE)
        frame = cv2.putText(frame, transition_text, (0, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        frame = cv2.putText(frame, motion_segment_text, (0, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        writer.write(frame)

if __name__ == "__main__":
    args = parseargs()
    do_segmentation(**args.__dict__)
