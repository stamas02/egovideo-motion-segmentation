import argparse
import logging
import os
import pickle
import textwrap

import cv2
from tqdm import tqdm

from src.frame_generator import FrameGeneratorVideo, FrameGeneratorImageSequence
from src.segmnet import segment_view, segment_visit
from src.utils import view_sparse_segmentation_to_df, visit_sparse_segmentation_to_df
from src.video import get_video_info

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
OUTPUT_FRAME_SIZE = (400, 400)


def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=textwrap.dedent(
                                         '''segment video based on optical flow'''))
    parser.add_argument('--optical-flow-file', '-f', type=str, help="path to the video file")
    parser.add_argument('--video_file', '-v', type=str, help="path to the video file")
    parser.add_argument('--transition-threshold', type=float, help="The threshold parameter for visit segmentation")
    parser.add_argument('--motion-threshold', type=float, help="The threshold parameter for view segmentation")
    parser.add_argument('--min_view_section_length', type=float, default=25,
                        help="The minimum number of frames a view has to contain")
    parser.add_argument('--min_visit_section_length', type=float, default=75,
                        help="The minimum number of frames a visit has to contain")
    parser.add_argument("--output-dir", "-o", type=str,
                        help="Folder where the segmentation and plots are saved")
    parser.add_argument("--video_type", type=str, default="video", choices=["video", "image_sequence"],
                        help="Folder where the segmentation and plots are saved")

    args = parser.parse_args()
    return args


def do_segmentation(video_file, video_type, optical_flow_file, transition_threshold, motion_threshold, min_view_section_length,
                    min_visit_section_length, output_dir
                    ):
    with open(optical_flow_file, 'rb') as handle:
        optical_flow = pickle.load(handle)

    if video_type == "video":
        fg = FrameGeneratorVideo(video_file, show_video_info=True, use_rgb=False)
    elif video_type == "image_sequence":
        fg = FrameGeneratorImageSequence(video_file, use_rgb=False)

    out_video_file = os.path.join(output_dir, "segmented.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_video_file, fourcc, 25.0, OUTPUT_FRAME_SIZE)

    frame_cnt = 0
    visit_segmentation = []
    view_segmentation = []
    view_count = 0

    for is_visit, is_new_segment, frame in tqdm(zip(segment_visit(optical_flow, transition_threshold),
                                                    segment_view(optical_flow, motion_threshold),
                                                    fg),
                                                total=len(fg),
                                                unit="Frame"):
        frame_cnt += 1
        visit_segmentation.append(is_visit)
        view_count += int(is_new_segment)
        view_count += 0 if len(visit_segmentation) < 2 else int(visit_segmentation[-2] != visit_segmentation[-1])
        view_segmentation.append(view_count)

        transition_text = "In visit:{}".format(str(is_visit))
        motion_segment_text = "sgmt:{}".format(str(view_count))
        frame = cv2.resize(frame, OUTPUT_FRAME_SIZE)
        frame = cv2.putText(frame, transition_text, (0, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        frame = cv2.putText(frame, motion_segment_text, (0, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        writer.write(frame)
    writer.release()
    view_segmentation_df = view_sparse_segmentation_to_df(view_segmentation, min_view_section_length)
    visit_segmentation_df = visit_sparse_segmentation_to_df(visit_segmentation, min_visit_section_length)
    view_segmentation_df.to_pickle(os.path.join(output_dir, "view_segmentation.pickle"))
    visit_segmentation_df.to_pickle(os.path.join(output_dir, "visit_segmentation.pickle"))

if __name__ == "__main__":
    args = parseargs()
    do_segmentation(**args.__dict__)
