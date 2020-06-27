import argparse
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm

from src.frame_generator import FrameGenerator
from src.optical_flow import get_displacements
from src.video import get_video_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an optical flow for a video.")
    parser.add_argument(
        "--video-file", "-v", type=str, help="Path to the video file to be processed"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Path to the log folder where the result is saved.",
    )
    parser.add_argument(
        "--grid-size",
        "-g",
        nargs="+",
        type=int,
        help="A touple representing the ncols and nrows of the grid.",
    )
    return parser.parse_args()


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    h, w = array.shape

    assert h % nrows == 0, "{} height is not divisble by nrows {}".format(h, nrows)
    assert w % ncols == 0, "{} width is not divisble by rows {}".format(w, ncols)
    return np.swapaxes(array.reshape(nrows, h // nrows, ncols, w // ncols), 1, 2)


def get_grid_flow(image1, image2, grid):
    """ Calculate an optical flow for each image block defind by grid.

    Parameters
    ----------
    image1 : numpy array
        image1 a grayscale image
    image2 : numpy array
        image2 a grayscale image
    grid : Tuple(int, int)
        Defines the grid size st (ncols, nrows)

    Return
    ------
        A numpy array of shape (ncols, nrows, 2) where each image block is assigned with
        a 2D vector (relative to is centre) representing its average displacement.
    """
    # Get image blocks and flatten the grid dimensions.
    image1_blocks = split(image1, grid[0], grid[1])
    image1_blocks = image1_blocks.reshape(-1, *image1_blocks.shape[2:])
    image2_blocks = split(image2, grid[0], grid[1])
    image2_blocks = image2_blocks.reshape(-1, *image2_blocks.shape[2:])

    block_flow = [*map(lambda x: get_displacements(*x), zip(image1_blocks, image2_blocks))]

    # we get (None, None) from get_displacement if there is no good feature to track.
    # Replace these with 0 vectors. Basically pretend that the block has a 0
    # magnitude displacement.
    block_flow = [tmp if tmp[0] is not None else ([(0, 0)], [(0, 0)]) for tmp in block_flow]

    block_displacements = map(lambda x: np.subtract(*x), block_flow)

    mean_block_dispalcements = np.array(
        [*map(lambda d: np.mean(d, axis=0), block_displacements)])

    mean_block_dispalcements = np.nan_to_num(mean_block_dispalcements)
    return mean_block_dispalcements.reshape(*grid, 2).astype(np.int)


def get_grid_centres(canvas_size, grid):
    """Calculate a grid block centres on a given canvas size.

    Note that canvas size must me divisible by grid.

    Parameters
    ----------
    canvas_size : Tuple(int),
        canvas_size (width, height)
    grid :
        grid (ncols, nrows)

    Return
    ------
        A numpy array of size (ncols, nrows, 2) where a 2D coordinate is assigned to 
        each grid block representing its centre. 
    """
    assert canvas_size[0] % grid[0] == 0, "{} rows is not divisble by {}".format(
        canvas_size[0], grid[0]
    )
    assert canvas_size[1] % grid[1] == 0, "{} cols is not divisble by {}".format(
        canvas_size[1], grid[1]
    )
    block_width = canvas_size[0] // grid[0]
    block_height = canvas_size[1] // grid[1]
    x = np.linspace(0, canvas_size[0], grid[0], endpoint=False)
    y = np.linspace(0, canvas_size[1], grid[1], endpoint=False)
    x += block_width // 2
    y += block_height // 2
    return np.array(np.meshgrid(x, y)).T.reshape(grid[0], grid[1], 2).astype(np.int)


def main():
    args = parse_args()
    _, _, fps, _, h, w = get_video_info(args.video_file)
    fg = FrameGenerator(args.video_file, show_video_info=True, use_rgb=False)

    frame_iterator = iter(fg)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(join(args.output_dir, "cam_motion.mp4"), fourcc, fps, (w, h))

    centres = get_grid_centres((w, h), args.grid_size).reshape(-1, 2)

    p_frame = next(frame_iterator)
    for frame in tqdm(
            frame_iterator, desc="playing video", unit="frame", total=len(fg) - 1
    ):
        draw_frame = np.array(frame)
        gray_p_frame = cv2.cvtColor(p_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        displacements = get_grid_flow(gray_p_frame, gray_frame, args.grid_size)
        absolute_displacements = displacements.reshape(-1, 2) + centres
        p_frame = frame
        # Draw vectors.
        for abs_displacement, abs_centre in zip(absolute_displacements, centres):
            cv2.line(
                draw_frame, tuple(abs_centre), tuple(abs_displacement), (255, 0, 0), 2,
            )
            cv2.circle(draw_frame, tuple(abs_centre.astype(np.int)), 1, (0, 255, 0))
        out.write(draw_frame)

    out.release()


if __name__ == "__main__":
    main()
