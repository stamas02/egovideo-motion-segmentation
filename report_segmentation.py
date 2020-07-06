import argparse
import logging
import pickle
import src.visualize as visualize
import os
import numpy as np
import pandas

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation-file', '-s', type=str)
    parser.add_argument("--output-dir", "-o", type=str)
    args = parser.parse_args()
    return args

def render_report(segmentation_file, output_dir):
    segmentation = pickle.load(open(segmentation_file, "rb"))


    stationary = np.array(segments["stationary"])[:,1] - np.array(segments["stationary"])[:,0]
    moving = np.array(segments["moving"])[:,1] - np.array(segments["moving"])[:,0]
    df = pandas.DataFrame(data={"Longest stationary segment length": np.max(stationary),
                                "Shortest stationary segment length": np.min(stationary),
                                "Average stationary segment length": np.mean(stationary),
                                "Longest moving segment length": np.max(moving),
                                "Shortest moving segment length": np.min(moving),
                                "Average moving segment length": np.mean(moving),
                                }, index=[0])

    df.to_csv(os.path.join(output_dir, "segmentation_stat.csv"))
    visualize.render_line(classification, "Frame", "Magnitude", "Name", os.path.join(output_dir, "segmentation_plot"))


if __name__ == "__main__":
    args = parseargs()
    render_report(**args.__dict__)
