import argparse
import logging
import pickle
import src.visualize as visualize
import os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation-file', '-s', type=str)
    parser.add_argument("--output-dir", "-o", type=str)
    args = parser.parse_args()
    return args

def render_report(segmentation_file, output_dir):
    segmentation = pickle.load(open(segmentation_file, "rb"))
    segmentation['Length'] = segmentation.apply(lambda x: x['End frame'] - x['Start frame'], axis=1)

    gpd_segmentation = segmentation.groupby("Type")

    df = pd.DataFrame(data={"Longest stationary segment length": gpd_segmentation.max()["Length"]["visit"],
                                "Shortest stationary segment length": gpd_segmentation.min()["Length"]["visit"],
                                "Average stationary segment length": gpd_segmentation.mean()["Length"]["visit"],
                                "Longest moving segment length": gpd_segmentation.max()["Length"]["transition"],
                                "Shortest moving segment length": gpd_segmentation.min()["Length"]["transition"],
                                "Average moving segment length":gpd_segmentation.mean()["Length"]["transition"],
                                }, index=[0])

    df.to_csv(os.path.join(output_dir, "segmentation_stat.csv"))
    x = [[0]*l if t == "visit" else [1]*l for l, t in zip(segmentation['Length'], segmentation['Type'])]
    x = [x2 for x1 in x for x2 in x1]
    df = pd.DataFrame(data={"x": x, "y": range(len(x))})
    visualize.plot(df, "y", "x", save_to= output_dir+"segmentation.svg")


if __name__ == "__main__":
    args = parseargs()
    render_report(**args.__dict__)
