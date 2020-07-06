import numpy as np
from tqdm import tqdm
import logging
from src.camera_motion import estimate_z_transition

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")


def segment_motion(optical_flow, threshold):
    origins = optical_flow[0]
    displacements = optical_flow[1]
    cumulated = None
    for origin, displacement in tqdm(zip(origins, displacements), desc="Segmenting", unit="frame"):
        if cumulated is None:
            cumulated = displacement-origin
        else:
            cumulated += displacement-origin
        magnitudes = np.sqrt(np.sum(np.power(cumulated, 2), axis=2))
        mean_magnitude = np.mean(magnitudes)
        if mean_magnitude > threshold:
            cumulated = None
        yield mean_magnitude > threshold
    pass


def segment_transition(optical_flow, threshold, smooth_factor=0.99):
    origins = optical_flow[0]
    displacements = optical_flow[1]
    smoothed_displacement = displacements[0]
    for origin, displacement in tqdm(zip(origins, displacements), desc="Segmenting", unit="frame"):
        smoothed_displacement = smooth_factor * smoothed_displacement + (1 - smooth_factor) * displacement
        z_transition = estimate_z_transition(origin, smoothed_displacement)
        yield z_transition > threshold
    pass
