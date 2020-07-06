import cv2
import numpy as np

k_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

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
        return np.array([[0, 0]]), np.array([[0, 0]])

    p2, st, err = cv2.calcOpticalFlowPyrLK(image1, image2, p1, None, **k_params)
    if p2 is None:
        return np.array([[0, 0]]), np.array([[0, 0]])

    if not np.any(st == 1):
        return np.array([[0, 0]]), np.array([[0, 0]])

    origin = p1[st == 1].reshape(-1, 2)
    dispacement = p2[st == 1].reshape(-1, 2)
    return origin, dispacement