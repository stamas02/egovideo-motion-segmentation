import cv2
import numpy as np
import scipy.optimize as optimize


def _z_translation_fn(data, parameter, focal_length):
    x = data[:, :, 0]
    y = data[:, :, 1]
    f = focal_length
    new_data = np.array(data)
    new_data[:, :, 0] = f * (np.arctan(x / f)) * (1 + (x ** 2 / f ** 2)) * parameter
    new_data[:, :, 1] = f * (np.arctan(y / f)) * (1 + (y ** 2 / f ** 2)) * parameter

    return new_data.flatten()


def estimate_z_transition(origins, displacements, focal_length=150):
    """
    Estimates amount of camera transition on the z axis based on optical flow.

    Returns
    -------
        - The estimated amount of transition on the z axis of the camera.
        - The perfect optical flow based on the estimation
    """

    # x = np.linspace(-1.0, 1.0, optical_flow.shape[0])
    # y = np.linspace(-1.0, 1.0, optical_flow.shape[1])
    # xv, yv = np.meshgrid(y, x)
    # origins = np.stack((xv,yv),-1)

    cx = (np.max(origins[:, :, 0]) + np.min(origins[:, :, 0])) // 2
    cy = (np.max(origins[:, :, 1]) + np.min(origins[:, :, 1])) // 2

    origins = np.array(origins, dtype=np.float)
    origins[:, :, 0] -= cx
    origins[:, :, 1] -= cy
    origins[:, :, 0] = origins[:, :, 0] / cx * 2
    origins[:, :, 1] = origins[:, :, 1] / cy * 2

    displacements = np.array(displacements, dtype=np.float)
    displacements[:, :, 0] -= cx
    displacements[:, :, 1] -= cy
    displacements[:, :, 0] = displacements[:, :, 0] / cx * 2
    displacements[:, :, 1] = displacements[:, :, 1] / cy * 2

    func = lambda data, parameter: _z_translation_fn(data, parameter, focal_length)

    # abs_optical_flow = optical_flow+origins

    param, pcov = optimize.curve_fit(func, origins, displacements.flatten())
    return param[0]
