import numpy as np
from util.quat_utils import quat2angle

def l2_error(p1, p2):
    return np.linalg.norm(p1 - p2)

def quat_error(q1, q2):
    return np.rad2deg(quat2angle(q1, q2))

def avg_landmark_error(l1, l2):
    n_l = np.shape(l1)[0] // 3
    return 3.0 * np.linalg.norm(l1 - l2) / n_l
