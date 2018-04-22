import numpy as np
import matplotlib.pyplot as plt

def normalize_descriptor(res, stats=None):
    """
    Normalizes the descriptor into RGB color space
    :param res: numpy.array [H,W,D]
        Output of the network, per-pixel dense descriptor
    :param stats: dict, with fields ['min', 'max', 'mean'], which are used to normalize descriptor
    :return: numpy.array
        normalized descriptor
    """

    if stats is None:
        res_min = res.min()
        res_max = res.max()
    else:
        res_min = np.array(stats['min'])
        res_max = np.array(stats['max'])

    eps = 1e-10
    scale = (res_max - res_min) + eps
    normed_res = (res - res_min) / scale
    return normed_res

def normalize_descriptor_pair(res_a, res_b):
    """
    Normalizes the descriptor into RGB color space
    :param res_a, res_b: numpy.array [H,W,D]
        Two outputs of the network, per-pixel dense descriptor
    :return: numpy.array, numpy.array
        normalized descriptors
    """
    both_min = min(np.min(res_a), np.min(res_b))
    normed_res_a = res_a - both_min
    normed_res_b = res_b - both_min

    both_max = max(np.max(normed_res_a), np.max(normed_res_b))
    normed_res_a = normed_res_a / both_max
    normed_res_b = normed_res_b / both_max

    return normed_res_a, normed_res_b