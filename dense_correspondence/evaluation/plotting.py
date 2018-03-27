import numpy as np
import matplotlib.pyplot as plt

def normalize_descriptor(res):
    """
    Normalizes the descriptor into RGB color space
    :param res: numpy.array
        Output of the network, per-pixel dense descriptor
    :return: numpy.array
        normalized descriptor
    """
    normed_res = res - np.min(res)
    normed_res = normed_res / np.max(normed_res)
    return normed_res

