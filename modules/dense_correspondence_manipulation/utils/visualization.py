import sys
import os
import cv2
import numpy as np
import copy


def compute_gaussian_kernel_heatmap_from_norm_diffs(norm_diffs, variance):
    """
    Computes and RGB heatmap from norm diffs
    :param norm_diffs: distances in descriptor space to a given keypoint
    :type norm_diffs: numpy array of shape [H,W]
    :param variance: the variance of the kernel
    :type variance:
    :return: RGB image [H,W,3]
    :rtype:
    """

    """
    Computes an RGB heatmap from the norm_diffs
    :param norm_diffs:
    :type norm_diffs:
    :return:
    :rtype:
    """

    heatmap = np.copy(norm_diffs)

    heatmap = np.exp(-heatmap / variance)  # these are now in [0,1]
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    print "heatmap dtype", heatmap.dtype
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color