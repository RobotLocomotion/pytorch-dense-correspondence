import sys
import os
import cv2
import numpy as np
from PIL import Image
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
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color

def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle on the image at the given (u,v) position

    :param img:
    :type img:
    :param u:
    :type u:
    :param v:
    :type v:
    :param label_color:
    :type label_color:
    :return:
    :rtype:
    """
    # cast to int
    u = int(u)
    v = int(v)

    white = (255, 255, 255)
    cv2.circle(img, (u, v), 10, label_color, 1)
    cv2.circle(img, (u, v), 11, white, 1)
    cv2.circle(img, (u, v), 12, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)

def draw_reticles(img,
                  u_vec,
                  v_vec,
                  label_color=None,
                  label_color_list=None,
                  ):
    # draws multiple reticles
    n = len(u_vec)
    for i in range(n):
        u = u_vec[i]
        v = v_vec[i]

        color = None
        if label_color is not None:
            color = label_color
        else:
            color = label_color_list[i]

        draw_reticle(img, u, v, color)


def colormap_from_heatmap(h, # numpy array [H, W]
                          normalize=False, # whether or not to normalize to [0,1]
                          ): # np.ndarray [H, W, 3] 'rgb' ordering

    h_255 = None
    if normalize:
        h_255 = np.uint8(255 * h / np.max(h))
    else:
        h_255 = np.uint8(255 * h)

    colormap = cv2.applyColorMap(h_255, cv2.COLORMAP_JET)
    colormap_rgb = np.zeros_like(colormap)
    colormap_rgb[:, :, 0] = colormap[:, :, 2]
    colormap_rgb[:, :, 2] = colormap[:, :, 0]
    return colormap_rgb


