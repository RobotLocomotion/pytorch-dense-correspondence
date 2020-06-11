"""
Image processing utilities
"""

import numpy as np
import cv2

def pil_image_to_cv2(pil_image):
    """
    This correctly converts between BGR and RGB

    :param pil_image: rgb image in PIL image format
    :type pil_image: PIL.image
    :return: cv2 image in bgr format
    :rtype:
    """

    rgb_image = np.array(pil_image).copy()
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


