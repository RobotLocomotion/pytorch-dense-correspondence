import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def pil_image_to_cv2(pil_image):
    """
    Converts a PIL.Image to cv2 format
    :param pil_image: RGB PIL Image
    :type pil_image: PIL.Image
    :return: a numpy array that cv2 likes
    :rtype: numpy array 
    """
    return np.array(pil_image)[:, :, ::-1].copy() # open and convert between BGR and RGB

def draw_correspondence_points_cv2(img, pixels):
    """
    Draws nice reticles on the img, at each of pixels

    :param img: a cv2 image (really, just a numpy array, for example the return of pil_image_to_cv2)
    :type img: numpy array
    :param pixels: a list of dicts where each dict contains "u", "v" keys
    :type pixels: list

    :return: the img with the pixel reticles drawn on it
    :rtype: a cv2 image
    """
    label_colors = [(255,0,0), (0,255,0), (0,0,255), (255,0,255), (0,125,125), (125,125,0), (200,255,50), (255, 125, 220), (10, 125, 255)]
    for index, pixel in enumerate(pixels):
        img = draw_reticle_cv2(img, int(pixel["u"]), int(pixel["v"]), label_colors[index])
    return img

def draw_reticle_cv2(img, u, v, label_color):
    """
    Draws a nice reticle at pixel location u, v.

    The reticle is part white, and part label_color

    :param img: a cv2 image (really, just a numpy array, for example the return of pil_image_to_cv2)
    :type img: numpy array
    :param u, v: u,v pixel position
    :type u, v: int, int
    :param label_color: b, g, r color
    :type label_color: tuple of 3 ints in range 0 to 255.  for example: (255,255,255) for white

    :return: the img with a new pixel reticle drawn on it
    :rtype: a cv2 image
    """
    white = (255,255,255)
    black = (0,0,0)
    cv2.circle(img,(u,v),10,label_color,1)
    cv2.circle(img,(u,v),11,white,1)
    cv2.circle(img,(u,v),12,label_color,1)
    cv2.line(img,(u,v+1),(u,v+3),white,1)
    cv2.line(img,(u+1,v),(u+3,v),white,1)
    cv2.line(img,(u,v-1),(u,v-3),white,1)
    cv2.line(img,(u-1,v),(u-3,v),white,1)
    return img