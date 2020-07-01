import random
import numpy as np
from PIL import Image, ImageOps
import torch


import imgaug as ia
import imgaug.augmenters as iaa

class ImageAugmentation(object):
    """
    Helper class for data augmentation
    """

    def __init__(self, enabled):
        self.set_enabled(enabled)
        self._seq = None # augmentation sequence, iaa.Sequential

    def set_enabled(self, val):
        self._enabled = val

    def make_default(self):
        aug_list = []
        aug_list.append(ImageAugmentation.default_hue_and_brightness())
        aug_list.append(ImageAugmentation.default_contrast())
        # aug_list.append(ImageAugmentation.default_change_color_temperature())
        # aug_list.append(ImageAugmentation.default_grayscale())
        aug_list.append(ImageAugmentation.default_colorspace())

        seq = iaa.Sequential(aug_list, random_order=True)
        self._seq = seq

    def augment_image(self,
                      image_data, # dict with image information, modifies it in place
                      ):

        if not self._enabled:
            # it's a noop in this case
            return image_data

        # for now we only support color augmentations on the RGB image
        rgb = image_data['rgb']
        rgb_aug = self._seq(images=[rgb])[0]

        # domain randomize the background
        if random.random() < 0.5:
            rgb_aug = domain_randomize_background(rgb_aug, image_data['mask'])

        image_data['rgb'] = rgb_aug
        image_data['rgb_original'] = rgb

        return image_data


    def affine_augmentation(self,
                            data, # dict, the output of correspondence_data
                            image_key, # 'data_a' or 'data_b'
                            uv_key, # 'uv_a' or 'uv_b'
                            debug=False,
                            ):
        """
        Performs affine augmentation on rgb, mask, depth and keypoints

        Note: if you are using the keypoints to project to 3D you should disable this
        The depth values at the pixel will make sense, but if you project the pointcloud to
        3D then of course it is no longer meaningful due to perspective projection through
        the camera intrinsic matrix
        """
        if debug:
            print("Performing AFFINE augmentation")


        if not self._enabled:
            # it's a noop in this case
            return data

        # simple sanity checks
        if image_key == 'data_a':
            assert uv_key == 'uv_a'

        if image_key == 'data_b':
            assert uv_key == 'uv_b'

        image_types = ['rgb', 'mask', 'depth', 'depth_int16']
        image_types = [x for x in image_types if x in data[image_key]]

        if debug:
            print("image_types", image_types)

        image_list = []
        for image_type in image_types:
            image_list.append(data[image_key][image_type])


        uv_types = ['matches', 'non_matches', 'background_non_matches', 'masked_non_matches']
        uv_types = [x for x in uv_types if x in data]


        if debug:
            print("uv_types", uv_types)

        uv_pixel_positions_list = []
        for uv_type in uv_types:
            uv_pixel_positions_list.append(data[uv_type][uv_key])

        images_aug, uv_pixel_positions_aug = affine_augmentation(image_list, uv_pixel_positions_list, DEBUG=debug)

        # store the augmented images in the dict
        for idx, image_type in enumerate(image_types):
            data[image_key][image_type] = images_aug[idx]

        # update uv_positions and do FOV check
        for idx, uv_type in enumerate(uv_types):
            uv_tuple = uv_pixel_positions_aug[idx] # this is tuple, need to convert to tensor
            # print("type(uv_tuple)", type(uv_tuple))
            # print("uv_tuple[0].shape", uv_tuple[0].shape)

            # print("uv.shape", uv.shape)
            # print("uv.dtype", uv.dtype)

            # [2, N] tensor, currently its torch.float
            # need to convert it to torch.long so it can
            # be used to index into image
            uv = torch.round(torch.stack(uv_tuple)).type(torch.long)


            data[uv_type][uv_key] = uv
            valid = data[uv_type]['valid']
            valid_FOV = check_FOV(uv, image_shape=image_list[0].shape)

            # set the ones outside the valid FOV to [0, 0] so they don't
            # cause indexing errors later
            outside_FOV = ~valid_FOV
            uv[0][outside_FOV] = 0
            uv[1][outside_FOV] = 0

            data[uv_type]['valid'] = valid & valid_FOV

        return data

    def augment(self, data):
        """
        data is the output of "correspondence_data"

        keys include ['data_a', 'data_b', 'matches']
        """

        if not self._enabled:
            # it's a noop in this case
            return data

        key_list = [['data_a', 'uv_a'], ['data_b', 'uv_b']]

        for [image_key, uv_key] in key_list:
            data[image_key] = self.augment_image(data[image_key])
            self.affine_augmentation(data=data,
                                     image_key=image_key,
                                     uv_key=uv_key)



    @staticmethod
    def default_hue_and_brightness():
        seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.2)),  # change brightness, doesn't affect keypoints
            iaa.AddToHueAndSaturation((-50, 50)),
        ])

        seq = iaa.Sometimes(0.5, seq)

        return seq

    @staticmethod
    def default_contrast():
        aug = iaa.LinearContrast((0.75, 1.5))
        seq = iaa.Sometimes(0.5, aug)

        return seq

    @staticmethod
    def default_change_color_temperature():
        aug = iaa.ChangeColorTemperature((1100, 10000))
        seq = iaa.Sometimes(0.5, aug)
        return seq


    @staticmethod
    def default_grayscale():
        aug = iaa.Grayscale(alpha=(0.0, 1.0))
        seq = iaa.Sometimes(0.5, aug)
        return seq


    @staticmethod
    def default_colorspace():
        aug = iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((50, 100))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        ])

        seq = iaa.Sometimes(0.5, aug)
        return seq


def domain_randomize_background(image_rgb, image_mask):
    """
    This function applies domain randomization to the non-masked part of the image.

    :param image_rgb: rgb image for which the non-masked parts of the image will
                        be domain randomized
    :type  image_rgb: PIL.image.image

    :param image_mask: mask of part of image to be left alone, all else will be domain randomized
    :type image_mask: PIL.image.image

    :return domain_randomized_image_rgb:
    :rtype: PIL.image.image
    """
    # First, mask the rgb image
    image_rgb_numpy = np.asarray(image_rgb)
    image_mask_numpy = np.asarray(image_mask)
    three_channel_mask = np.zeros_like(image_rgb_numpy)
    three_channel_mask[:,:,0] = three_channel_mask[:,:,1] = three_channel_mask[:,:,2] = image_mask
    image_rgb_numpy = image_rgb_numpy * three_channel_mask

    # Next, domain randomize all non-masked parts of image
    three_channel_mask_complement = np.ones_like(three_channel_mask) - three_channel_mask
    random_rgb_image = get_random_image(image_rgb_numpy.shape)
    random_rgb_background = three_channel_mask_complement * random_rgb_image

    domain_randomized_image_rgb = image_rgb_numpy + random_rgb_background
    return domain_randomized_image_rgb

def get_random_image(shape):
    """
    Expects something like shape=(480,640,3)

    :param shape: tuple of shape for numpy array, for example from my_array.shape
    :type shape: tuple of ints

    :return random_image:
    :rtype: np.ndarray
    """
    if random.random() < 0.5:
        rand_image = get_random_solid_color_image(shape)
    else:
        rgb1 = get_random_solid_color_image(shape)
        rgb2 = get_random_solid_color_image(shape)
        vertical = bool(np.random.uniform() > 0.5)
        rand_image = get_gradient_image(rgb1, rgb2, vertical=vertical)

    if random.random() < 0.5:
        return rand_image
    else:
        return add_noise(rand_image)

def get_random_rgb():
    """
    :return random rgb colors, each in range 0 to 255, for example [13, 25, 255]
    :rtype: numpy array with dtype=np.uint8
    """
    return np.array(np.random.uniform(size=3) * 255, dtype=np.uint8)

def get_random_solid_color_image(shape):
    """
    Expects something like shape=(480,640,3)

    :return random solid color image:
    :rtype: numpy array of specificed shape, with dtype=np.uint8
    """
    return np.ones(shape,dtype=np.uint8)*get_random_rgb()

def get_random_entire_image(shape, max_pixel_uint8):
    """
    Expects something like shape=(480,640,3)

    Returns an array of that shape, with values in range [0..max_pixel_uint8)

    :param max_pixel_uint8: maximum value in the image
    :type max_pixel_uint8: int

    :return random solid color image:
    :rtype: numpy array of specificed shape, with dtype=np.uint8
    """
    return np.array(np.random.uniform(size=shape) * max_pixel_uint8, dtype=np.uint8)

# this gradient code roughly taken from:
# https://github.com/openai/mujoco-py/blob/master/mujoco_py/modder.py
def get_gradient_image(rgb1, rgb2, vertical):
    """
    Interpolates between two images rgb1 and rgb2

    :param rgb1, rgb2: two numpy arrays of shape (H,W,3)

    :return interpolated image:
    :rtype: same as rgb1 and rgb2
    """
    bitmap = np.zeros_like(rgb1)
    h, w = rgb1.shape[0], rgb1.shape[1]
    if vertical:
        p = np.tile(np.linspace(0, 1, h)[:, None], (1, w))
    else:
        p = np.tile(np.linspace(0, 1, w), (h, 1))

    for i in range(3):
        bitmap[:, :, i] = rgb2[:, :, i] * p + rgb1[:, :, i] * (1.0 - p)

    return bitmap

def add_noise(rgb_image):
    """
    Adds noise, and subtracts noise to the rgb_image

    :param rgb_image: image to which noise will be added
    :type rgb_image: numpy array of shape (H,W,3)

    :return image with noise:
    :rtype: same as rgb_image

    ## Note: do not need to clamp, since uint8 will just overflow -- not bad
    """
    max_noise_to_add_or_subtract = 50
    return rgb_image + get_random_entire_image(rgb_image.shape, max_noise_to_add_or_subtract) - get_random_entire_image(rgb_image.shape, max_noise_to_add_or_subtract)


def affine_augmentation(images,
                        uv_pixel_positions,
                        DEBUG=False):
    """
    images: list of PIL images (or numpy arrays)
    uv_pixel_positions: list of tuples of torch tensors


    return: tuple (images_aug, uv_pixel_positions_aug)
    """

    # what shape is this . . . .?
    all_uv_one_tensor = torch.stack((uv_pixel_positions[0][0], uv_pixel_positions[0][1])).permute(1, 0)
    all_uv_one_numpy = all_uv_one_tensor.numpy()

    images_numpy = [np.asarray(i) for i in images]

    rand_int = random.randint(0, 1e9)

    # HACK FOR VIS
    # some_uv_one_numpy = all_uv_one_numpy[:100,:100]
    # all_uv_one_numpy = some_uv_one_numpy
    # print images_numpy[0].shape

    seq = iaa.Sequential([
        # iaa.Multiply((0.8, 1.2)), # change brightness, doesn't affect keypoints
        # iaa.AddToHueAndSaturation((-50, 50)),
        iaa.Affine(
            rotate=(-180, 180),
            scale=(0.8, 1.2),
            shear=(-10, 10)
        )  # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    ])

    images_aug = []
    kps_aug = []

    for image_numpy in images_numpy:

        kp = ia.KeypointsOnImage.from_xy_array(all_uv_one_numpy, shape=image_numpy.shape)
        ia.seed(rand_int)

        image_aug, kp_aug = seq(image=image_numpy, keypoints=kp)

        if DEBUG:

            if len(image_numpy.shape) == 3:
                image_numpy_vis = image_numpy
                image_aug_vis = image_aug
            else:
                image_numpy_vis = np.expand_dims(image_numpy, axis=2)
                image_numpy_vis = np.repeat(image_numpy_vis, 3, axis=2)

                image_aug_vis = np.expand_dims(image_aug, axis=2)
                image_aug_vis = np.repeat(image_aug_vis, 3, axis=2)

            ia.imshow(
                np.hstack([
                    kp.draw_on_image(image_numpy_vis, size=7),
                    kp_aug.draw_on_image(image_aug_vis, size=7)
                ])
            )

        images_aug.append(image_aug)
        if len(kps_aug) == 0:
            kps_aug.append(kp_aug)

    if len(uv_pixel_positions) > 1:
        for addition_pixel_positions in uv_pixel_positions[1:]:
            all_uv_one_tensor = torch.stack((addition_pixel_positions[0], addition_pixel_positions[1])).permute(1, 0)
            all_uv_one_numpy = all_uv_one_tensor.numpy()

            kp = ia.KeypointsOnImage.from_xy_array(all_uv_one_numpy, shape=images_numpy[0].shape)
            ia.seed(rand_int)

            image_aug, kp_aug = seq(image=images_numpy[0], keypoints=kp)

            if DEBUG:
                ia.imshow(
                    np.hstack([
                        kp.draw_on_image(images_numpy[0], size=7),
                        kp_aug.draw_on_image(image_aug, size=7)
                    ])
                )

            kps_aug.append(kp_aug)

    if DEBUG:
        if len(images_numpy) > 1:

            import matplotlib.pyplot as plt
            plt.imshow(images_numpy[1], vmin=500.0, vmax=4000.0)
            plt.show()

            plt.imshow(images_aug[1], vmin=500.0, vmax=4000.0)
            plt.show()

            plt.imshow(images_numpy[2])
            plt.show()

            plt.imshow(images_aug[2])
            plt.show()


    uv_pixel_positions_aug = []
    for kp_aug in kps_aug:
        u = torch.from_numpy(kp_aug.to_xy_array()[:, 0])
        v = torch.from_numpy(kp_aug.to_xy_array()[:, 1])
        uv_pixel_positions_aug.append((u, v))

    return images_aug, uv_pixel_positions_aug


def check_FOV(uv,  # tuple (u_tensor, v_tensor) or tensor with shape [2, N]
              image_shape,  # list/tuple [H,W,C]
              ): # tensor (bool) [N,] whether in FOV or not

    u_tensor = uv[0]
    v_tensor = uv[1]

    assert u_tensor.shape == v_tensor.shape

    H = image_shape[0]
    W = image_shape[1]

    eps = 1e-3
    u_valid = (u_tensor > eps) & (u_tensor < (W - eps))
    v_valid = (v_tensor > eps) & (v_tensor < (H - eps))
    # v_valid = torch.logical_and(uv_tensor[1] > eps, uv_tensor[1] < (H - eps))
    # valid = torch.logical_and(u_valid, v_valid)
    valid = u_valid & v_valid

    return valid
