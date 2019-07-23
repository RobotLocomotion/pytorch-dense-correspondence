"""

The purpose of this file is to perform data augmentation for images
and lists of pixel positions in them.

- For operations on the images, we can use functions optimized 
for image data.

- For operations on a list of pixel indices, we need a matching
implementation.

"""

from PIL import Image, ImageOps
import numpy as np
import random
import torch
import imgaug.augmenters as iaa
import imgaug as ia

def random_image_and_indices_mutation(images, uv_pixel_positions):
    """
    This function takes a list of images and a list of pixel positions in the image, 
    and picks some subset of available mutations.

    :param images: a list of images (for example the rgb, depth, and mask) for which the 
                        **same** mutation will be applied
    :type  images: list of PIL.image.image

    :param uv_pixel_positions: pixel locations (u, v) in the image. 
    	See doc/coordinate_conventions.md for definition of (u, v)

    :type  uv_pixel_positions: a tuple of torch Tensors, each of length n, i.e:

    	(u_pixel_positions, v_pixel_positions)

    	Where each of the elements of the tuple are torch Tensors of length n

    	Note: aim is to support both torch.LongTensor and torch.FloatTensor,
    	      and return the mutated_uv_pixel_positions with same type

    :return mutated_image_list, mutated_uv_pixel_positions
    	:rtype: list of PIL.image.image, tuple of torch Tensors

    """

    # Current augmentation is:
    # 50% do nothing
    # 50% rotate the image 180 degrees (by applying flip vertical then flip horizontal) 

    if random.random() < 0.5:
        return images, uv_pixel_positions

    else:
        mutated_images, mutated_uv_pixel_positions = flip_vertical(images, uv_pixel_positions)
        mutated_images, mutated_uv_pixel_positions = flip_horizontal(mutated_images, mutated_uv_pixel_positions)

        return mutated_images, mutated_uv_pixel_positions


def flip_vertical(images, uv_pixel_positions):
    """
    Fip the images and the pixel positions vertically (flip up/down)

    See random_image_and_indices_mutation() for documentation of args and return types.

    """
    mutated_images = [ImageOps.flip(image) for image in images]
    v_pixel_positions = uv_pixel_positions[1]
    mutated_v_pixel_positions = (image.height-1) - v_pixel_positions
    mutated_uv_pixel_positions = (uv_pixel_positions[0], mutated_v_pixel_positions)
    return mutated_images, mutated_uv_pixel_positions

def flip_horizontal(images, uv_pixel_positions):
    """
    Randomly flip the image and the pixel positions horizontall (flip left/right)

    See random_image_and_indices_mutation() for documentation of args and return types.

    """

    mutated_images = [ImageOps.mirror(image) for image in images]
    u_pixel_positions = uv_pixel_positions[0]
    mutated_u_pixel_positions = (image.width-1) - u_pixel_positions
    mutated_uv_pixel_positions = (mutated_u_pixel_positions, uv_pixel_positions[1])
    return mutated_images, mutated_uv_pixel_positions

def random_domain_randomize_background(image_rgb, image_mask):
    """
    Ranomly call domain_randomize_background
    """
    if random.random() < 0.5:
        return image_rgb
    else:
        return domain_randomize_background(image_rgb, image_mask)


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
    return Image.fromarray(domain_randomized_image_rgb)

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


def merge_images_with_occlusions(image_a, image_b, mask_a, mask_b, matches_pair_a, matches_pair_b):
    """
    This function will take image_a and image_b and "merge" them.

    It will do this by:
    - randomly selecting either image_a or image_b to be the background
    - using the mask for the image that is not the background, it will put the other image on top.
    - critically there are two separate sets of matches, one is associated with image_a and some other image,
        and the other is associated with image_b and some other image.
    - both of these sets of matches must be pruned for any occlusions that occur.

    :param image_a, image_b: the two images to merge
    :type image_a, image_b: each a PIL.image.image
    :param mask_a, mask_b: the masks for these images
    :type mask_a, mask_b: each a PIL.image.image
    :param matches_a, matches_b:
    :type matches_a, mathces_b: each a tuple of torch Tensors, each of length n, i.e:

        (u_pixel_positions, v_pixel_positions)

        Where each of the elements of the tuple are torch Tensors of length n

        Note: only support torch.LongTensors

    :return: merged image, merged_mask, pruned_matches_a, pruned_associated_matches_a, pruned_matches_b, pruned_associated_matches_b
    :rtype: PIL.image.image, numpy array, rest are same types as matches_a and matches_b

    """

    if random.random() < 0.5:
        foreground = "B"
        background_image, background_mask, background_matches_pair = image_a, mask_a, matches_pair_a
        foreground_image, foreground_mask, foreground_matches_pair = image_b, mask_b, matches_pair_b
    else:
        foreground = "A"
        background_image, background_mask, background_matches_pair = image_b, mask_b, matches_pair_b
        foreground_image, foreground_mask, foreground_matches_pair = image_a, mask_a, matches_pair_a

    # First, mask the foreground rgb image
    foreground_image_numpy = np.asarray(foreground_image)
    foreground_mask_numpy  = np.asarray(foreground_mask)
    three_channel_mask = np.zeros_like(foreground_image_numpy)
    three_channel_mask[:,:,0] = three_channel_mask[:,:,1] = three_channel_mask[:,:,2] = foreground_mask
    foreground_image_numpy = foreground_image_numpy * three_channel_mask

    # Next, zero out this portion in the background image
    background_image_numpy = np.asarray(background_image)
    three_channel_mask_complement = np.ones_like(three_channel_mask) - three_channel_mask
    background_image_numpy = three_channel_mask_complement * background_image_numpy

    # Finally, merge these two images
    merged_image_numpy = foreground_image_numpy + background_image_numpy

    # Prune occluded matches
    background_matches_pair = prune_matches_if_occluded(foreground_mask_numpy, background_matches_pair)
 
    if foreground == "A":
        matches_a            = foreground_matches_pair[0]
        associated_matches_a = foreground_matches_pair[1]
        matches_b            = background_matches_pair[0]
        associated_matches_b = background_matches_pair[1]
    elif foreground == "B":
        matches_a            = background_matches_pair[0]
        associated_matches_a = background_matches_pair[1]
        matches_b            = foreground_matches_pair[0]
        associated_matches_b = foreground_matches_pair[1]
    else:
        raise ValueError("Should not be here?")

    merged_masked_numpy = foreground_mask_numpy + np.asarray(background_mask)
    merged_masked_numpy = merged_masked_numpy.clip(0,1) # in future, could preserve identities of masks
    return Image.fromarray(merged_image_numpy), merged_masked_numpy, matches_a, associated_matches_a, matches_b, associated_matches_b


def prune_matches_if_occluded(foreground_mask_numpy, background_matches_pair):
    """
    Checks if any of the matches have been occluded.

    If yes, prunes them from the list of matches.

    NOTE:
    - background_matches is a tuple
    - the first element of the tuple HAS to be the one that we are actually checking for occlusions
    - the second element of the tuple must also get pruned

    :param foreground_mask_numpy: The mask of the foreground image
    :type foreground_mask_numpy: numpy 2d array of shape (H,W)
    :param background_matches: a tuple of torch Tensors, each of length n, i.e:

        (u_pixel_positions, v_pixel_positions)

        Where each of the elements of the tuple are torch Tensors of length n

        Note: only support torch.LongTensors
    """

    background_matches_a = background_matches_pair[0] 
    background_matches_b = background_matches_pair[1]

    idxs_to_keep  = []
    
    # this is slow but works
    for i in range(len(background_matches_a[0])):
        u = background_matches_a[0][i]
        v = background_matches_a[1][i]

        if foreground_mask_numpy[v,u] == 0:
            idxs_to_keep.append(i)

    if len(idxs_to_keep) == 0:
        return (None, None)

    idxs_to_keep = torch.LongTensor(idxs_to_keep)
    background_matches_a = (torch.index_select(background_matches_a[0], 0, idxs_to_keep), torch.index_select(background_matches_a[1], 0, idxs_to_keep))
    background_matches_b = (torch.index_select(background_matches_b[0], 0, idxs_to_keep), torch.index_select(background_matches_b[1], 0, idxs_to_keep))

    return (background_matches_a, background_matches_b)

def merge_matches(matches_one, matches_two):
    """
    :param matches_one, matches_two: each a tuple of torch Tensors, each of length n, i.e:

        (u_pixel_positions, v_pixel_positions)

        Where each of the elements of the tuple are torch Tensors of length n

        Note: only support torch.LongTensors
    """
    concatenated_u = torch.cat((matches_one[0], matches_two[0]))
    concatenated_v = torch.cat((matches_one[1], matches_two[1]))
    return (concatenated_u, concatenated_v)


def affine_augmentation(images, uv_pixel_positions):
    """
    images: list of PIL images
    uv_pixel_positions: list of tuples of torch tensors
    """
    DEBUG = False

    all_uv_one_tensor = torch.stack((uv_pixel_positions[0][0],uv_pixel_positions[0][1])).permute(1,0)
    all_uv_one_numpy = all_uv_one_tensor.numpy()
    
    images_numpy = [np.asarray(i) for i in images]

    rand_int = random.randint(0,1e9)

    # HACK FOR VIS
    #some_uv_one_numpy = all_uv_one_numpy[:100,:100]
    #all_uv_one_numpy = some_uv_one_numpy
    #print images_numpy[0].shape
    
    seq = iaa.Sequential([
        #iaa.Multiply((0.8, 1.2)), # change brightness, doesn't affect keypoints
        #iaa.AddToHueAndSaturation((-50, 50)),
        iaa.Affine(
            rotate=(-180,180),
            scale=(0.8, 1.2),
            shear=(-10,10)
        ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
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
            all_uv_one_tensor = torch.stack((addition_pixel_positions[0],addition_pixel_positions[1])).permute(1,0)
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

        import matplotlib.pyplot as plt
        plt.imshow(images_numpy[1], vmin=500.0, vmax=4000.0)
        plt.show()

        plt.imshow(images_aug[1], vmin=500.0, vmax=4000.0)
        plt.show()

        plt.imshow(images_numpy[2])
        plt.show()

        plt.imshow(images_aug[2])
        plt.show()

    images_aug_pil = [Image.fromarray(img) for img in images_aug]

    uv_pixel_positions_aug = []
    for kp_aug in kps_aug:
        u = torch.from_numpy(kp_aug.to_xy_array()[:,0])
        v = torch.from_numpy(kp_aug.to_xy_array()[:,1])
        uv_pixel_positions_aug.append((u,v))

    return images_aug_pil, uv_pixel_positions_aug




    

