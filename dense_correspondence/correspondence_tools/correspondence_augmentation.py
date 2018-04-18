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

def domain_randomize_mask_complement(image_rgb, image_mask):
    """
    This function applies domain randomization to the non-masked part of the image.

    :param image_rgb: rgb image for which the non-masked parts of the image will 
                        be domain randomized
    :type  image_rgb: PIL.image.image

    :return domain_randomized_image_rgb:
    :rtype: PIL.image.image

    """
    # First, mask the rgb image
    image_rgb_numpy = np.asarray(image_rgb)
    image_mask_numpy = np.asarray(image_mask)
    three_channel_mask = np.zeros_like(image_rgb_numpy)
    three_channel_mask[:,:,0] = three_channel_mask[:,:,1] = three_channel_mask[:,:,2] = image_mask
    image_rgb_numpy = image_rgb_numpy * three_channel_mask
    print np.max(image_rgb_numpy)

    # Next, domain randomize all non-masked parts of image
    three_channel_mask_complement = np.ones_like(three_channel_mask) - three_channel_mask
    random_rgb_image = np.array(np.random.uniform(size=3) * 255, dtype=np.uint8)
    random_rgb_background = three_channel_mask_complement * random_rgb_image

    domain_randomized_image_rgb = image_rgb_numpy + random_rgb_background

    return Image.fromarray(domain_randomized_image_rgb)
