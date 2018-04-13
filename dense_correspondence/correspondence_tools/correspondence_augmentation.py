"""

The purpose of this file is to perform data augmentation for images
and lists of pixel positions in them.

- For operations on the images, we can use functions optimized 
for image data.

- For operations on a list of pixel indices, we need a matching
implementation.

"""

from PIL import Image, ImageOps
import random


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
    mutated_images, mutated_uv_pixel_positions = random_flip_vertical(images, uv_pixel_positions)
    mutated_images, mutated_uv_pixel_positions = random_flip_horizontal(mutated_images, mutated_uv_pixel_positions)
    return mutated_images, mutated_uv_pixel_positions


def random_flip_vertical(images, uv_pixel_positions):
    """
    Randomly flip the images and the pixel positions vertically (flip up/down)

    See random_image_and_indices_mutation() for documentation of args and return types.

    """

    if random.random() < 0.5:
        return images, uv_pixel_positions  # Randomly do not apply

    print "Flip vertically"
    mutated_images = [ImageOps.flip(image) for image in images]
    v_pixel_positions = uv_pixel_positions[1]
    mutated_v_pixel_positions = image.height - v_pixel_positions
    mutated_uv_pixel_positions = (uv_pixel_positions[0], mutated_v_pixel_positions)
    return mutated_images, mutated_uv_pixel_positions

def random_flip_horizontal(images, uv_pixel_positions):
    """
    Randomly flip the image and the pixel positions horizontall (flip left/right)

    See random_image_and_indices_mutation() for documentation of args and return types.

    """

    if random.random() < 0.5:
        return images, uv_pixel_positions  # Randomly do not apply

    print "Flip left and right"
    mutated_images = [ImageOps.mirror(image) for image in images]
    u_pixel_positions = uv_pixel_positions[0]
    mutated_u_pixel_positions = image.width - u_pixel_positions
    mutated_uv_pixel_positions = (mutated_u_pixel_positions, uv_pixel_positions[1])
    return mutated_images, mutated_uv_pixel_positions