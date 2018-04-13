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


def random_image_and_indices_mutation(image, uv_pixel_positions):
    """
    This function takes an image and a list of pixel positions in the image, 
    and picks some subset of available mutations.

    :param image: image for which to augment
    :type  image: PIL.image.image

    :param uv_pixel_positions: pixel locations (u, v) in the image. 
    	See doc/coordinate_conventions.md for definition of (u, v)

    :type  uv_pixel_positions: a tuple of torch Tensors, each of length n, i.e:

    	(u_pixel_positions, v_pixel_positions)

    	Where each of the elements of the tuple are torch Tensors of length n

    	Note: aim is to support both torch.LongTensor and torch.FloatTensor,
    	      and return the mutated_uv_pixel_positions with same type

    :return mutated_image, mutated_uv_pixel_positions
    	:rtype: PIL.image.image, tuple of torch Tensors

    """
    mutated_image, mutated_uv_pixel_positions = random_flip_vertical(image, uv_pixel_positions)
    mutated_image, mutated_uv_pixel_positions = random_flip_horizontal(mutated_image, mutated_uv_pixel_positions)
    return mutated_image, mutated_uv_pixel_positions


def random_flip_vertical(image, uv_pixel_positions):
    """
    Randomly flip the image and the pixel positions vertically (flip up/down)

    See random_image_and_indices_mutation() for documentation of args and return types.

    """

    if random.random() < 0.5:
        return image, uv_pixel_positions  # Randomly do not apply

    print "Flip vertically"
    mutated_image = ImageOps.flip(image)
    v_pixel_positions = uv_pixel_positions[1]
    mutated_v_pixel_positions = image.height - v_pixel_positions
    mutated_uv_pixel_positions = (uv_pixel_positions[0], mutated_v_pixel_positions)
    return mutated_image, mutated_uv_pixel_positions

def random_flip_horizontal(image, uv_pixel_positions):
    """
    Randomly flip the image and the pixel positions horizontall (flip left/right)

    See random_image_and_indices_mutation() for documentation of args and return types.

    """

    if random.random() < 0.5:
        return image, uv_pixel_positions  # Randomly do not apply

    print "Flip left and right"
    mutated_image = ImageOps.mirror(image)
    u_pixel_positions = uv_pixel_positions[0]
    mutated_u_pixel_positions = image.width - u_pixel_positions
    mutated_uv_pixel_positions = (mutated_u_pixel_positions, uv_pixel_positions[1])
    return mutated_image, mutated_uv_pixel_positions