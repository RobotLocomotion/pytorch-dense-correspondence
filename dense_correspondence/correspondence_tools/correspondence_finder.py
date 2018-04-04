# torch
import torch

# math
import numpy as numpy
import numpy as np
import math
from numpy.linalg import inv
import random

# io
from PIL import Image

# torchvision
import sys
sys.path.insert(0, '../pytorch-segmentation-detection/vision/') # from subrepo
from torchvision import transforms


from dense_correspondence_manipulation.utils.constants import *

# turns out to be faster to do this match generation on the CPU
# for the general size of params we expect
# also this will help by not taking up GPU memory, 
# allowing batch sizes to stay large
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

def pytorch_rand_select_pixel(width,height,num_samples=1):
    two_rand_numbers = torch.rand(2,num_samples)
    two_rand_numbers[0,:] = two_rand_numbers[0,:]*width
    two_rand_numbers[1,:] = two_rand_numbers[1,:]*height
    two_rand_ints    = torch.floor(two_rand_numbers).type(dtype_long)
    return (two_rand_ints[0], two_rand_ints[1])

def get_K_matrix():
    K = numpy.zeros((3,3))
    K[0,0] = 533.6422696034836 # focal x
    K[1,1] = 534.7824445233571 # focal y
    K[0,2] = 319.4091030774892 # principal point x
    K[1,2] = 236.4374299691866 # principal point y
    K[2,2] = 1.0
    return K

def get_body_to_rdf():
    body_to_rdf = numpy.zeros((3,3))
    body_to_rdf[0,1] = -1.0
    body_to_rdf[1,2] = -1.0
    body_to_rdf[2,0] = 1.0
    return body_to_rdf

def invert_transform(transform4):
    transform4_copy = numpy.copy(transform4)
    R = transform4_copy[0:3,0:3]
    R = numpy.transpose(R)
    transform4_copy[0:3,0:3] = R
    t = transform4_copy[0:3,3]
    inv_t = -1.0 * numpy.transpose(R).dot(t)
    transform4_copy[0:3,3] = inv_t
    return transform4_copy

def apply_transform_torch(vec3, transform4):
    ones_row = torch.ones_like(vec3[0,:]).type(dtype_float).unsqueeze(0)
    vec4 = torch.cat((vec3,ones_row),0)
    vec4 = transform4.mm(vec4)
    return vec4[0:3]

def random_sample_from_masked_image(img_mask, num_samples):
    """
    Samples num_samples pixel locations from the masked image
    :param img_mask: numpy.ndarray
        - masked image, we will select from the non-zero entries
        - shape is H x W
    :param num_samples: int
        - number of random indices to return
    :return: List of np.array
    """
    idx_tuple = img_mask.nonzero()
    num_nonzero = len(idx_tuple[0])
    rand_inds = random.sample(range(0,num_nonzero), num_samples)

    sampled_idx_list = []
    for i, idx in enumerate(idx_tuple):
        sampled_idx_list.append(idx[rand_inds])

    return sampled_idx_list

def pinhole_projection_image_to_world(uv, z, K):
    """
    Takes a (u,v) pixel location to it's 3D location in camera frame.
    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for a detailed explanation.

    :param uv: pixel location in image
    :type uv:
    :param z: depth, in camera frame
    :type z: float
    :param K: 3 x 3 camera intrinsics matrix
    :type K: numpy.ndarray
    :return: (x,y,z) in camera frame
    :rtype: numpy.array size (3,)
    """

    u_v_1 = np.array([uv[0], uv[1], 1])
    pos = z * np.matmul(inv(K),u_v_1)
    return pos


# in torch 0.3 we don't yet have torch.where(), although this
# is there in 0.4 (not yet stable release)
# for more see: https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
def where(cond, x_1, x_2):
    cond = cond.type(dtype_float)    
    return (cond * x_1) + ((1-cond) * x_2)

# This function currenlty only depends on pixel positions,
# nothing about the underlying data.
# It may be worthile in future to use the depth
# To check that we have depth information
def create_non_correspondences(uv_a, uv_b_matches, num_non_matches_per_match=100):
    if uv_a == None:
        return None
    num_matches = len(uv_a[0])
    # for each in uv_a, we want non-matches
    # shape of uv_a : just a vector, of length corresponding to how many samples
    
    # first just randomly sample "non_matches"
    # we will later move random samples that were too close to being matches
    uv_b_non_matches = pytorch_rand_select_pixel(width=640,height=480,num_samples=num_matches*num_non_matches_per_match)
    uv_b_non_matches = (uv_b_non_matches[0].view(num_matches,num_non_matches_per_match), uv_b_non_matches[1].view(num_matches,num_non_matches_per_match))

    # uv_b_matches can now be used to make sure no "non_matches" are too close
    # to preserve tensor size, rather than pruning, we can perturb these in pixel space
    copied_uv_b_matches_0 = torch.t(uv_b_matches[0].repeat(num_non_matches_per_match, 1))
    copied_uv_b_matches_1 = torch.t(uv_b_matches[1].repeat(num_non_matches_per_match, 1))

    diffs_0 = copied_uv_b_matches_0 - uv_b_non_matches[0].type(dtype_float)
    diffs_1 = copied_uv_b_matches_1 - uv_b_non_matches[1].type(dtype_float)

    diffs_0_flattened = diffs_0.view(-1,1)
    diffs_1_flattened = diffs_1.view(-1,1)

    diffs_0_flattened = torch.abs(diffs_0_flattened).squeeze(1)
    diffs_1_flattened = torch.abs(diffs_1_flattened).squeeze(1)

    need_to_be_perturbed = torch.zeros_like(diffs_0_flattened)
    ones = torch.zeros_like(diffs_0_flattened)
    num_pixels_too_close = 1.0
    threshold = torch.ones_like(diffs_0_flattened)*num_pixels_too_close

    need_to_be_perturbed = where(diffs_0_flattened < threshold, ones, need_to_be_perturbed)
    need_to_be_perturbed = where(diffs_1_flattened < threshold, ones, need_to_be_perturbed)

    minimal_perturb        = num_pixels_too_close/2
    minimal_perturb_vector = (torch.rand(len(need_to_be_perturbed))*2).floor()*(minimal_perturb*2)-minimal_perturb
    std_dev = 10
    random_vector = torch.randn(len(need_to_be_perturbed))*std_dev + minimal_perturb_vector
    perturb_vector = need_to_be_perturbed*random_vector

    uv_b_non_matches_0_flat = uv_b_non_matches[0].view(-1,1).type(dtype_float).squeeze(1)
    uv_b_non_matches_1_flat = uv_b_non_matches[1].view(-1,1).type(dtype_float).squeeze(1)

    uv_b_non_matches_0_flat = uv_b_non_matches_0_flat + perturb_vector
    uv_b_non_matches_1_flat = uv_b_non_matches_1_flat + perturb_vector

    # now just need to wrap around any that went out of bounds

    # handle wrapping in width
    lower_bound = 0.0
    upper_bound = 639.0
    lower_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * upper_bound

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat > upper_bound_vec, 
        uv_b_non_matches_0_flat - upper_bound_vec, 
        uv_b_non_matches_0_flat)

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat < lower_bound_vec, 
        uv_b_non_matches_0_flat + upper_bound_vec, 
        uv_b_non_matches_0_flat)

    # handle wrapping in height
    lower_bound = 0.0
    upper_bound = 479.0
    lower_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * upper_bound

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat > upper_bound_vec, 
        uv_b_non_matches_1_flat - upper_bound_vec, 
        uv_b_non_matches_1_flat)

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat < lower_bound_vec, 
        uv_b_non_matches_1_flat + upper_bound_vec, 
        uv_b_non_matches_1_flat)

    return (uv_b_non_matches_0_flat.view(num_matches, num_non_matches_per_match),
        uv_b_non_matches_1_flat.view(num_matches, num_non_matches_per_match))

# Optionally, uv_a specifies the pixels in img_a for which to find matches
# If uv_a is not set, then random correspondences are attempted to be found
def batch_find_pixel_correspondences(img_a_depth, img_a_pose, img_b_depth, img_b_pose, uv_a=None, num_attempts=20, device='CPU', img_a_mask=None):
    """
    Computes pixel correspondences in batch

    :param img_a_depth:
    :type img_a_depth:
    :param img_a_pose:
    :type img_a_pose:
    :param img_b_depth:
    :type img_b_depth:
    :param img_b_pose:
    :type img_b_pose:
    :param uv_a:
    :type uv_a:
    :param num_attempts:
    :type num_attempts:
    :param device:
    :type device:
    :param img_a_mask:
    :type img_a_mask:
    :return: Tuple (uv_a, uv_b). Each of uv_a is a tuple of torch.FloatTensors
    :rtype:
    """
    global dtype_float
    global dtype_long
    if device == 'CPU':
        dtype_float = torch.FloatTensor
        dtype_long = torch.LongTensor
    if device =='GPU':
        dtype_float = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

    if uv_a is None:
        uv_a = pytorch_rand_select_pixel(width=640,height=480, num_samples=num_attempts)
    else:
        uv_a = (torch.LongTensor([uv_a[0]]).type(dtype_long), torch.LongTensor([uv_a[1]]).type(dtype_long))
        num_attempts = 1

    if img_a_mask is None:
        uv_a_vec = (torch.ones(num_attempts).type(dtype_long)*uv_a[0],torch.ones(num_attempts).type(dtype_long)*uv_a[1])
        print uv_a_vec[0].shape
        print "unmasked shape"
        uv_a_vec_flattened = uv_a_vec[1]*640+uv_a_vec[0]
    else:
        img_a_mask = torch.from_numpy(img_a_mask).type(dtype_float)
        mask_a = img_a_mask.squeeze(0)
        mask_a = mask_a/torch.max(mask_a)
        nonzero = (torch.nonzero(mask_a)).type(dtype_long)
        uv_a_vec = (nonzero[:,1], nonzero[:,0])
        uv_a_vec_flattened = uv_a_vec[1]*640+uv_a_vec[0]

        # mask_a = mask_a.view(640*480,1).squeeze(1)
        # mask_a_indices_flat = torch.nonzero(mask_a)
        # if len(mask_a_indices_flat) == 0:
        #     return (None, None)
        # num_samples = 10000
        # rand_numbers_a = torch.rand(num_samples)*len(mask_a_indices_flat)
        # rand_indices_a = torch.floor(rand_numbers_a).type(dtype_long)
        # uv_a_vec_flattened = torch.index_select(mask_a_indices_flat, 0, rand_indices_a).squeeze(1)
        # uv_a_vec = (uv_a_vec_flattened/640, uv_a_vec_flattened%640)

    K = get_K_matrix()
    K_inv = inv(K)
    body_to_rdf = get_body_to_rdf()
    rdf_to_body = inv(body_to_rdf)

    img_a_depth_torch = torch.from_numpy(img_a_depth).type(dtype_float)
    img_a_depth_torch = torch.squeeze(img_a_depth_torch, 0)
    img_a_depth_torch = img_a_depth_torch.view(-1,1)

    
    depth_vec = torch.index_select(img_a_depth_torch, 0, uv_a_vec_flattened)*1.0/DEPTH_IM_SCALE
    depth_vec = depth_vec.squeeze(1)
    
    # Prune based on
    # Case 1: depth is zero (for this data, this means no-return)
    nonzero_indices = torch.nonzero(depth_vec)
    if nonzero_indices.dim() == 0:
        return (None, None)
    nonzero_indices = nonzero_indices.squeeze(1)
    depth_vec = torch.index_select(depth_vec, 0, nonzero_indices)

    # prune u_vec and v_vec, then multiply by already pruned depth_vec
    u_a_pruned = torch.index_select(uv_a_vec[0], 0, nonzero_indices)
    u_vec = u_a_pruned.type(dtype_float)*depth_vec

    v_a_pruned = torch.index_select(uv_a_vec[1], 0, nonzero_indices)
    v_vec = v_a_pruned.type(dtype_float)*depth_vec

    z_vec = depth_vec

    full_vec = torch.stack((u_vec, v_vec, z_vec))

    K_inv_torch = torch.from_numpy(K_inv).type(dtype_float)
    point_camera_frame_rdf_vec = K_inv_torch.mm(full_vec)

    point_world_frame_rdf_vec = apply_transform_torch(point_camera_frame_rdf_vec, torch.from_numpy(img_a_pose).type(dtype_float))
    point_camera_2_frame_rdf_vec = apply_transform_torch(point_world_frame_rdf_vec, torch.from_numpy(invert_transform(img_b_pose)).type(dtype_float))

    K_torch = torch.from_numpy(K).type(dtype_float)
    vec2_vec = K_torch.mm(point_camera_2_frame_rdf_vec)

    u2_vec = vec2_vec[0]/vec2_vec[2]
    v2_vec = vec2_vec[1]/vec2_vec[2]

    maybe_z2_vec = point_camera_2_frame_rdf_vec[2]

    z2_vec = vec2_vec[2]

    # Prune based on
    # Case 2: the pixels projected into image b are outside FOV
    # u2_vec bounds should be: 0, 640
    # v2_vec bounds should be: 0, 480

    ## this example prunes any elements in the vector above or below the bounds

    ## do u2-based pruning
    u2_vec_lower_bound = 0.0
    u2_vec_upper_bound = 639.999  # careful, needs to be epsilon less!!
    lower_bound_vec = torch.ones_like(u2_vec) * u2_vec_lower_bound
    upper_bound_vec = torch.ones_like(u2_vec) * u2_vec_upper_bound
    zeros_vec       = torch.zeros_like(u2_vec)

    u2_vec = where(u2_vec < lower_bound_vec, zeros_vec, u2_vec)
    u2_vec = where(u2_vec > upper_bound_vec, zeros_vec, u2_vec)
    in_bound_indices = torch.nonzero(u2_vec)
    if in_bound_indices.dim() == 0:
        return (None, None)
    in_bound_indices = in_bound_indices.squeeze(1)

    # apply pruning
    u2_vec = torch.index_select(u2_vec, 0, in_bound_indices)
    v2_vec = torch.index_select(v2_vec, 0, in_bound_indices)
    z2_vec = torch.index_select(z2_vec, 0, in_bound_indices)
    u_a_pruned = torch.index_select(u_a_pruned, 0, in_bound_indices) # also prune from first list
    v_a_pruned = torch.index_select(v_a_pruned, 0, in_bound_indices) # also prune from first list

    ## do v2-based pruning
    v2_vec_lower_bound = 0.0
    v2_vec_upper_bound = 479.999
    lower_bound_vec = torch.ones_like(v2_vec) * v2_vec_lower_bound
    upper_bound_vec = torch.ones_like(v2_vec) * v2_vec_upper_bound
    zeros_vec       = torch.zeros_like(v2_vec)    

    v2_vec = where(v2_vec < lower_bound_vec, zeros_vec, v2_vec)
    v2_vec = where(v2_vec > upper_bound_vec, zeros_vec, v2_vec)
    in_bound_indices = torch.nonzero(v2_vec)
    if in_bound_indices.dim() == 0:
        return (None, None)
    in_bound_indices = in_bound_indices.squeeze(1)

    # apply pruning
    u2_vec = torch.index_select(u2_vec, 0, in_bound_indices)
    v2_vec = torch.index_select(v2_vec, 0, in_bound_indices)
    z2_vec = torch.index_select(z2_vec, 0, in_bound_indices)
    u_a_pruned = torch.index_select(u_a_pruned, 0, in_bound_indices) # also prune from first list
    v_a_pruned = torch.index_select(v_a_pruned, 0, in_bound_indices) # also prune from first list

    # Prune based on
    # Case 3: the pixels in image b are occluded, OR there is no depth return in image b so we aren't sure

    img_b_depth_torch = torch.from_numpy(img_b_depth).type(dtype_float)
    img_b_depth_torch = torch.squeeze(img_b_depth_torch, 0)
    img_b_depth_torch = img_b_depth_torch.view(-1,1)

    uv_b_vec_flattened = (v2_vec.type(dtype_long)*640+u2_vec.type(dtype_long))  # simply round to int -- good enough 
                                                                       # occlusion check for smooth surfaces


    this_max = torch.max(uv_b_vec_flattened)
    if this_max >= 307200:
        print this_max, "is max here"
        print img_b_depth_torch.shape
        print ""
        print ""
        print "WTF!!"
        print 
        print torch.max(v2_vec)
        print torch.max(u2_vec)
        print "end WTF"
    this_min = torch.min(uv_b_vec_flattened)
    if this_min < 0:
        print "less than 0?"
        exit(0)

    depth2_vec = torch.index_select(img_b_depth_torch, 0, uv_b_vec_flattened)*1.0/1000
    depth2_vec = depth2_vec.squeeze(1)

    # occlusion margin, in meters
    occlusion_margin = 0.03
    z2_vec = z2_vec - occlusion_margin
    zeros_vec = torch.zeros_like(depth2_vec)

    depth2_vec = where(depth2_vec < zeros_vec, zeros_vec, depth2_vec) # to be careful, prune any negative depths
    depth2_vec = where(depth2_vec < z2_vec, zeros_vec, depth2_vec)    # prune occlusions
    non_occluded_indices = torch.nonzero(depth2_vec)
    if non_occluded_indices.dim() == 0:
        return (None, None)
    non_occluded_indices = non_occluded_indices.squeeze(1)
    depth2_vec = torch.index_select(depth2_vec, 0, non_occluded_indices)

    # apply pruning
    u2_vec = torch.index_select(u2_vec, 0, non_occluded_indices)
    v2_vec = torch.index_select(v2_vec, 0, non_occluded_indices)
    u_a_pruned = torch.index_select(u_a_pruned, 0, non_occluded_indices) # also prune from first list
    v_a_pruned = torch.index_select(v_a_pruned, 0, non_occluded_indices) # also prune from first list

    uv_b_vec = (u2_vec, v2_vec)
    uv_a_vec = (u_a_pruned, v_a_pruned)
    return (uv_a_vec, uv_b_vec)