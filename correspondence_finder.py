# io
from PIL import Image

# WARNING repo must be
# same level as pytorch-segmentation-detection
import sys
sys.path.insert(0, '../pytorch-segmentation-detection/vision/')
from torchvision import transforms

# math
import numpy as numpy
import math
from numpy.linalg import inv

# torch
import torch

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
    K[0,0] = 528.0 # focal x
    K[1,1] = 528.0 # focal y
    K[0,2] = 320.0 # principal point x
    K[1,2] = 240.0 # principal point y
    K[2,2] = 1.0
    return K

def get_body_to_rdf():
    body_to_rdf = numpy.zeros((3,3))
    body_to_rdf[0,1] = -1.0
    body_to_rdf[1,2] = -1.0
    body_to_rdf[2,0] = 1.0
    return body_to_rdf

def get_poses(log_dir, img_a, img_b):
    img1_time_filename = log_dir+"images/"+img_a+"_utime.txt"
    img2_time_filename = log_dir+"images/"+img_b+"_utime.txt"

    def get_time(time_filename):
        with open (time_filename) as f:
            content = f.readlines()
        return int(content[0])/1e6

    img1_time = get_time(img1_time_filename)
    img2_time = get_time(img2_time_filename)

    posegraph_filename = log_dir+"posegraph.posegraph"
    with open(posegraph_filename) as f:
        content = f.readlines()
    pose_list = [x.strip().split() for x in content] 

    def get_pose(time, pose_list):
        if (time <= float(pose_list[0][0])):
            pose = pose_list[0]
            pose = [float(x) for x in pose[1:]]
            return pose
        for pose in pose_list:
            if (time <= float(pose[0])):
                pose = [float(x) for x in pose[1:]]
                return pose
        print "did not find matching pose"

    img1_pose = get_pose(img1_time, pose_list)
    img2_pose = get_pose(img2_time, pose_list)

    _EPS = numpy.finfo(float).eps * 4.0

    def quaternion_matrix(quaternion):
        q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
        n = numpy.dot(q, q)
        if n < _EPS:
            return numpy.identity(4)
        q *= math.sqrt(2.0 / n)
        q = numpy.outer(q, q)
        return numpy.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
            [                0.0,                 0.0,                 0.0, 1.0]])

    def labelfusion_pose_to_homogeneous_transform(lf_pose):
        homogeneous_transform = quaternion_matrix([lf_pose[6], lf_pose[3], lf_pose[4], lf_pose[5]])
        homogeneous_transform[0,3] = lf_pose[0]
        homogeneous_transform[1,3] = lf_pose[1]
        homogeneous_transform[2,3] = lf_pose[2]
        return homogeneous_transform

    img1_pose_4 = labelfusion_pose_to_homogeneous_transform(img1_pose)
    img2_pose_4 = labelfusion_pose_to_homogeneous_transform(img2_pose)
    return img1_pose_4, img2_pose_4

def invert_transform(transform4):
    transform4_copy = numpy.copy(transform4)
    R = transform4_copy[0:3,0:3]
    R = numpy.transpose(R)
    transform4_copy[0:3,0:3] = R
    t = transform4_copy[0:3,3]
    inv_t = -1.0 * R.dot(t)
    transform4_copy[0:3,3] = inv_t
    return transform4_copy

def apply_transform(vec3, transform4):
    vec4 = numpy.array([vec3[0], vec3[1], vec3[2], 1.0])
    vec4 = transform4.dot(vec4)
    return numpy.array([vec4[0], vec4[1], vec4[2]])

def apply_transform_torch(vec3, transform4):
    ones_row = torch.ones_like(vec3[0,:]).type(dtype_float).unsqueeze(0)
    vec4 = torch.cat((vec3,ones_row),0)
    vec4 = transform4.mm(vec4)
    return vec4[0:3]

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
    num_pixels_too_close = 3.0
    threshold = torch.ones_like(diffs_0_flattened)*num_pixels_too_close

    need_to_be_perturbed = where(diffs_0_flattened < threshold, ones, need_to_be_perturbed)
    need_to_be_perturbed = where(diffs_1_flattened < threshold, ones, need_to_be_perturbed)

    perturb_amount = 10.0
    perturb_vector = need_to_be_perturbed*perturb_amount

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
def batch_find_pixel_correspondences(log_dir, img_a, img_b, uv_a=None, num_attempts=20, device='CPU'):
    global dtype_float
    global dtype_long
    if device == 'CPU':
        dtype_float = torch.FloatTensor
        dtype_long = torch.LongTensor
    if device =='GPU':
        dtype_float = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

    img1_depth_filename = log_dir+"images/"+img_a+"_depth.png"
    img2_depth_filename = log_dir+"images/"+img_b+"_depth.png"
    img1_pose_4, img2_pose_4 = get_poses(log_dir, img_a, img_b)
    img2_pose_4_inverted = invert_transform(img2_pose_4)

    if uv_a is None:
        uv_a = pytorch_rand_select_pixel(width=640,height=480, num_samples=num_attempts)
    else:
        uv_a = (torch.LongTensor([uv_a[0]]).type(dtype_long), torch.LongTensor([uv_a[1]]).type(dtype_long))
        num_attempts = 1

    K = get_K_matrix()
    K_inv = inv(K)
    body_to_rdf = get_body_to_rdf()
    rdf_to_body = inv(body_to_rdf)

    to_tensor_transform = transforms.Compose(
    [
         transforms.ToTensor(),
    ])
    img1_depth_torch = to_tensor_transform(Image.open(img1_depth_filename)).type(dtype_float)
    img1_depth_torch = torch.squeeze(img1_depth_torch, 0)
    img1_depth_torch = img1_depth_torch.view(-1,1)

    uv_a_vec = (torch.ones(num_attempts).type(dtype_long)*uv_a[0],torch.ones(num_attempts).type(dtype_long)*uv_a[1])
    uv_a_vec_flattened = uv_a_vec[1]*640+uv_a_vec[0]
    depth_vec = torch.index_select(img1_depth_torch, 0, uv_a_vec_flattened)*1.0/1000
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

    point_world_frame_rdf_vec = apply_transform_torch(point_camera_frame_rdf_vec, torch.from_numpy(img1_pose_4).type(dtype_float))
    point_camera_2_frame_rdf_vec = apply_transform_torch(point_world_frame_rdf_vec, torch.from_numpy(img2_pose_4_inverted).type(dtype_float))

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
    u2_vec_upper_bound = 640.0
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
    v2_vec_upper_bound = 480.0
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
    to_tensor_transform = transforms.Compose(
    [
         transforms.ToTensor(),
    ])
    img2_depth_torch = to_tensor_transform(Image.open(img2_depth_filename)).type(dtype_float)
    img2_depth_torch = torch.squeeze(img2_depth_torch, 0)
    img2_depth_torch = img2_depth_torch.view(-1,1)

    uv_b_vec_flattened = (v2_vec.type(dtype_long)*640+u2_vec.type(dtype_long))  # simply round to int -- good enough 
                                                                       # occlusion check for smooth surfaces

    depth2_vec = torch.index_select(img2_depth_torch, 0, uv_b_vec_flattened)*1.0/1000
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