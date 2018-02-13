# io
import imageio

# math
import numpy as numpy
import math
import random
from numpy.linalg import inv

def rand_select_pixel(width,height):
    x = random.randint(0,width)
    y = random.randint(0,height)
    return (x,y)

def find_pixel_correspondence(log_dir, img_a, img_b, uv_a=None):
    img1_filename = log_dir+"images/"+img_a+"_rgb.png"
    img2_filename = log_dir+"images/"+img_b+"_rgb.png"
    img1_depth_filename = log_dir+"images/"+img_a+"_depth.png"
    img2_depth_filename = log_dir+"images/"+img_b+"_depth.png"
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

    img1_depth = imageio.imread(img1_depth_filename)

    if uv_a is None:
        uv_a = rand_select_pixel(width=640,height=480)
    print uv_a

    body_to_rdf = numpy.zeros((3,3))
    body_to_rdf[0,1] = -1.0
    body_to_rdf[1,2] = -1.0
    body_to_rdf[2,0] = 1.0
    rdf_to_body = inv(body_to_rdf)

    K = numpy.zeros((3,3))
    K[0,0] = 528.0 # focal x
    K[1,1] = 528.0 # focal y
    K[0,2] = 320.0 # principal point x
    K[1,2] = 240.0 # principal point y
    K[2,2] = 1.0
    K_inv = inv(K)

    depth = img1_depth[uv_a[::-1]]*1.0/1000
    print "depth, ", depth
    u = uv_a[0]
    v = uv_a[1]

    x = u*depth
    y = v*depth
    z = depth
    vec = numpy.array([x,y,z])

    point_camera_frame_rdf = K_inv.dot(vec)

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

    point_world_frame_rdf = apply_transform(point_camera_frame_rdf, img1_pose_4)
    point_camera_2_frame_rdf = apply_transform(point_world_frame_rdf, invert_transform(img2_pose_4))

    vec2 = K.dot(point_camera_2_frame_rdf)
    u2 = vec2[0]/vec2[2]
    v2 = vec2[1]/vec2[2]
    uv_b = (u2, v2)
    print uv_b
    return (uv_a, uv_b)