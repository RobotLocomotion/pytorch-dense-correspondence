# Basic I/O utils
import yaml
import numpy as np
import os
import sys
import time
import socket
import getpass
import fnmatch

import dense_correspondence_manipulation.utils.transformations as transformations

def getDictFromYamlFilename(filename):
    """
    Read data from a YAML files
    """
    stream = file(filename)
    return yaml.load(stream)

def saveToYaml(data, filename):
    """
    Save a data to a YAML file
    """
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def getDenseCorrespondenceSourceDir():
    return os.getenv("DC_SOURCE_DIR")

def getPdcPath():
    return os.path.join(getDenseCorrespondenceSourceDir(), "data_volume", "pdc")

def dictFromPosQuat(pos, quat):
    """
    Make a dictionary from position and quaternion vectors
    """
    d = dict()
    d['translation'] = dict()
    d['translation']['x'] = pos[0]
    d['translation']['y'] = pos[1]
    d['translation']['z'] = pos[2]

    d['quaternion'] = dict()
    d['quaternion']['w'] = quat[0]
    d['quaternion']['x'] = quat[1]
    d['quaternion']['y'] = quat[2]
    d['quaternion']['z'] = quat[3]

    return d


def getQuaternionFromDict(d):
    """
    Get the quaternion from a dict describing a transform. The dict entry could be
    one of orientation, rotation, quaternion depending on the convention
    """
    quat = None
    quatNames = ['orientation', 'rotation', 'quaternion']
    for name in quatNames:
        if name in d:
            quat = d[name]


    if quat is None:
        raise ValueError("Error when trying to extract quaternion from dict, your dict doesn't contain a key in ['orientation', 'rotation', 'quaternion']")

    return quat

def getPaddedString(idx, width=6):
    return str(idx).zfill(width)

def set_cuda_visible_devices(gpu_list):
    """
    Sets CUDA_VISIBLE_DEVICES environment variable to only show certain gpus
    If gpu_list is empty does nothing
    :param gpu_list: list of gpus to set as visible
    :return: None
    """

    if len(gpu_list) == 0:
        print "using all CUDA gpus"
        return

    cuda_visible_devices = ""
    for gpu in gpu_list:
        cuda_visible_devices += str(gpu) + ","

    print "setting CUDA_VISIBLE_DEVICES = ", cuda_visible_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def set_default_cuda_visible_devices():
    config = get_defaults_config()
    host_name = socket.gethostname()
    user_name = getpass.getuser()
    gpu_list = config[host_name][user_name]["cuda_visible_devices"]
    set_cuda_visible_devices(gpu_list)

def get_defaults_config():
    dc_source_dir = getDenseCorrespondenceSourceDir()
    default_config_file = os.path.join(dc_source_dir, 'config', 'defaults.yaml')

    return getDictFromYamlFilename(default_config_file)


def add_dense_correspondence_to_python_path():
    dc_source_dir = getDenseCorrespondenceSourceDir()
    sys.path.append(dc_source_dir)
    sys.path.append(os.path.join(dc_source_dir, 'pytorch-segmentation-detection'))

    # for some reason it is critical that this be at the beginning . . .
    sys.path.insert(0, os.path.join(dc_source_dir, 'pytorch-segmentation-detection', 'vision'))


def convert_to_absolute_path(path):
    """
    Converts a potentially relative path to an absolute path by pre-pending the home directory
    :param path: absolute or relative path
    :type path: str
    :return: absolute path
    :rtype: str
    """

    if os.path.isdir(path):
        return path


    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, path)

def get_current_time_unique_name():
    """
    Converts current date to a unique name
    :return:
    :rtype: str
    """

    unique_name = time.strftime("%Y%m%d-%H%M%S")
    return unique_name

def homogenous_transform_from_dict(d):
    """
    Returns a transform from a standard encoding in dict format
    :param d:
    :return:
    """
    pos = [0]*3
    pos[0] = d['translation']['x']
    pos[1] = d['translation']['y']
    pos[2] = d['translation']['z']

    quatDict = getQuaternionFromDict(d)
    quat = [0]*4
    quat[0] = quatDict['w']
    quat[1] = quatDict['x']
    quat[2] = quatDict['y']
    quat[3] = quatDict['z']

    transform_matrix = transformations.quaternion_matrix(quat)
    transform_matrix[0:3,3] = np.array(pos)

    return transform_matrix

def compute_distance_between_poses(pose_a, pose_b):
    """
    Computes the linear difference between pose_a and pose_b
    :param pose_a: 4 x 4 homogeneous transform
    :type pose_a:
    :param pose_b:
    :type pose_b:
    :return: Distance between translation component of the poses
    :rtype:
    """

    pos_a = pose_a[0:3,3]
    pos_b = pose_b[0:3,3]

    return np.linalg.norm(pos_a - pos_b)

def compute_angle_between_quaternions(q, r):
    """
    Computes the angle between two quaternions.

    theta = arccos(2 * <q1, q2>^2 - 1)

    See https://math.stackexchange.com/questions/90081/quaternion-distance
    :param q: numpy array in form [w,x,y,z]. As long as both q,r are consistent it doesn't matter
    :type q:
    :param r:
    :type r:
    :return: angle between the quaternions, in radians
    :rtype:
    """

    theta = 2*np.arccos(2 * np.dot(q,r)**2 - 1)
    return theta

def compute_angle_between_poses(pose_a, pose_b):
    """
    Computes the angle distance in radians between two homogenous transforms
    :param pose_a: 4 x 4 homogeneous transform
    :type pose_a:
    :param pose_b:
    :type pose_b:
    :return: Angle between poses in radians
    :rtype:
    """

    quat_a = transformations.quaternion_from_matrix(pose_a)
    quat_b = transformations.quaternion_from_matrix(pose_b)

    return compute_angle_between_quaternions(quat_a, quat_b)



def get_model_param_file_from_directory(model_folder, iteration=None):
    """
    Gets the 003500.pth and 003500.pth.opt files from the specified folder

    :param model_folder: location of the folder containing the param files 001000.pth. Can be absolute or relative path. If relative then it is relative to pdc/trained_models/
    :type model_folder:
    :param iteration: which index to use, e.g. 3500, if None it loads the latest one
    :type iteration:
    :return: model_param_file, optim_param_file, iteration
    :rtype: str, str, int
    """

    if not os.path.isdir(model_folder):
        pdc_path = getPdcPath()
        model_folder = os.path.join(pdc_path, "trained_models", model_folder)

    # find idx.pth and idx.pth.opt files
    if iteration is None:
        files = os.listdir(model_folder)
        model_param_file = sorted(fnmatch.filter(files, '*.pth'))[-1]
        iteration = int(model_param_file.split(".")[0])
        optim_param_file = sorted(fnmatch.filter(files, '*.pth.opt'))[-1]
    else:
        prefix = getPaddedString(iteration, width=6)
        model_param_file = prefix + ".pth"
        optim_param_file = prefix + ".pth.opt"

    print "model_param_file", model_param_file
    model_param_file = os.path.join(model_folder, model_param_file)
    optim_param_file = os.path.join(model_folder, optim_param_file)

    return model_param_file, optim_param_file, iteration


def flattened_pixel_locations_to_u_v(flat_pixel_locations, image_width):
    """
    :param flat_pixel_locations: A torch.LongTensor of shape torch.Shape([n,1]) where each element
     is a flattened pixel index, i.e. some integer between 0 and 307,200 for a 640x480 image

    :type flat_pixel_locations: torch.LongTensor

    :return A tuple torch.LongTensor in (u,v) format
    the pixel and the second column is the v coordinate

    """
    return (flat_pixel_locations%image_width, flat_pixel_locations/image_width)

def uv_to_flattened_pixel_locations(uv_tuple, image_width):
    """
    Converts to a flat tensor
    """
    flat_pixel_locations = uv_tuple[1]*image_width + uv_tuple[0]
    return flat_pixel_locations

class CameraIntrinsics(object):
    """
    Useful class for wrapping camera intrinsics and loading them from a
    camera_info.yaml file
    """
    def __init__(self, cx, cy, fx, fy, width, height):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.width = width
        self.height = height

        self.K = self.get_camera_matrix()

    def get_camera_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0,0,1]])

    @staticmethod
    def from_yaml_file(filename):
        config = getDictFromYamlFilename(filename)

        fx = config['camera_matrix']['data'][0]
        cx = config['camera_matrix']['data'][2]

        fy = config['camera_matrix']['data'][4]
        cy = config['camera_matrix']['data'][5]

        width = config['image_width']
        height = config['image_height']

        return CameraIntrinsics(cx, cy, fx, fy, width, height)

