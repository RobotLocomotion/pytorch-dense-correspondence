
import numpy as np
import os


from director import imageview
from director import vtkAll as vtk
from director import transformUtils
from director import visualization as vis
from director import viewbehaviors
from director import vtkNumpy as vnp
from director.debugVis import DebugData
from director.timercallback import TimerCallback
from director import ioUtils
from director import transformUtils
from director import filterUtils
import PythonQt

import dense_correspondence_manipulation.utils.director_utils as director_utils
import dense_correspondence_manipulation.utils.utils as utils
import dense_correspondence_manipulation.utils.segmentation as segmentation

from dense_correspondence_manipulation.utils.constants import *

from dense_correspondence.dataset.scene_structure import SceneStructure

class FusionCameraPoses(object):
    """
    Abstract class for storing poses coming from a fusion reconstruction

    """
    pass


class ElasticFusionCameraPoses(FusionCameraPoses):

    def __init__(self, posegraph_filename):
        self.posegraph_filename = posegraph_filename
        pass

    @property
    def first_frame_to_world(self):
        return self._first_frame_to_world

    @first_frame_to_world.setter
    def first_frame_to_world(self, value):
        assert isinstance(value, vtk.vtkTransform)
        self._first_frame_to_world = value

    @property
    def posegraph_filename(self):
        return self._posegraph_filename

    @posegraph_filename.setter
    def posegraph_filename(self, value):
        self._posegraph_filename = value
        self.load_camera_poses(self.posegraph_filename)

    def load_camera_poses(self, posegraphFile):
        data = np.loadtxt(posegraphFile)
        self.poseTimes = np.array(data[:, 0] * 1e6, dtype=int)
        self.poses = []
        for idx, pose in enumerate(data[:, 1:]):
            pos = pose[:3]
            quat = pose[6], pose[3], pose[4], pose[5]  # quat data from file is ordered as x, y, z, w
            self.poses.append((pos, quat))


    def get_camera_pose(self, idx):
        pos, quat = self.poses[idx]
        transform = transformUtils.transformFromPose(pos, quat)
        return pos, quat, transform

    def get_camera_to_world_pose(self, idx):
        _, _, camera_to_first_frame = self.get_camera_pose(idx)
        camera_to_world = transformUtils.concatenateTransforms([camera_to_first_frame,
                                                                self.first_frame_to_world])

        return camera_to_world

class CameraPoses(object):
    """
    Simple wrapper class for getting camera poses

    """
    def __init__(self, pose_dict):
        self.pose_dict = pose_dict

    @property
    def pose_dict(self):
        return self._pose_dict

    @pose_dict.setter
    def pose_dict(self, value):
        self._pose_dict = value

    def get_camera_pose(self, idx):
        camera_pose_dict = self.pose_dict[idx]['camera_to_world']
        return director_utils.transformFromPose(camera_pose_dict)

    def get_data(self, idx):
        return self.pose_dict[idx]

    def num_poses(self):
        return len(self.pose_dict)

class FusionReconstruction(object):
    """
    A utility class for storing information about a 3D reconstruction

    e.g. reconstruction produced by ElasticFusion
    """

    def __init__(self):
        self.poly_data_type = "points"
        pass


    def setup(self):
        self.load_poly_data()
        self.image_dir = os.path.join(self.data_dir, 'images')

    def save_poly_data(self, filename):
        """
        Save the poly data to a file
        :param filename:
        :type filename:
        :return:
        :rtype:
        """
        ioUtils.writePolyData(self.poly_data, filename)

    @property
    def poly_data_type(self):
        return self._poly_data_type

    @poly_data_type.setter
    def poly_data_type(self, value):
        self._poly_data_type = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value):
        self._data_dir = value
        self._dataset_structure = SceneStructure(self._data_dir)

    @property
    def dataset_structure(self):
        return self._dataset_structure

    @property
    def image_dir(self):
        return self._image_dir

    @image_dir.setter
    def image_dir(self, value):
        self._image_dir = value

    @property
    def kinematics_pose_data(self):
        """
        Of type CameraPoses
        :return:
        """
        return self._kinematics_pose_data

    @kinematics_pose_data.setter
    def kinematics_pose_data(self, value):
        self._kinematics_pose_data = CameraPoses(value)
        self._reconstruction_to_world = self.kinematics_pose_data.get_camera_pose(0)

    @property
    def fusion_pose_data(self):
        return self._fusion_pose_data

    @fusion_pose_data.setter
    def fusion_pose_data(self, value):
        self._fusion_pose_data = value

    @property
    def camera_info(self):
        return self._camera_info

    @camera_info.setter
    def camera_info(self, value):
        self._camera_info = value

    @property
    def reconstruction_filename(self):
        return self._reconstruction_filename

    @reconstruction_filename.setter
    def reconstruction_filename(self, value):
        self._reconstruction_filename = value

    @property
    def foreground_reconstruction_filename(self):
        return self._foreground_reconstruction_filename

    @foreground_reconstruction_filename.setter
    def foreground_reconstruction_filename(self, value):
        self._foreground_reconstruction_filename = value

    @property
    def fusion_posegraph_filename(self):
        return self._fusion_posegraph_filename

    @fusion_posegraph_filename.setter
    def fusion_posegraph_filename(self, value):
        self._fusion_posegraph_filename = value
        self.fusion_pose_data = ElasticFusionCameraPoses(self._fusion_posegraph_filename)

    @property
    def config(self):
        return self._config

    @property
    def vis_obj(self):
        """
        The visualization object
        :return:
        :rtype:
        """
        return self.reconstruction_vis_obj

    @config.setter
    def config(self, value):
        self._config = value

    def load_poly_data(self):
        self.poly_data_raw = ioUtils.readPolyData(self.reconstruction_filename)
        self.poly_data = filterUtils.transformPolyData(self.poly_data_raw, self._reconstruction_to_world)
        self.crop_poly_data()

    def crop_poly_data(self):

        dim_x = self.config['crop_box']['dimensions']['x']
        dim_y = self.config['crop_box']['dimensions']['y']
        dim_z = self.config['crop_box']['dimensions']['z']
        dimensions = [dim_x, dim_y, dim_z]

        transform = director_utils.transformFromPose(self.config['crop_box']['transform'])

        # store the old poly data
        self.poly_data_uncropped = self.poly_data

        self.poly_data = director_utils.cropToBox(self.poly_data, transform, dimensions, data_type=self.poly_data_type)

    def visualize_reconstruction(self, view, point_size=None, vis_uncropped=False):
        if point_size is None:
            point_size = self.config['point_size']

        self.reconstruction_vis_obj = vis.updatePolyData(self.poly_data, 'Fusion Reconstruction',
                                                       view=view, colorByName='RGB')
        self.reconstruction_vis_obj.setProperty('Point Size', point_size)

        if vis_uncropped:
            vis_obj = vis.updatePolyData(self.poly_data_uncropped, 'Uncropped Fusion Reconstruction',
                               view=view, colorByName='RGB')
            vis_obj.setProperty('Point Size', point_size)

    def get_camera_to_world(self, idx):
        return self.fusion_pose_data.get_camera_to_world_pose(idx)


    @staticmethod
    def from_data_folder(data_folder, config=None):
        fr = FusionReconstruction()
        fr.data_dir = data_folder

        if config is None:
            config = FusionReconstruction.load_default_config()

        pose_data_filename = os.path.join(data_folder, 'images', 'pose_data.yaml')
        camera_info_filename = os.path.join(data_folder, 'images', 'camera_info.yaml')

        fr.config = config
        fr.kinematics_pose_data = utils.getDictFromYamlFilename(pose_data_filename)
        fr.camera_info = utils.getDictFromYamlFilename(camera_info_filename)
        fr.fusion_posegraph_filename = os.path.join(data_folder, 'images.posegraph')
        fr.fusion_pose_data.first_frame_to_world = transformUtils.copyFrame(fr._reconstruction_to_world)

        fr.reconstruction_filename = os.path.join(fr.data_dir, 'images.vtp')
        fr.setup()
        return fr

    @staticmethod
    def load_default_config():
        default_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'stations', 'RLG_iiwa_1',
                                           'change_detection.yaml')
        config = utils.getDictFromYamlFilename(default_config_file)
        return config

class TSDFReconstruction(FusionReconstruction):

    def __init__(self, load_foreground_mesh):
        FusionReconstruction.__init__(self)
        self.poly_data_type = "cells"
        self._load_foreground_mesh = load_foreground_mesh

    def setup(self):
        self.load_poly_data()
        self.image_dir = os.path.join(self.data_dir, 'images')

    def load_poly_data(self):
        reconstruction_filename = self.dataset_structure.fusion_reconstruction_file
        self.poly_data_raw = ioUtils.readPolyData(reconstruction_filename)
        self.poly_data = self.poly_data_raw


        if self._load_foreground_mesh:
            foreground_reconstruction_filename =\
                self.dataset_structure.foreground_fusion_reconstruction_file
            if not os.path.isfile(foreground_reconstruction_filename):
                print "Foreground mesh file doesn't exist, falling back" \
                " to cropping mesh"
                self.crop_poly_data()
            else:
                self.poly_data = ioUtils.readPolyData(foreground_reconstruction_filename)
        else:
            self.crop_poly_data()

    @property
    def fusion_pose_data(self):
        raise ValueError("TSDFReconstruction doesn't have fusion_pose_data")

    def get_image_indices(self):
        """
        Returns a list of image indices
        :return: list(int)
        :rtype:
        """
        return self.kinematics_pose_data.pose_dict.keys()

    def get_camera_to_world(self, idx):
        return self.kinematics_pose_data.get_camera_pose(idx)


    def visualize_reconstruction(self, view, vis_uncropped=False, name=None):


        if name is None:
            vis_name = "Fusion Reconstruction " + self.name
        else:
            vis_name = name

        self.reconstruction_vis_obj = vis.updatePolyData(self.poly_data, vis_name,
                                                       view=view, colorByName='RGB')

        if vis_uncropped:
            vis_obj = vis.updatePolyData(self.poly_data_raw, 'Uncropped Fusion Reconstruction',
                               view=view, colorByName='RGB')

    @staticmethod
    def from_data_folder(data_folder, config=None, name=None, load_foreground_mesh=True):
        """

        :param data_folder: The 'processed' subfolder of a top level log folder
        :type data_folder:
        :param config: YAML file containing parameters. The default file is
        change_detection.yaml. This file contains the parameters used to crop
        the fusion reconstruction and extract the foreground.
        :type config:YAML file
        :param name:
        :type name:
        :return:
        :rtype:
        """
        fr = TSDFReconstruction(load_foreground_mesh)
        fr.data_dir = data_folder

        if name is None:
            name = ""

        if config is None:
            print "no config passed in, loading default"
            config = FusionReconstruction.load_default_config()

        pose_data_filename = os.path.join(data_folder, 'images', 'pose_data.yaml')
        camera_info_filename = os.path.join(data_folder, 'images', 'camera_info.yaml')

        fr.config = config
        fr.name = name
        fr.kinematics_pose_data = utils.getDictFromYamlFilename(pose_data_filename)
        fr.camera_info = utils.getDictFromYamlFilename(camera_info_filename)
        fr.setup()

        return fr