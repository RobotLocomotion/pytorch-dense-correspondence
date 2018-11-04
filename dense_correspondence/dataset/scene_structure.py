import os

import dense_correspondence_manipulation.utils.utils as utils

class SceneStructure(object):

    def __init__(self, processed_folder_dir):
        self._processed_folder_dir = processed_folder_dir

    @property
    def fusion_reconstruction_file(self):
        """
        The filepath for the fusion reconstruction
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'fusion_mesh.ply')

    @property
    def foreground_fusion_reconstruction_file(self):
        """
        The filepath for the fusion reconstruction corresponding only to the
        foreground. Note, this may not exist if you haven't done some processing
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'fusion_mesh_foreground.ply')

    @property
    def camera_info_file(self):
        """
        Full filepath for yaml file containing camera intrinsics parameters
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'images', 'camera_info.yaml')


    @property
    def camera_pose_file(self):
        """
        Full filepath for yaml file containing the camera poses
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'images', 'pose_data.yaml')

    @property
    def rendered_images_dir(self):
        return os.path.join(self._processed_folder_dir, 'rendered_images')


    def mesh_descriptors_dir(self, network_name):
        """
        Directory where we store descriptors corresponding to a particular network
        :param network_name:
        :type network_name:
        :return:
        :rtype:
        """
        return os.path.join(self._processed_folder_dir, 'mesh_descriptors', network_name)

    def mesh_cells_image_filename(self, img_idx):
        """
        Returns the full filename for the cell labels image
        :param img_idx:
        :type img_idx:
        :return:
        :rtype:
        """
        filename = utils.getPaddedString(img_idx) + "_mesh_cells.png"
        return os.path.join(self.rendered_images_dir, filename)


    def mesh_descriptors_filename(self, network_name, img_idx):
        """
        Returns the full filename for the .npz file that contains two arrays

        .npz reference https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.savez.html#numpy.savez

        D = descriptor dimension

        - cell_ids: np.array of size N, dtype=np.int64
        - cell_descriptors: np.array with np.shape = [N,D dtype = np.float64
        -
        :param img_idx:
        :type img_idx:
        :return:
        :rtype:
        """

        filename = utils.getPaddedString(img_idx) + "_mesh_descriptors.npz"
        return os.path.join(self.mesh_descriptors_dir(network_name), filename)

    def mesh_descriptor_statistics_filename(self, network_name):
        """
        Filename containing mesh descriptor statistics

        N = number of cells for which we have descriptor information

        - cell_valid: np.array of size N, dtype=np.int64. Value is the 
        index of that cell in the ply file description
        - cell_descriptor_mean: np.array with np.shape = [N,D] dtype = np.float64
        - cell_location: Location of the cell in object frame np.array with
                        np.shape = [N,3], dtype=np.float64

        :param: network_name
        :return: filename
        :rtype: str
        """
        return os.path.join(self.mesh_descriptors_dir(network_name), "mesh_descriptor_stats.npz")

    @staticmethod
    def descriptor_image_filename(img_idx):
        filename = utils.getPaddedString(img_idx) + "_descriptor_image.npy"
        return filename





