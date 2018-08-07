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





