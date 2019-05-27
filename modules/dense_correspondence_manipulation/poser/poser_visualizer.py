# system
import os
import yaml
from yaml import CLoader

import director.objectmodel as om
import director.visualization as vis
from director import ioUtils

class PoserVisualizer(object):

    def __init__(self, poser_output_folder = None):
        self._clear_visualization()
        self._poser_output_folder = poser_output_folder

    @property
    def poser_output_folder(self):
        """
        The full path to the poser output folder
        :return:
        :rtype:
        """
        return self._poser_output_folder

    @poser_output_folder.setter
    def poser_output_folder(self, value):
        self._poser_output_folder = value

    def load_poser_response(self):
        """
        Load the poser_response.yaml file
        :return:
        :rtype: dict
        """

        filename = self._convert_relative_path_to_absolute("poser_response.yaml")
        return yaml.load(file(filename), Loader=CLoader)


    def _convert_relative_path_to_absolute(self, path):
        """
        Converts a path that is relative to self.poser_output_folder to an
        absolute path.

        You must ensure that self.poser_output_folder is not
        None before calling this function
        :param path:
        :type path:
        :return:
        :rtype:
        """
        if self._poser_output_folder is None:
            raise ValueError("poser_output_folder cannot be None")

        return os.path.join(self._poser_output_folder, path)

    def _clear_visualization(self):
        """
        Delete the Poser vis container, create a new one with the same name
        :return:
        :rtype:
        """
        self._poser_vis_container = om.getOrCreateContainer("Poser")
        om.removeFromObjectModel(self._poser_vis_container)
        self._poser_vis_container = om.getOrCreateContainer("Poser")


    def visualize_result(self, poser_response):
        """
        Visualizes the results of running poser

        :param poser_response:
        :type poser_response: dict loaded from poser_response.yaml file
        :return:
        :rtype:
        """

        self._object_vis_containers = dict()

        # visualize the observation
        for object_name, data in poser_response.iteritems():
            vis_dict = dict()
            self._object_vis_containers[object_name] = vis_dict
            vis_dict['container'] = om.getOrCreateContainer(object_name,
                                                            parentObj=self._poser_vis_container)


            rigid_transform = []
            template_ply = self._convert_relative_path_to_absolute(data['image_1']['save_template'])
            template_poly_data = ioUtils.readPolyData(template_ply)
            vis.updatePolyData(template_poly_data, 'template', parent=vis_dict['container'])

