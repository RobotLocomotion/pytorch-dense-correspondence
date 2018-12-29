# system
import os

import director.objectmodel as om
import director.visualization as vis
from director import ioUtils

class PoserVisualizer(object):

    def __init__(self):
        self._clear_visualization()

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
        :type poser_response: dict
        :return:
        :rtype:
        """

        # visualize the observation
        for object_name, data in poser_response.iteritems():
            rigid_transform = []

            template_ply = data['template']
            ioUtils.readPolyData(template_ply)
            image_name = 'image_1'