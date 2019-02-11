# system
import os
import functools

# director
from director.pointpicker import PointPicker
import director.visualization as vis
from director.debugVis import DebugData
import director.objectmodel as om
import director.transformUtils as transformUtils

# pdc
import dense_correspondence_manipulation.utils.utils as pdc_utils
from dense_correspondence_manipulation.fusion.fusion_reconstruction import TSDFReconstruction


class KeypointAnnotation(object):

    def __init__(self, view, config=None, measurement_panel=None):
        self._view = view
        self._config = config
        self.clear()
        self.clear_vis_container()
        self._point_picker = PointPicker(view, callback=self._on_point_clicked, numberOfPoints=1)
        self._point_picker.pickType = "cells"
        self._measurement_panel = measurement_panel


        self._scene_list = self._config["scene_list"]
        self.keypoint_names = self._config["keypoint_names"]

        self._current_scene_name = None

        self.load_next_scene()
        self.start()

    def load_next_scene(self):
        """
        Updates the self._current_scene_name field and loads the tool
        with that scene
        :return:
        :rtype:
        """
        if len(self._scene_list) > 0:
            self._current_scene_name = self._scene_list.pop(0)
            self.load_scene(scene_name=self._current_scene_name)
        else:
            print("No more scenes left to label, you are done!!!")

    def reload_scene(self):
        """
        Reloads the current scene (in case you messed up use this)
        :return:
        :rtype:
        """
        self.load_scene(scene_name=self._current_scene_name)

    def clear_vis_container(self):
        """
        Clear the vis container
        :return:
        :rtype:
        """
        container_name = "Keypoint Annotations"
        c = om.getOrCreateContainer(container_name)
        om.removeFromObjectModel(c)
        self._vis_container = om.getOrCreateContainer(container_name)


    def clear(self):
        """
        Clear data
        :return:
        :rtype:
        """
        self.keypoint_idx = 0
        self._data_folder = None
        self._data = dict()
        self._cache = dict()
        self._fusion_reconstruction = None
        self._keypoints = dict() # should store vis object as well
        self._scene_name = None
        self.clear_vis_container()

    def load_scene(self, scene_name=None, logs_dir=None):
        """
        Loads a scene for visualization
        :param scene_name:
        :type scene_name:
        :param logs_dir:
        :type logs_dir:
        :return:
        :rtype:
        """
        self.clear()
        print("Keypoint List:", self.keypoint_names)

        if scene_name is None:
            scene_name = "2018-05-14-22-10-53"

        if logs_dir is None:
            logs_dir = self._config["logs_dir"]

        self._scene_name = scene_name
        self._data_folder = pdc_utils.convert_data_relative_path_to_absolute_path(os.path.join(logs_dir, scene_name, "processed"))


        # load fusion reconstruction
        self._fusion_reconstruction = TSDFReconstruction.from_data_folder(self._data_folder, load_foreground_mesh=True)
        self._fusion_reconstruction.visualize_reconstruction(self._view, vis_uncropped=True, parent=self._vis_container)

        # set the alphas to 0.5 for reconstruction?


    def start(self):
        self._point_picker.start()
        # self._measurement_panel.isEnabled(True)

    def stop(self):
        self._point_picker.stop()

    def _point_picker_complete_callback(self, *pts):
        """
        Callback from PointPicker when it finishes
        :param pts:
        :type pts:
        :return:
        :rtype:
        """

        assert(len(pts) == 1)
        p = pts[0]
        self._on_point_clicked(p)

    def get_current_keypoint_idx(self):
        return len(self._keypoints.keys())

    def _on_point_clicked(self, point):
        """

        :param pts:
        :type pts:
        :return:
        :rtype:
        """


        print("point", point)
        d = DebugData()
        d.addSphere([0,0,0], radius=0.005, color=[1,0,0])
        t = transformUtils.transformFromPose(point, [1,0,0,0])
        keypoint_name = self.keypoint_names[self.get_current_keypoint_idx()]

        obj = vis.showPolyData(d.getPolyData(), keypoint_name, color=[1,0,0], parent=self._vis_container)

        obj.actor.SetUserTransform(t)
        vis.addChildFrame(obj)

        delete_annotation_func = functools.partial(self.delete_annotation, keypoint_name)
        obj.connectRemovedFromObjectModel(delete_annotation_func)


        data = dict()
        data['position'] = point
        data['vis_obj'] = obj
        self._keypoints[keypoint_name] = data

    def delete_annotation(self, keypoint_name, objModel, item):
        """
        Delete that annotation from dict
        :param keypoint_name:
        :type keypoint_name:
        :return:
        :rtype:
        """

        if keypoint_name in self._keypoints:
            print ("removing keypoint %s" %(keypoint_name))
            del self._keypoints[keypoint_name]


    def save_annotations(self, filename=None):
        """
        Go through keypoints, get position of each one and save to dict
        """

        annotation_dict = dict()
        annotation_dict['scene_name'] = self._scene_name
        annotation_dict['keypoints'] = dict()
        annotation_dict['object_type'] = self._config['object_type']
        annotation_dict['annotation_type'] = self._config["annotation_type"]


        date_time = pdc_utils.get_current_YYYY_MM_DD_hh_mm_ss()
        annotation_dict['date_time'] = date_time
        for keypoint_name in self.keypoint_names:
            if keypoint_name not in self._keypoints:
                raise ValueError("you didn't provide an annotation for keypoint %s" %(keypoint_name))

            data = self._keypoints[keypoint_name]
            position = list(data['vis_obj'].actor.GetUserTransform().GetPosition())

            annotation_dict["keypoints"][keypoint_name] = dict()
            annotation_dict["keypoints"][keypoint_name]["position"] = position


        if filename is None:
            filename = "scene_%s_date_%s.yaml" % (self._scene_name, date_time)

        annotation_filename = os.path.join(self._config['save_dir'], filename)
        annotation_filename = pdc_utils.convert_data_relative_path_to_absolute_path(annotation_filename)

        if os.path.exists(annotation_filename):
            raise ValueError("annotation file already exists, refusing to overwrite. Filename %s" %(annotation_filename))

        if not os.path.exists(os.path.dirname(annotation_filename)):
            os.makedirs(os.path.dirname(annotation_filename))

        pdc_utils.saveToYaml([annotation_dict], annotation_filename)

        print("annotation_filename", annotation_filename)
        self._annotation_dict = annotation_dict