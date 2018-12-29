# system
import os
import subprocess
import time
import shutil
import copy

# director
import director.objectmodel as om
from director import transformUtils
from director import visualization as vis
from director import ioUtils

# pdc
import dense_correspondence_manipulation.poser.utils as poser_utils
import dense_correspondence_manipulation.utils.utils as pdc_utils



class PoserClient(object):
    """
    Python client for interfacing with poser. Includes visualization using director
    """

    def __init__(self, use_director=True, visualize=True):

        self._use_director = use_director
        self._visualize = visualize

        self._poser_vis_container = None

    def _setup_visualization(self):
        """
        Initialize the visualization by creating the appropriate containers
        :return:
        :rtype:
        """
        assert (self._use_director == True)
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


    def test_poser_on_example_data(self):
        """
        Runs poser using the supplied example data.
        Visualizes the result using director
        :return:
        :rtype:
        """
        example_data_dir = poser_utils.poser_don_example_data_dir()


        output_dir = PoserClient.get_poser_output_dir()

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        poser_request_file = os.path.join(example_data_dir, 'poser_request.yaml')

        poser_request = pdc_utils.getDictFromYamlFilename(poser_request_file)
        poser_request['shoe_1']['template'] = os.path.join(example_data_dir, 'descriptor_template.pcd')

        poser_request['shoe_1']['image_1']['descriptor_img'] = os.path.join(example_data_dir, '000000_descriptor.npy')

        poser_request['shoe_1']['image_1']['rgb_img'] = os.path.join(example_data_dir, '000000_rgb.png')

        poser_request['shoe_1']['image_1']['depth_img'] = os.path.join(example_data_dir, '000000_depth.png')

        # don't use a mask for now
        del poser_request['shoe_1']['image_1']['mask_img']

        poser_request['shoe_1']['image_1']['visualize'] = 0

        poser_request['shoe_1']['image_1']['save_processed_cloud'] = os.path.join(output_dir, 'processed_world_cloud.ply')


        PoserClient.copy_input_files_to_output_dir(poser_request, output_dir)
        poser_response, output_dir = self.run_poser(poser_request, output_dir)

        if self._visualize:
            self.visualize_result(poser_response)

    def run_poser(self, poser_request, output_dir):
        """
        Saves the poser_request to tmp directory
        Runs poser
        Returns the resulting yaml file
        :param poser_request:
        :type poser_request:
        :return:
        :rtype:
        """

        poser_request_file = os.path.join(output_dir, "poser_request.yaml")
        print "poser_request\n", poser_request
        pdc_utils.saveToYaml(poser_request, poser_request_file, flush=True)


        poser_response_file = os.path.join(output_dir, 'poser_response.yaml')

        cmd = "%s %s %s" %(poser_utils.poser_don_executable_filename(), poser_request_file,
                           poser_response_file)
        print "cmd: ", cmd

        start_time = time.time()
        process = subprocess.Popen(cmd, shell=True)
        print "started subprocess, waiting for it to finish"
        process.wait()
        elapsed = time.time() - start_time
        print "poser took %.2f seconds" %(elapsed)

        poser_response = pdc_utils.getDictFromYamlFilename(poser_response_file)
        poser_response = PoserClient.convert_response_to_relative_paths(poser_response, output_dir)
        pdc_utils.saveToYaml(poser_response, poser_response_file)

        return poser_response, output_dir

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

            template_pcd = data['template']
            ioUtils.readPolyData(template_pcd)
            image_name = 'image_1'

    @staticmethod
    def get_poser_output_dir():
        """
        Return poser output dir
        :return:
        :rtype:
        """
        return os.getenv("POSER_OUTPUT_DIR")

    @staticmethod
    def poser_response_file():
        return os.path.join(PoserClient.get_poser_output_dir(), "poser_response.yaml")


    @staticmethod
    def vtk_transform_from_poser_response(rigid_transform_matrix_vec):
        """
        Converts a 4 x 4 matrix expressed in column major order to vtkTransform
        :param rigid_transform_matrix_vec:
        :type rigid_transform_matrix_vec:
        :return:
        :rtype:
        """
        pass

    @staticmethod
    def copy_input_files_to_output_dir(poser_request, output_dir):
        """
        Copy all files in poser_request to the output directory
        :param poser_request:
        :type poser_request:
        :param output_dir:
        :type output_dir:
        :return:
        :rtype:
        """
        print "copying files"

        def move_file(filename):
            dst = os.path.join(output_dir, os.path.basename(filename))
            shutil.copy(filename, dst)
            return dst

        files_to_copy = ['descriptor_img', 'rgb_img', 'depth_img']
        for obj_name, data in poser_request.iteritems():
            dst = os.path.join(output_dir, )
            data['template'] = move_file(data['template'])
            for key in files_to_copy:
                data['image_1'][key] = move_file(data['image_1'][key])


    @staticmethod
    def convert_response_to_relative_paths(poser_response, output_dir=None):

        if output_dir is None:
            raise ValueError("must specify output dir")

        pr_new = copy.deepcopy(poser_response)

        files_to_copy = ['descriptor_img', 'rgb_img', 'depth_img', 'save_processed_cloud']
        for obj_name, data in pr_new.iteritems():
            data['template'] = os.path.relpath(data['template'], output_dir)
            for key in files_to_copy:
                data['image_1'][key] = os.path.relpath(data['image_1'][key], output_dir)


        return pr_new













