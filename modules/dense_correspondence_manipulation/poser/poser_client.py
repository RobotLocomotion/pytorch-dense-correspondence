# system
import os
import subprocess
import time
import shutil
import copy
import cv2
import numpy as np

# pdc
import dense_correspondence_manipulation.poser.utils as poser_utils
import dense_correspondence_manipulation.utils.utils as pdc_utils
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork


class PoserClient(object):
    """
    Python client for interfacing with poser.
    """

    def __init__(self, use_director=True, visualize=True, config=None):

        self._use_director = use_director
        self._visualize = visualize

        self._poser_vis_container = None
        self._config = config
        self._template_file = None

    def load_network(self):
        """
        Loads the network
        :return:
        :rtype:
        """

        print "loading dcn"

        pdc_utils.set_default_cuda_visible_devices()

        path_to_network_params = self._config['network']['path_to_network_params']
        path_to_network_params = pdc_utils.convert_data_relative_path_to_absolute_path(path_to_network_params,
                                                                                   assert_path_exists=True)
        model_folder = os.path.dirname(path_to_network_params)

        self._dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder, model_param_file=path_to_network_params)
        self._dataset = self._dcn.load_training_dataset()  # why do we need to do this?

        print "finished loading dcn"

    @property
    def template_file(self):
        if self._template_file is None:
            return pdc_utils.convert_data_relative_path_to_absolute_path(self._config['template'])
        else:
            return self._template_file


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


    def run_on_images(self, image_data_list, output_dir=None, visualize=False):
        """

        :param image_data_list: list of dicts. Each dict contains the following fields
        - 'rgb': rgb image of type PIL.Image
        - 'depth': rgb image of type cv2 image (numpy array, dtype=uint16)
        - 'camera_to_world': itself dict of form

            camera_to_world:
              quaternion:
                w: 0.11955521666256178
                x: -0.7072223465820128
                y: 0.6767424859550267
                z: -0.16602021071908557
              translation:
                x: 0.29710354158229585
                y: -0.008081499080098517
                z: 0.8270976316822616


        :type image_data_list: list of dicts
        :return: str, path to poser_output folder
        :rtype:
        """
        if output_dir is None:
            output_dir = self.get_poser_output_dir()

        # make output directory, clear existing data if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        poser_request = dict()
        descriptor_images = dict()

        # compute the descriptor images
        for image_num, img_data in enumerate(image_data_list):
            rgb_img_tensor = self._dataset.rgb_image_to_tensor(img_data['rgb'])
            res_numpy = self._dcn.forward_single_image_tensor(rgb_img_tensor).data.cpu().numpy()


            img_data['descriptor_img'] = res_numpy
            img_data['descriptor_image_filename'] = os.path.join(output_dir, "%d_descriptor.npy" %(image_num))

            np.save(img_data['descriptor_image_filename'], res_numpy)


        object_names = ["shoe"]
        for object_name in object_names:
            poser_request[object_name] = dict()
            poser_request[object_name]['template'] = self.template_file
            for img_num, img_data in enumerate(image_data_list):
                image_num = img_num + 1
                image_key = "image_%d" %(image_num)
                
                poser_data = dict()
                file_prefix = "%s_image_%d_" %(object_name, image_num)

                rgb_img_filename = os.path.join(output_dir, file_prefix+"rgb.png")
                depth_img_filename = os.path.join(output_dir, file_prefix + "depth.png")

                img_data['rgb'].save(rgb_img_filename)
                poser_data['rgb_img'] = rgb_img_filename

                cv2.imwrite(depth_img_filename, img_data['depth'])
                poser_data['depth_img'] = depth_img_filename

                poser_data['descriptor_img'] = img_data['descriptor_image_filename']

                poser_data['save_template'] = os.path.join(output_dir, file_prefix + "template.pcd")
                poser_data['save_processed_cloud'] = os.path.join(output_dir, file_prefix + "processed_cloud.pcd")

                if visualize:
                    poser_data['visualize'] = 1
                else:
                    poser_data['visualize'] = 0

                poser_data['camera_to_world'] = img_data['camera_to_world']

                poser_request[object_name][image_key] = poser_data



        # pdc_utils.saveToYaml(poser_request, os.path.join(output_dir, "poser_request.yaml"))
        self.run_poser(poser_request, output_dir)
        return output_dir


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

        files_to_modify = ['descriptor_img', 'rgb_img', 'depth_img', 'save_processed_cloud',
                           'save_template']
        for obj_name, data in pr_new.iteritems():
            data['template'] = os.path.relpath(data['template'], output_dir)
            for key in files_to_modify:
                data['image_1'][key] = os.path.relpath(data['image_1'][key], output_dir)


        return pr_new

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

        # poser_request['shoe_1']['image_1']['save_processed_cloud'] = os.path.join(output_dir, 'processed_world_cloud.ply')

        poser_request['shoe_1']['image_1']['save_processed_cloud'] = os.path.join(output_dir,
                                                                                  'processed_world_cloud.pcd')

        # poser_request['shoe_1']['image_1']['save_template'] = os.path.join(output_dir, 'descriptor_template.ply')

        poser_request['shoe_1']['image_1']['save_template'] = os.path.join(output_dir, 'descriptor_template.pcd')


        PoserClient.copy_input_files_to_output_dir(poser_request, output_dir)
        poser_response, output_dir = self.run_poser(poser_request, output_dir)













