# system
import numpy as np
import os
import scipy.spatial
import sys
import math
import time
import random
import cv2

# director
from director import visualization as vis
from director.debugVis import DebugData
import director.objectmodel as om

# pdc
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
import dense_correspondence.correspondence_tools.correspondence_finder as \
    correspondence_finder
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation
from dense_correspondence_manipulation.utils.constants import *
import simple_pixel_correspondence_labeler.annotate_correspondences as annotate_correspondences


class DescriptorPoseEstimator(object):

    def __init__(self, mesh_descriptor_stats_filename):
        self._mesh_descriptor_stats_filename = mesh_descriptor_stats_filename
        self._setup_config()
        self._initialize()
        self._random_seed = 9

    def _setup_config(self):
        self._config = dict()

        # threshold for something being considered a match
        self._config['match_threshold'] = 0.15
        self._config['recursion_limit'] = int(1e4)
        self._config['downsample_factor'] = 5

    def _initialize(self):
        """
        Loads the stored data about descriptors
        :return:
        :rtype:
        """
        self._descriptor_stats = np.load(self._mesh_descriptor_stats_filename)
        self._cell_ids = self._descriptor_stats['cell_valid']
        self._cell_descriptor_mean = self._descriptor_stats['cell_descriptor_mean']
        self._cell_location = self._descriptor_stats['cell_location']

        downsample_factor = self._config['downsample_factor']
        num_valid_cells = self._cell_ids.size
        num_downsample_cells = math.floor(1.0/downsample_factor * self._cell_ids.size)

        print "num valid cells", num_valid_cells
        print "num downsample cells", num_downsample_cells
        downsample_idx = np.arange(num_downsample_cells, dtype=np.int64) * downsample_factor
        self._cell_ids = self._cell_ids[downsample_idx]
        self._cell_descriptor_mean = self._cell_descriptor_mean[downsample_idx, :]
        self._cell_location = self._cell_location[downsample_idx, :]

        sys.setrecursionlimit(self._config['recursion_limit'])
        self._debug = False

        self._build_kd_tree()

    def _build_kd_tree(self):
        """
        Builds a KD Tree for nearest neighbor lookups on cell descriptors
        :return:
        :rtype:
        """
        print "building kdtree"
        start_time = time.time()

        # note cKDTree is super slow (24 seconds to build tree) for some unknown reason
        # revisit https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/
        # for deciding which KDTree implementation to use
        self._kd_tree = scipy.spatial.KDTree(self._cell_descriptor_mean)
        elapsed = time.time() - start_time

        print "building KDTree took %.2f seconds" %elapsed

    def initialize_debug(self):
        self._scene_name = "2018-04-10-16-02-59"
        dc_source_dir = utils.getDenseCorrespondenceSourceDir()
        dataset_config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence', 'dataset', 'composite',
                                           'caterpillar_single_scene_test.yaml')
        dataset_config = utils.getDictFromYamlFilename(dataset_config_file)
        self._dataset = SpartanDataset(config=dataset_config)

        config_filename = os.path.join(dc_source_dir, 'config', 'dense_correspondence',
                                       'evaluation', 'lucas_evaluation.yaml')
        eval_config = utils.getDictFromYamlFilename(config_filename)
        default_config = utils.get_defaults_config()
        utils.set_cuda_visible_devices(default_config['cuda_visible_devices'])

        dce = DenseCorrespondenceEvaluation(eval_config)
        network_name = "caterpillar_M_background_0.500_3"
        self._dcn = dce.load_network_from_config(network_name)

        self._camera_intrinsics = self._dataset.get_camera_intrinsics(self._scene_name)
        self._camera_matrix = self._camera_intrinsics.get_camera_matrix()
        self._debug = True

        self.setup_visualization()

    @property
    def poly_data(self):
        return self._poly_data

    @poly_data.setter
    def poly_data(self, value):
        self._poly_data = value

    def estimate_pose(self, descriptor_img, rgb_img, depth_img, mask=None):
        """

        :param res: The descriptor image [H,W,D]
        :type res:
        :param mask: Foreground object mask
        :type mask:
        :return:
        :rtype:
        """

    def get_nearest_neighbor(self, descriptor):
        """
        Nearest neighbor query
        :param descriptor:
        :type descriptor:
        :return:
        :rtype:
        """
        dist, idx = self._kd_tree.query(descriptor, distance_upper_bound=self._config['match_threshold'])


        if np.isinf(dist):
            match_found = False
        else:
            match_found = True

        return match_found, dist, idx

    def setup_visualization(self):
        self._vis_container = om.getOrCreateContainer("Pose Estimation")
        self._vis_ground_truth = om.getOrCreateContainer("Ground Truth", parentObj=self._vis_container)
        self._vis_best_match = om.getOrCreateContainer("Best Match", parentObj=self._vis_container)

    def clear_vis(self):
        """
        Clear vis objects
        :return:
        :rtype:
        """
        om.removeFromObjectModel(self._vis_ground_truth)
        om.removeFromObjectModel(self._vis_best_match)
        self.setup_visualization()


    def reset_random_seed(self):
        np.random.seed(self._random_seed)
        random.seed(self._random_seed)

    def test(self, img_idx=0, uv=[145,256]):

        self.clear_vis()
        self.reset_random_seed()

        cv2.namedWindow('RGB')


        rgb, depth_PIL, mask_PIL, pose = self._dataset.get_rgbd_mask_pose(self._scene_name, img_idx)

        rgb_cv2 = annotate_correspondences.pil_image_to_cv2(rgb)
        reticle_color = (255,255,255)


        print "type(depth_PIL)", type(depth_PIL)
        # convert to numpy

        depth = np.asarray(depth_PIL) / DEPTH_IM_SCALE

        print "type(depth)", type(depth)
        mask = np.asarray(mask_PIL)

        rgb_tensor = self._dataset.rgb_image_to_tensor(rgb)
        descriptor_img = self._dcn.forward_single_image_tensor(rgb_tensor).data.cpu()

        # sample 3 random indices from masked image
        num_samples = 1
        random_pixels = correspondence_finder.random_sample_from_masked_image(mask, num_samples)

        for i in xrange(0, num_samples):
            # row = random_pixels[0][i]
            # col = random_pixels[1][i]
            # uv = [col, row]
            row, col = uv[1], uv[0]
            p_depth = depth[row, col]
            descriptor = descriptor_img[row, col]



            # get position in world frame
            pos_in_world = correspondence_finder.pinhole_projection_image_to_world_coordinates(uv, p_depth,
                                                                                               self._camera_matrix, pose)

            # plot position in world frame in director
            pt_poly_data = DescriptorPoseEstimator.make_point(pos_in_world, color=[0,1,0])
            name = "Point %d" % i
            vis.showPolyData(pt_poly_data, name, parent=self._vis_ground_truth, color=[0, 1, 0])

            # look up nearest neighbor
            match_found, dist, idx = self.get_nearest_neighbor(descriptor)


            if match_found:
                best_match_pos = self._cell_location[idx, :]
                best_match_poly_data = DescriptorPoseEstimator.make_point(best_match_pos)
                name = "Best match %d" %i
                vis.showPolyData(best_match_poly_data, name, parent=self._vis_best_match, color=[0,0,1])
            else:
                print "NO MATCH FOUND"
                print "dist", dist
                print "i", i

            if self._debug:
                print "\n i = ", i
                print "match found", match_found
                print "dist", dist
                print "descriptor", descriptor

                annotate_correspondences.draw_reticle(rgb_cv2, uv[0], uv[1], reticle_color)


                if match_found:
                    error = np.linalg.norm(best_match_pos - pos_in_world)
                    print "best_match_error", error
                    print "best match descriptor"



        cv2.imshow('RGB', rgb_cv2)


    def test2(self):
        uv = [0,0]
        z = 1.0
        correspondence_finder.pinhole_projection_image_to_world(uv, z, self._camera_matrix)


    @staticmethod
    def make_point(position, color=[0,1,0], radius=0.01):
        d = DebugData()
        d.addSphere(position, radius=radius, color=color)
        return d.getPolyData()
