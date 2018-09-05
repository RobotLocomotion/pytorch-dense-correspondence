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
import director.vtkAll as vtk
from director.debugVis import DebugData
import director.objectmodel as om
from director.fieldcontainer import FieldContainer
from director import transformUtils
from vtk.util import numpy_support

# pdc
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
import dense_correspondence.correspondence_tools.correspondence_finder as \
    correspondence_finder
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation
from dense_correspondence_manipulation.utils.constants import *
import simple_pixel_correspondence_labeler.annotate_correspondences as annotate_correspondences
import dense_correspondence_manipulation.pose_estimation.utils as pose_utils
from dense_correspondence_manipulation.fusion.fusion_reconstruction import TSDFReconstruction


class PoseEstimationData(FieldContainer):
    """
    Class for storing data for single run of pose estimation
    """
    fields = {'kd_tree': None,
              'rgb_PIL':None,
              'depth_PIL': None,
              'mask_PIL': None,
              'camera_pose': None, # 4 x 4 numpy transform
              'depth': None, # depth img, numpy float64
              'mask': None, # foreground mask, numpy
              'descriptor_img': None, # [H,W,D] numpy array
              'rgb': None, # [H,W,3]
              'random_pixels': None, # in row col format
              'pixel_descriptors': None,
              'sparse_distance_matrix': None,
              }

    def __init__(self, **kwargs):
        FieldContainer.__init__(self, **PoseEstimationData.fields)
        self._set_fields(**kwargs)


class DescriptorPoseEstimator(object):

    def __init__(self, mesh_descriptor_stats_filename):
        self._mesh_descriptor_stats_filename = mesh_descriptor_stats_filename
        self._setup_config()
        self._initialize()
        self._random_seed = 9

    def _setup_config(self):
        self._config = dict()

        # threshold for something being considered a match
        self._config['descriptor_match_threshold'] = 0.15
        self._config['kd_tree_eps'] = 0.1 # approximate threshold for kdtree search
        self._config['recursion_limit'] = int(1e4)
        self._config['downsample_factor'] = 10
        self._config['num_image_pixels'] = 5000 # num image pixels to use to sample
        self._config['num_RANSAC_pixels'] = 100
        self._config['inlier_threshold'] = 0.03 # 3cm inlier threshold


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
        # default scene_name to use
        self._scene_name = "2018-04-10-16-02-59"

        # another scene with caterpillar in different configuration
        self._scene_name_alternate = "2018-04-16-14-25-19"

        dc_source_dir = utils.getDenseCorrespondenceSourceDir()
        dataset_config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence', 'dataset', 'composite',
                                           'caterpillar_only_9.yaml')
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


        self.setup_visualization()
        self._debug_data = dict()


    @property
    def poly_data(self):
        return self._poly_data

    @poly_data.setter
    def poly_data(self, value):
        self._poly_data = value

    @property
    def view(self):
        return self._view

    @view.setter
    def view(self, value):
        self._view = value

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
        dist, idx = self._kd_tree.query(descriptor, distance_upper_bound=self._config['descriptor_match_threshold'])


        if np.isinf(dist):
            match_found = False
        else:
            match_found = True

        return match_found, dist, idx

    def get_nearest_neighbor_set(self, descriptor):
        """
        Return the all points within a radius of the query point
        :param descriptor: np.array with shape [D,]
        :type descriptor:
        :return:
        :rtype:
        """

        dist, idx = self._kd_tree.query(descriptor, distance_upper_bound=self._config['descriptor_match_threshold'])

        if np.isinf(dist):
            match_found = False
        else:
            match_found = True

        return match_found, dist, idx


    def setup_visualization(self):
        self._vis_container = om.getOrCreateContainer("Pose Estimation")
        self._vis_ground_truth = om.getOrCreateContainer("Ground Truth", parentObj=self._vis_container)
        self._vis_best_match = om.getOrCreateContainer("Best Match", parentObj=self._vis_container)
        self._poly_data_item = vis.updatePolyData(self.poly_data, 'model', parent=self._vis_container)
        self._poly_data_item.setProperty('Visible', False)
        self.view.forceRender()

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

    def setup_pose_estimation_data(self, scene_name, img_idx):
        rgb_PIL, depth_PIL, mask_PIL, pose = self._dataset.get_rgbd_mask_pose(scene_name, img_idx)

        pose_est_data = PoseEstimationData()
        pose_est_data.rgb_PIL = rgb_PIL
        pose_est_data.mask_PIL = mask_PIL
        pose_est_data.depth_PIL = depth_PIL
        pose_est_data.camera_pose = pose

        # compute the descriptor image
        depth = np.asarray(depth_PIL) / DEPTH_IM_SCALE
        mask = np.asarray(mask_PIL)

        rgb_tensor = self._dataset.rgb_image_to_tensor(rgb_PIL)
        descriptor_img = self._dcn.forward_single_image_tensor(rgb_tensor).data.cpu()
        pose_est_data.descriptor_img = descriptor_img

        pose_est_data.depth = depth
        pose_est_data.mask = mask

        # randomly sample given number of pixels from the mask
        # build kd_tree
        random_pixels = correspondence_finder.random_sample_from_masked_image(mask, self._config['num_image_pixels'])

        pose_est_data.random_pixels = random_pixels

        # get descriptors for these
        pixel_descriptors = descriptor_img[random_pixels[0], random_pixels[1], :]
        pose_est_data.pixel_descriptors = pixel_descriptors


        # build KdTree
        start_time = time.time()
        kd_tree = scipy.spatial.KDTree(pixel_descriptors)
        elapsed = time.time() - start_time

        if self._debug:
            print "building kd_tree took %.2f seconds" % elapsed

        pose_est_data.kd_tree = kd_tree

        # compute sparse distance matrix
        # start_time = time.time()
        # sparse_distance_matrix = kd_tree.sparse_distance_matrix(self._kd_tree, self._config['descriptor_match_threshold'])
        # elapsed = time.time() - start_time
        # pose_est_data.sparse_distance_matrix = sparse_distance_matrix
        #
        # if self._debug:
        #     print "computing sparse_distance_matrix took %.2f seconds" % elapsed


        return pose_est_data


    def generate_initial_hypotheses(self, kd_tree_model, depth, camera_pose, random_pixels,
                                    random_pixel_descriptors, num_hypothesis=100,
                                    num_points_per_pose=3,
                                    visualize=False):
        """
        Generate a set of initial hypothesis using 3 points to initialize the transform
        via the Kabsch algorithm
        :param num_hypothesis:
        :type num_hypothesis:
        :return:
        :rtype:
        """



        num_samples = num_hypothesis * num_points_per_pose
        num_pixels = len(random_pixels[0])

        sample_indices = np.random.choice(np.arange(0, num_pixels), size=num_samples, replace=False)
        sample_descriptors = random_pixel_descriptors[sample_indices, :]

        start_time = time.time()
        kd_tree_samples = scipy.spatial.KDTree(sample_descriptors)
        ball_tree_results = kd_tree_samples.query_ball_tree(kd_tree_model, self._config['descriptor_match_threshold'], eps=self._config['kd_tree_eps'])
        elapsed = time.time() - start_time



        if self._debug:
            print "ball search took %.2f seconds" % elapsed


        valid_hypothesis = []

        for k in xrange(0, num_hypothesis):
            num_matches_found = 0
            # 3D points projected from RGBD image into world frame
            target_points = np.zeros([num_points_per_pose, 3])

            # 3D model points coming from descriptor matches
            source_points = np.zeros([num_points_per_pose, 3])

            for i in xrange(0, num_points_per_pose):
                sample_idx = num_points_per_pose*k + i
                pixel_idx = sample_indices[sample_idx]
                uv = (random_pixels[1][pixel_idx], random_pixels[0][pixel_idx])
                pixel_depth = depth[uv[1], uv[0]]
                pos_in_world = \
                    correspondence_finder.pinhole_projection_image_to_world_coordinates(uv, pixel_depth,
                                                                                                   self._camera_matrix,
                                                                                                   camera_pose)

                target_points[i,:] = pos_in_world

                # look up nearest neighbor
                descriptor = random_pixel_descriptors[pixel_idx]
                match_found = len(ball_tree_results[sample_idx]) > 0
                # match_found, dist, idx = self.get_nearest_neighbor(descriptor)

                if match_found:
                    num_matches_found += 1
                    cell_idx = random.sample(ball_tree_results[sample_idx], 1)[0]
                    source_points[i, :] = self._cell_location[cell_idx, :]
                else:
                    # break out if no match found for one of the three points
                    break


            # run Kabsch algorithm
            if num_matches_found == num_points_per_pose:
                pose_hypothesis = pose_utils.compute_landmark_transform(source_points, target_points)
                valid_hypothesis.append(pose_hypothesis)



        if self._debug:
            print "found %d valid pose hypothesis" % len(valid_hypothesis)

        return valid_hypothesis

    def get_pixel_samples(self, mask, depth, camera_pose, num_pixels):
        random_pixels = correspondence_finder.random_sample_from_masked_image(mask, num_pixels)
        pixels_uv = (random_pixels[1], random_pixels[0])
        depth_vec = depth[pixels_uv[1], pixels_uv[0]]

        pixel_3D_camera_frame = correspondence_finder.pinhole_projection_image_to_camera_coordinates_vectorized(pixels_uv, depth_vec, self._camera_matrix)


        pixel_3D_world_frame = pose_utils.transform_3D_points(camera_pose, pixel_3D_camera_frame)

        return pixels_uv, depth_vec, pixel_3D_camera_frame, pixel_3D_world_frame



    def evaluate_and_refine_hypothesis(self, kd_tree_model, current_hypotheses_list, pixels_3D, pixel_descriptors, inlier_threshold):
        """
        Evaluate the energy function for each pose hypothesis

        N = num pixel samples, typically 100
        M = num model points
        K = num_pose hypothesis
        D = descriptor dimension

        :param kd_tree_model: kdtree for the model points
        :type kd_tree_model:
        :param current_hypotheses_list:
        :type current_hypotheses_list: list of 4x4 numpy arrays
        :param pixels_3D: list of 3D pixel locations
        :type pixels_3D: numpy.array shape [N, 3]
        :param pixel_descriptors: array of pixel descriptors
        :type pixel_descriptors: np.array shape [N, D]
        :param inlier_threshold: inlier threshold for 3D distance
        :type inlier_threshold: float
        :return:
        :rtype:
        """

        cell_locations = self._cell_location
        num_pixels = pixels_3D.shape[0]
        num_hypotheses = len(current_hypotheses_list)

        # build the ball_tree_query
        start_time = time.time()
        kd_tree_samples = scipy.spatial.KDTree(pixel_descriptors)
        ball_tree_results = kd_tree_samples.query_ball_tree(kd_tree_model, self._config['descriptor_match_threshold'],
                                                            eps=self._config['kd_tree_eps'])
        elapsed = time.time() - start_time

        if self._debug:
            print "ball search took %.2f seconds" % elapsed

        # restrict only to pixels that actually have a match
        pixels_with_match = []
        for i in xrange(0, num_pixels):
            if len(ball_tree_results[i]) > 0:
                pixels_with_match.append(i)

        pixels_with_match = np.array(pixels_with_match, dtype=np.int64)
        if self._debug:
            print "num pixels with match", pixels_with_match.size

        pixels_3D = pixels_3D[pixels_with_match, :]
        pixel_descriptors = pixel_descriptors[pixels_with_match, :]
        ball_tree_results = [ball_tree_results[i] for i in pixels_with_match]
        num_pixels = len(pixels_with_match)

        # transform pixel locations to 3D locations in world frame
        best_match_cell_idx = np.zeros([num_hypotheses, num_pixels], dtype=np.int64)
        best_match_cell_dist = np.zeros([num_hypotheses, num_pixels])
        num_hypotheses = len(current_hypotheses_list)

        for k in xrange(0, num_hypotheses):
            transform = current_hypotheses_list[k]
            model_points = pose_utils.transform_3D_points(transform, self._cell_location)

            if self._debug:
                print "\n"
                print "model_points.shape", model_points.shape

            # evaluate energy for given pose hypothesis
            for i in xrange(0, num_pixels):
                # nearest neighbors of model

                # have already dealt with case where M_i is empty above
                # so we can assume it is non-empty
                M_i_cell_idx = ball_tree_results[i]
                M_i = model_points[M_i_cell_idx, :] # shape [|M_i|, 3]

                M_i_dists = np.linalg.norm(M_i - pixels_3D[i], axis=1)



                min_idx = np.argmin(M_i_dists)
                best_match_cell_idx[k,i] = M_i_cell_idx[min_idx]
                best_match_cell_dist[k,i] = M_i_dists[min_idx]

                if self._debug:
                    print "M_i.shape", M_i.shape
                    print "M_i_dists.shape", M_i_dists.shape
                    print "min_idx", min_idx
                    print "min_dist", M_i_dists[min_idx]


        # now evaluate the energies
        energy = np.ones(best_match_cell_dist.shape)
        inliers_idx = np.abs(best_match_cell_dist) < inlier_threshold
        energy[inliers_idx] = 0
        energy_vec = np.sum(energy, axis=1)



        # preempt the half of the pose samples with the largest energies
        # sort the energies, from low to high
        sorted_idx = np.argsort(energy_vec)


        num_hypotheses_to_keep = int(max(1, num_hypotheses/2))
        hypotheses_to_keep = sorted_idx[:num_hypotheses_to_keep]

        if self._debug:
            print "energy_vec.shape", energy_vec.shape
            print "energy_vec", energy_vec
            print "sorted idx", sorted_idx
            print "hypotheses to keep", hypotheses_to_keep

        pruned_hypothesis = [current_hypotheses_list[i] for i in hypotheses_to_keep]

        refined_hypotheses = []

        for idx, k in enumerate(hypotheses_to_keep):


            # resolve Kabsch algorithm that transformed inlier set of model
            # cells/vertices to the set of 3D pixel locations
            # source_points = inlier cell locations
            # target points = pixel 3D locations
            source_points = cell_locations[best_match_cell_idx[k,:], :]

            if self._debug:
                print "\n\n"
                print "best_match_cell_idx[k,:].shape", best_match_cell_idx[k,:].shape
                print "cell_locations.shape", cell_locations.shape
                print "source_points.shape", source_points.shape
                print "pixels_3D.shape", pixels_3D.shape

            transform_vtk = pose_utils.compute_landmark_transform(source_points, pixels_3D)
            transform_numpy = transformUtils.getNumpyFromTransform(transform_vtk)
            refined_hypotheses.append(transform_numpy)

            if idx == 0 and self._debug:
                source_point_poly_data = DescriptorPoseEstimator.make_points(source_points)
                vis.updatePolyData(source_point_poly_data, "source points", color=[1,0,0],
                                   parent=self._vis_container)

                target_poly_data = DescriptorPoseEstimator.make_points(pixels_3D)

                vis.updatePolyData(target_poly_data, "target points", color=[0, 1, 0],
                                   parent=self._vis_container)

        return refined_hypotheses, pruned_hypothesis


    def test(self, img_idx=0, uv=None, random=True, num_points=3):
        """
        Test the 3 point registration + Kabsch algorithm for pose initialization
        :param img_idx:
        :type img_idx:
        :param uv:
        :type uv:
        :param random:
        :type random:
        :return:
        :rtype:
        """

        self.clear_vis()

        if not random:
            self.reset_random_seed()

        cv2.namedWindow('RGB')


        rgb, depth_PIL, mask_PIL, pose = self._dataset.get_rgbd_mask_pose(self._scene_name, img_idx)
        rgb_cv2 = annotate_correspondences.pil_image_to_cv2(rgb)
        reticle_color = (255,255,255)



        num_matches_found = 0
        # 3D points projected from RGBD image into world frame
        target_points = np.zeros([num_points, 3])

        # 3D model points coming from descriptor matches
        source_points = np.zeros([num_points, 3])


        # convert to numpy
        depth = np.asarray(depth_PIL) / DEPTH_IM_SCALE
        mask = np.asarray(mask_PIL)

        rgb_tensor = self._dataset.rgb_image_to_tensor(rgb)
        descriptor_img = self._dcn.forward_single_image_tensor(rgb_tensor).data.cpu()

        # sample 3 random indices from masked image
        random_pixels = correspondence_finder.random_sample_from_masked_image(mask, num_points)

        for i in xrange(0, num_points):
            row = random_pixels[0][i]
            col = random_pixels[1][i]
            uv = [col, row]
            p_depth = depth[row, col]
            descriptor = descriptor_img[row, col]

            # get position in world frame
            pos_in_world = correspondence_finder.pinhole_projection_image_to_world_coordinates(uv, p_depth,
                                                                                               self._camera_matrix, pose)

            target_points[i,:] = pos_in_world


            # plot position in world frame in director
            pt_poly_data = DescriptorPoseEstimator.make_point(pos_in_world, color=[0,1,0])
            name = "Point %d" % i
            vis.showPolyData(pt_poly_data, name, parent=self._vis_ground_truth, color=[0, 1, 0])

            # look up nearest neighbor
            match_found, dist, idx = self.get_nearest_neighbor(descriptor)


            if match_found:
                best_match_pos = self._cell_location[idx, :]
                source_points[i, :] = best_match_pos
                best_match_poly_data = DescriptorPoseEstimator.make_point(best_match_pos)
                name = "Best match %d" %i
                vis.showPolyData(best_match_poly_data, name, parent=self._vis_best_match, color=[0,0,1])
                num_matches_found += 1
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

        # run the Kabsch algorithm
        object_to_world = pose_utils.compute_landmark_transform(source_points, target_points)
        self.object_to_world = object_to_world

        # set transform for poly data
        self._poly_data_item.actor.SetUserTransform(object_to_world)
        self._poly_data_item.setProperty('Visible', True)



    def test_find_match(self, img_idx=0, uv=None, random=False):
        """
        Finds the best match on the mesh for the pixel in the RGB image at location
        uv. If uv is None then we will randomly choose a pixel on the masked image.

        :param img_idx:
        :type img_idx:
        :param uv:
        :type uv:
        :param random:
        :type random:
        :return:
        :rtype:
        """

        scene_name = self._scene_name_alternate

        self.clear_vis()
        if not random:
            self.reset_random_seed()

        cv2.namedWindow('RGB')
        rgb, depth_PIL, mask_PIL, pose = self._dataset.get_rgbd_mask_pose(scene_name, img_idx)
        rgb_cv2 = annotate_correspondences.pil_image_to_cv2(rgb)
        reticle_color = (255, 255, 255)

        # convert to numpy
        depth = np.asarray(depth_PIL) / DEPTH_IM_SCALE
        mask = np.asarray(mask_PIL)

        rgb_tensor = self._dataset.rgb_image_to_tensor(rgb)
        descriptor_img = self._dcn.forward_single_image_tensor(rgb_tensor).data.cpu()

        # sample 3 random indices from masked image
        num_samples = 1
        if uv is None:
            random_pixels = correspondence_finder.random_sample_from_masked_image(mask, num_samples)
            uv = (random_pixels[1][0], random_pixels[0],[0])

        for i in xrange(0, num_samples):
            row = uv[1]
            col = uv[0]
            p_depth = depth[row, col]
            descriptor = descriptor_img[row, col]

            # get position in world frame
            pos_in_world = correspondence_finder.pinhole_projection_image_to_world_coordinates(uv, p_depth,
                                                                                               self._camera_matrix,
                                                                                               pose)

            # plot position in world frame in director
            pt_poly_data = DescriptorPoseEstimator.make_point(pos_in_world, color=[0, 1, 0])
            name = "Point %d" % i
            vis.showPolyData(pt_poly_data, name, parent=self._vis_ground_truth, color=[0, 1, 0])

            # look up nearest neighbor
            match_found, dist, idx = self.get_nearest_neighbor(descriptor)

            if match_found:
                best_match_pos = self._cell_location[idx, :]
                best_match_poly_data = DescriptorPoseEstimator.make_point(best_match_pos)
                name = "Best match %d" % i
                vis.showPolyData(best_match_poly_data, name, parent=self._vis_best_match, color=[0, 0, 1])
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


    def test_pose_estimation(self, num_initial_hypotheses=100, random=False, img_idx=0,
                             scene_name="2018-04-16-14-25-19", visualize=True):



        if not random:
            self.reset_random_seed()

        start_time = time.time()
        data = self.setup_pose_estimation_data(scene_name, img_idx)




        # draw cv2 image
        cv2.namedWindow('RGB')
        rgb_cv2 = annotate_correspondences.pil_image_to_cv2(data.rgb_PIL)
        cv2.imshow('RGB', rgb_cv2)

        initial_hypothesis = self.generate_initial_hypotheses(self._kd_tree, data.depth,
                                                              data.camera_pose,
                                                              data.random_pixels,
                                                              data.pixel_descriptors,
                                                              num_hypothesis=num_initial_hypotheses)

        self.initial_hypothesis = initial_hypothesis


        current_hypotheses = []
        for vtk_transform in initial_hypothesis:
            current_hypotheses.append(transformUtils.getNumpyFromTransform(vtk_transform))

        prev_hypothesis = current_hypotheses


        counter = 1
        while len(current_hypotheses) > 1:

            print "iteration %d, num hypothesis %d" % (counter, len(current_hypotheses))
            # do one step of the preemptive RANSAC + pose refinement
            pixels_uv, pixels_depth, pixels_3D_camera_frame, pixels_3D_world_frame = \
                self.get_pixel_samples(data.mask, data.depth, data.camera_pose,
                                       num_pixels=self._config['num_RANSAC_pixels'])

            pixel_descriptors = data.descriptor_img[pixels_uv[1], pixels_uv[0], :]

            refined_hypotheses, pruned_hypotheses = \
                self.evaluate_and_refine_hypothesis(self._kd_tree, current_hypotheses,
                                                pixels_3D_world_frame,
                                                pixel_descriptors,
                                                self._config['inlier_threshold'])

            counter += 1
            current_hypotheses = refined_hypotheses


        elapsed = time.time() - start_time


        print "test_pose_estimation took %.2f seconds" %elapsed

        self._debug_data['current_hypothesis'] = current_hypotheses
        self._debug_data['prev_hypothesis'] = pruned_hypotheses


        if visualize:
            self.debug_vis(vis_prev_hypotheses=False)

            full_path_for_scene = self._dataset.get_full_path_for_scene(scene_name)
            config = utils.getDictFromYamlFilename(CHANGE_DETECTION_CONFIG_FILE)
            fusion_reconstruction = TSDFReconstruction.from_data_folder(full_path_for_scene,
                                                                        config=config,
                                                                        load_foreground_mesh=False)
            scene_poly_data = fusion_reconstruction.poly_data
            reconstruction_obj = vis.updatePolyData(scene_poly_data, "reconstruction", parent=self._vis_container,
                               color=[0,0.5,0])

            reconstruction_obj.setProperty('Alpha', 0.4)





    def visualize_current_hypothesis(self, idx=0):
        transform = self._debug_data['current_hypothesis'][idx]
        vtk_transform = transformUtils.getTransformFromNumpy(transform)

        # set transform for poly data
        self._poly_data_item.actor.SetUserTransform(vtk_transform)
        self._poly_data_item.setProperty('Visible', True)
        self.view.forceRender()

    def visualize_pose_estimate(self, transform, name="pose estimate", parent=None):
        if not isinstance(transform, vtk.vtkTransform):
            transform = transformUtils.getTransformFromNumpy(transform)

        poly_data_copy = vtk.vtkPolyData()
        poly_data_copy.CopyStructure(self.poly_data)

        obj = vis.updatePolyData(poly_data_copy, name, parent=parent)
        obj.actor.SetUserTransform(transform)
        obj.setProperty('Visible', True)
        self.view.forceRender()

    def debug_vis(self, vis_prev_hypotheses=False):

        container = om.getOrCreateContainer("debug", parentObj=self._vis_container)
        om.removeFromObjectModel(container)
        container = om.getOrCreateContainer("debug", parentObj=self._vis_container)

        current_hypothesis = self._debug_data['current_hypothesis']
        curr_hypothesis_container = om.getOrCreateContainer('current hypothesis', parentObj=container)

        for idx, transform in enumerate(self._debug_data['current_hypothesis']):
            self.visualize_pose_estimate(transform, "Pose %d" %idx, parent=curr_hypothesis_container)



        if vis_prev_hypotheses:
            prev_hypothesis = self._debug_data['prev_hypothesis']
            prev_hypothesis_container = om.getOrCreateContainer('prev hypothesis', parentObj=container)

            for idx, transform in enumerate(self._debug_data['prev_hypothesis']):
                self.visualize_pose_estimate(transform, "Prev Pose %d" %idx, parent=prev_hypothesis_container)



    def test_pixel_3D(self, pixel_idx=0):
        self.reset_random_seed()
        img_idx = 0
        data = self.setup_pose_estimation_data(img_idx)

        pixels_uv, pixels_depth, pixels_3D_camera_frame, pixels_3D_world_frame = \
            self.get_pixel_samples(data.mask, data.depth, data.camera_pose,
                                   num_pixels=5)

        uv = (pixels_uv[0][pixel_idx], pixels_uv[1][pixel_idx])
        pixel_depth = data.depth[uv[1], uv[0]]
        print "pixel_depth", pixel_depth
        camera_pose = data.camera_pose



        pos_in_world = correspondence_finder.pinhole_projection_image_to_world_coordinates(uv, pixel_depth,
                                                                                                   self._camera_matrix,
                                                                                                   camera_pose)

        point = DescriptorPoseEstimator.make_point(pos_in_world)
        vis.updatePolyData(point, "single point projection", parent=self._vis_container,
                         color=[0,1,0])

        print "pos_in_world", pos_in_world

        pos_in_world_2 = pixels_3D_world_frame[pixel_idx,:]

        print "pixels_3D_camera_frame[0,:]", pixels_3D_camera_frame[pixel_idx,:]
        print "pixels_3D_world_frame.shape", pixels_3D_world_frame.shape
        print "pixels_3D_world_frame[0,:]", pos_in_world_2

        point = DescriptorPoseEstimator.make_point(pos_in_world_2)
        vis.updatePolyData(point, "vectorized projection", parent=self._vis_container,
                         color=[1, 0, 0])

        print "pixels_3D_camera_frame", pixels_3D_camera_frame
        print "pixels_3D_world_frame", pixels_3D_world_frame


    @staticmethod
    def make_point(position, color=[0,1,0], radius=0.01):
        d = DebugData()
        d.addSphere(position, radius=radius, color=color)
        return d.getPolyData()

    @staticmethod
    def make_points(points):
        d = DebugData()
        num_rows = points.shape[0]
        for i in xrange(num_rows):
            p = points[i,:]
            d.addPolyData(DescriptorPoseEstimator.make_point(p))

        return d.getPolyData()

