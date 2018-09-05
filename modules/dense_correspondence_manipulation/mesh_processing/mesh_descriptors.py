# system
import torch
import numpy as np
import os
import time

# pdc
from dense_correspondence.dataset.scene_structure import SceneStructure
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.mesh_processing.mesh_render import MeshColorizer
import dense_correspondence_manipulation.pose_estimation.utils as pose_utils

# director
from director import ioUtils

# torchvision
from torchvision import transforms

class MeshDescriptors(object):
    """
    Class for computing the (mean) descriptors corresponding to each cell for
    later use in a pose estimation pipeline
    """

    def __init__(self, scene_name, dataset, dcn, network_name="default"):
        self._dataset = dataset
        self._scene_name = scene_name
        self._dcn = dcn
        self.initialize()

    def initialize(self):
        scene_full_path = self._dataset.get_full_path_for_scene(self._scene_name)
        self._scene_structure = SceneStructure(scene_full_path)

        # load the mesh to figure out how many cells it has
        self._poly_data = ioUtils.readPolyData(self._scene_structure.foreground_fusion_reconstruction_file)


    def compute_cell_descriptors(self):
        """
        Iterates through all images of a scene, computes the the descriptors
        Saves files <image_index>_mesh_descriptors.npz that have numpy arrays describing
        the descriptors for each cell in the mesh
        :return: None
        :rtype:
        """
        dcn = self._dcn
        dataset = self._dataset
        scene_name = self._scene_name

        pose_data = dataset.get_pose_data(scene_name)

        image_idxs = pose_data.keys()
        image_idxs.sort()

        num_images = len(pose_data)
        # num_images = 10 # testing only

        D = dcn.descriptor_dimension

        num_cells = self._poly_data.GetNumberOfCells()
        print "num_cells", num_cells

        # now create the torch vector to store the data
        cell_descriptors = torch.FloatTensor([num_cells, num_images, D])
        cell_visible = torch.FloatTensor([num_cells, num_images])

        logging_frequency = 50
        start_time = time.time()

        # make the mesh_descriptors dir if it doesn't already exist
        mesh_descriptors_dir = self._scene_structure.mesh_descriptors_dir
        if not os.path.isdir(mesh_descriptors_dir):
            os.makedirs(mesh_descriptors_dir)

        for counter, img_idx in enumerate(image_idxs):

            if (counter % logging_frequency) == 0:
                print "processing image %d of %d" %(counter, num_images)

            rgb_img = dataset.get_rgb_image_from_scene_name_and_idx(scene_name, img_idx)

            # note that this has already been normalized
            rgb_img_tensor = dataset.rgb_image_to_tensor(rgb_img)
            res = dcn.forward_single_image_tensor(rgb_img_tensor).data
            cell_id_img_file = self._scene_structure.mesh_cells_image_filename(img_idx)

            cells_idx_img = MeshDescriptors.load_mesh_cells_img(cell_id_img_file)


            cell_ids_tensor, cell_descriptors_tensor, cell_idx_tensor = self.compute_cell_descriptors_single_image(cells_idx_img, res)


            max_cell_idx = np.max(cells_idx_img)
            if max_cell_idx > num_cells - 1:
                raise ValueError("Ecnountered cell index %d that exceeds number of cells" %(max_cell_idx))


            mesh_descriptors_filename = self._scene_structure.mesh_descriptors_filename(img_idx)
            np.savez(mesh_descriptors_filename, cell_ids=cell_ids_tensor.cpu(),
                     cell_descriptors=cell_descriptors_tensor.cpu(),
                     cell_idx=cell_idx_tensor.cpu())




        elapsed_time = time.time() - start_time
        print "computing mesh descriptors took %d seconds" %(elapsed_time)

    def load_cell_descriptors(self):
        """
        Loads cell descriptors from file

        N = num images
        D = descriptor dimension

        Loads descriptors for the cells
        Note you should run compute_cell_descriptors BEFORE running this function
        :return: cell_visible_array, cell_descriptor_array
        :rtype: np.array of [N], np.array [N,D]
        """

        dcn = self._dcn
        dataset = self._dataset
        scene_name = self._scene_name

        pose_data = dataset.get_pose_data(scene_name)

        image_idxs = pose_data.keys()
        image_idxs.sort()

        num_images = len(pose_data)

        D = dcn.descriptor_dimension

        num_cells = self._poly_data.GetNumberOfCells()
        mesh_descriptors_dir = self._scene_structure.mesh_descriptors_dir

        print "num_cells", num_cells

        cell_descriptor_array = np.zeros([num_cells, num_images, D])
        cell_visible_array = np.zeros([num_cells, num_images], dtype=np.int64) # binary flag for whether or not that cell appeared in the image

        for counter, img_idx in enumerate(image_idxs):

            print "processing img %d of %d" %(img_idx, num_images)

            filename = self._scene_structure.mesh_descriptors_filename(img_idx)
            # instance of NpZ file class, also a dict storing the arrays
            data = np.load(filename)
            cell_descriptors = data['cell_descriptors']
            cell_ids = data['cell_ids']


            print "num cell ids loaded", cell_ids.size

            # print "np.min(cell_ids)", np.min(cell_ids)
            # print "np.max(cell_ids)", np.max(cell_ids)

            # use cell_ids as an index
            cell_visible_array[cell_ids, counter] = 1
            cell_descriptor_array[cell_ids, counter, :] = cell_descriptors


        return cell_visible_array, cell_descriptor_array


    def compute_cell_descriptors_mean(self, cell_visible_array, cell_descriptor_array,
                                      min_visible_frames = 10):
        """
        Processes the arrays to compute the mean



        :param cell_visible_array:
        :type cell_visible_array:
        :param cell_descriptor_array:
        :type cell_descriptor_array:
        :param min_visible_frames: minimum number of frames that this cell should be visible from to
        be included
        :type min_visible_frames: int
        :return: A dict with several numpy arrays
         - "cell_valid". numpy array of shape [N,]. List of cell ids which were visible in at least `min_visible_frames`
         - "cell_descriptor_mean" numpy.array of shape [N,D] with mean descriptor for each valid cell array

        :rtype: dict
        """

        return_data = dict()

        D = cell_descriptor_array.shape[2]

        # compute the mean
        cell_descriptor_sum = np.sum(cell_descriptor_array, axis=1)
        cell_visible_count = np.sum(cell_visible_array, axis=1)

        print "cell_descriptor_array.shape", cell_descriptor_array.shape
        print "cell_descriptor_sum.shape", cell_descriptor_sum.shape
        print "cell_visible_array.shape", cell_visible_array.shape
        print "cell_visible_count.shape", cell_visible_count.shape

        cell_visible_count_no_zeros = cell_visible_count
        cell_zero_idx = np.where(cell_visible_count == 0)[0]
        cell_visible_count_no_zeros[cell_zero_idx] = 1 # just so we don't divide by zero

        num_cells = cell_visible_count_no_zeros.size



        # reshape it to be [num_cells, D]
        temp = np.reshape(cell_visible_count_no_zeros, [num_cells, 1])
        cell_visible_count_no_zeros_replicated = np.repeat(temp, D, axis=1)

        cell_descriptor_mean_all = np.divide(cell_descriptor_sum, cell_visible_count_no_zeros_replicated)
        cell_valid = np.where(cell_visible_count > min_visible_frames)[0]
        cell_descriptor_mean_valid = cell_descriptor_mean_all[cell_valid] # only for valid cells


        return_data['cell_descriptor_mean'] = cell_descriptor_mean_valid

        # compute normalized value for visualization purposes
        # Each component will be in range [0,1]

        return_data['cell_valid'] = cell_valid

        print "cell_valid.size", cell_valid.size

        return return_data


    def process_cell_descriptors(self):
        """
        Processes the cell descriptors, saves the results to a file
        :return:
        :rtype:
        """
        print "processing cell descriptors"
        save_dict = dict()
        cell_visible_array, cell_descriptor_array = self.load_cell_descriptors()

        save_dict['cell_visible_array'] = cell_visible_array
        save_dict['cell_descriptor_array'] = cell_descriptor_array

        mean_data = self.compute_cell_descriptors_mean(cell_visible_array, cell_descriptor_array)
        cell_valid = mean_data['cell_valid']
        save_dict['cell_valid'] = cell_valid
        save_dict['cell_descriptor_mean'] = mean_data['cell_descriptor_mean']

        cell_locations = pose_utils.compute_cell_locations(self._poly_data, cell_valid)
        save_dict['cell_location'] = cell_locations

        print "\n\n"
        print "cell_valid.shape", save_dict["cell_valid"].shape
        print "cell_descriptor_mean.shape", save_dict['cell_descriptor_mean'].shape

        save_file = self._scene_structure.mesh_descriptor_statistics_filename()
        np.savez(save_file, **save_dict)


    @staticmethod
    def load_mesh_cells_img(filename):
        """
        Loads the image corresponding that labels which cell goes to which pixel
        :param filename: filename of the cell_id image
        :type filename:
        :return: Image of int64 where each int represents the cell id that corresponds to
        that pixel. Pixels corresponding to background are set to -1.
        :rtype: numpy.array [H,W] with dtype = int64
        """

        img_PIL = utils.load_rgb_image(filename)  # this is PIL image
        img = np.array(img_PIL)  # now it's an np array
        idx_img = MeshColorizer.rgb_img_to_idx_img(img)
        return idx_img


    def compute_cell_descriptors_single_image(self, cell_id_img, res):
        """
        :param cell_id_img: numpy array of [H,W] of mesh cell ids corresponding
        to each pixel
        :type cell_id_img:
        :param res: [H,W,D] of torch.FloatTensor, the descriptor image
        :type res:
        :return: cell_ids_tensor, cell_descriptors_tensor
        :rtype: torch.cuda.LongTensor, torch.cuda.FloatTensor
        """

        cell_id_img_flat = cell_id_img.flatten()
        valid_cell_idx_flat = np.where(cell_id_img_flat > -1)[0] # -1 corresponds to background
        _, unique_cell_idx_flat = np.unique(cell_id_img_flat, return_index=True)
        valid_and_unique_cell_idx = np.intersect1d(valid_cell_idx_flat, unique_cell_idx_flat, assume_unique=False)


        cell_idx_flat = valid_and_unique_cell_idx
        cell_ids = np.take(cell_id_img_flat, cell_idx_flat)

        num_unique_cells = cell_ids.size

        print "num unique cells", num_unique_cells


        # [H,W] has been collapsed down to one dimension
        # so we will need to flatten the torch.FloatTensor of res to index into it
        H, W = self._dcn.image_shape
        D = self._dcn.descriptor_dimension
        res = res.contiguous() # make sure memory is contiguous

        res_flat = res.view(H*W, D) # has torch.Shape([H*W,D])


        # create new torch.FloatTensor by doing an index select
        cell_ids_tensor = torch.cuda.LongTensor(cell_ids).cuda() # torch.cuda.LongTensor
        cell_idx_flat_tensor = torch.cuda.LongTensor(cell_idx_flat) # put it on the GPU

        # should have shape torch.Shape([num_unique_cels, D])
        cell_descriptors = torch.index_select(res_flat, 0, cell_idx_flat_tensor)
        return cell_ids_tensor, cell_descriptors, cell_idx_flat_tensor

