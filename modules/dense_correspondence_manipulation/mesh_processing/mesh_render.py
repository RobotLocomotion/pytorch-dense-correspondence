# system
import numpy as np
import cv2
import os
from sklearn.preprocessing import scale

# director
import director.vtkAll as vtk
import director.vtkNumpy as vnp
from director import mainwindowapp
import director.visualization as vis
from director import screengrabberpanel as sgp

# pdc
from dense_correspondence_manipulation.fusion.fusion_reconstruction import FusionReconstruction, TSDFReconstruction
from dense_correspondence.dataset.scene_structure import SceneStructure
import dense_correspondence_manipulation.utils.director_utils as director_utils
import dense_correspondence_manipulation.utils.utils as utils

"""
Class to colorize the mesh for later rendering
"""

SQUARED_256 = 65536

class MeshColorizer(object):

    def __init__(self, poly_data_item):
        self._poly_data_item = poly_data_item
        self._poly_data = poly_data_item.polyData


    def add_colors_to_mesh(self):
        """
        Adds the colors the mesh by creating the array and adding it to the
        CellData for the poly data
        :return:
        :rtype:
        """

        num_cells = self._poly_data.GetNumberOfCells()
        color_array = self.make_vtk_color_array(num_cells)
        array_name = 'cell colors'
        color_array.SetName(array_name)
        self._poly_data.GetCellData().AddArray(color_array)
        self._poly_data.GetCellData().SetActiveScalars(array_name)

        self._poly_data_item.mapper.ScalarVisibilityOn()
        # self._poly_data_item.setProperty('Surface Mode', 'Surface with edges')
        self._poly_data_item._renderAllViews() # render all the views just to be safe



    @staticmethod
    def index_to_color(idx):
        """
        Converts an integer (idx+1) into a base 256 representation
        Can handle numbers up to 256**3 - 1 = 16777215

        We don't want to use color (0,0,0) since that is reserved for the background

        Color is assumed to be in (r,g,b) format

        :param idx: The integer index to convert
        :type idx: int or long
        :return:
        :rtype:
        """
        base = 2
        idx_str = np.base_repr(idx+1, base=base)

        # 24 because it is 3 * 8, and 256 = 2**8
        idx_str = str(idx_str).zfill(24)

        r_str = idx_str[0:8]
        g_str = idx_str[8:16]
        b_str = idx_str[16:24]

        rgb = (int(r_str, base), int(g_str, base), int(b_str, base))

        return rgb


    @staticmethod
    def color_to_index(color):
        """
        Converts a color (r,g,b) with r,g,b \in [0,255] back to an index
        The color is the representation of the index in base 256.


        :param color: (r,g,b) color representation
        :type color: list(int) with length 3
        :return: int
        :rtype:
        """


        idx = SQUARED_256 * color[0] + 256 * color[1] + color[2]
        idx -= - 1 #subtract one to account for the fact that the background is idx 0
        return idx

    @staticmethod
    def rgb_img_to_idx_img(rgb_image):
        """
        Note background will have index -1 now

        :param rgb_image: [H,W,3] image in (r,g,b) format
        :type rgb_image: numpy.ndarray
        :return: numpy.array [H,W] with dtype = uint64
        :rtype:
        """

        # cast to int64 to avoid overflow
        idx_img = np.array(rgb_image, dtype=np.int64)
        idx_img = idx_img[:,:,0] * SQUARED_256 + idx_img[:,:,1]*256 + idx_img[:,:,2]
        idx_img -= 1 # subtract 1 because previously the background was 0
        return idx_img

    @staticmethod
    def make_color_array(num_cells):
        """
        Makes a color array with the given number of rows
        :param num_cells:
        :type num_cells:
        :return:
        :rtype:
        """

        a = np.zeros([num_cells, 3], dtype=np.uint8)
        for i in xrange(0, num_cells):
            a[i,:] = np.array(MeshColorizer.index_to_color(i))

        return a

    @staticmethod
    def make_vtk_color_array(num_cells, debug=True):
        """
        Makes a color array with the given number of rows
        :param num_cells:
        :type num_cells:
        :return: vtkUnsignedCharacterArray
        :rtype:
        """
        a = vtk.vtkUnsignedCharArray()
        a.SetNumberOfComponents(3)
        a.SetNumberOfTuples(num_cells)

        for i in xrange(0, num_cells):

            color = MeshColorizer.index_to_color(i)
            a.InsertTuple(i, color)

        return a


class DescriptorMeshColor(object):

    def __init__(self, poly_data_item, processed_data_folder):
        self._poly_data_item = poly_data_item
        self._poly_data = self._poly_data_item.polyData
        self._scene_structure = SceneStructure(processed_data_folder)

        descriptor_stats_file = self._scene_structure.mesh_descriptor_statistics_filename()
        self._descriptor_stats = np.load(descriptor_stats_file)
        self._num_cells = self._poly_data_item.polyData.GetNumberOfCells()

    def color_mesh_using_descriptors(self, std_dev_scale_factor=2.0):
        """
        Add colors to mesh using computed mean descriptors
        :return:
        :rtype:
        """

        descriptor_stats_file = self._scene_structure.mesh_descriptor_statistics_filename()
        descriptor_stats = np.load(descriptor_stats_file)

        # normalize data
        num_cells = self._poly_data_item.polyData.GetNumberOfCells()
        cell_valid = descriptor_stats['cell_valid']
        self._cell_valid = cell_valid
        print "cell_valid\n", cell_valid
        cell_descriptor_mean_all = descriptor_stats['cell_descriptor_mean']

        D = cell_descriptor_mean_all.shape[1]
        if not D == 3:
            raise ValueError("descriptor dimension must be 3, it is %d" %D)


        cell_descriptor_mean_valid = cell_descriptor_mean_all[cell_valid, :]

        # now the data has zero mean, unit std_dev
        cell_descriptor_norm = scale(cell_descriptor_mean_valid, axis=0)


        cell_colors = 1/std_dev_scale_factor * cell_descriptor_norm
        cell_colors = np.clip(cell_colors, -1, 1)
        cell_colors = (cell_colors + 1)/2 * 255 # now it is in range [0,255]
        cell_colors = cell_colors.astype(np.int64) # should now be ints in [0,255]

        cell_colors_all = np.zeros([num_cells, 3], dtype=np.int64)
        cell_colors_all[cell_valid, :] = cell_colors


        cell_colors_vtk = DescriptorMeshColor.make_vtk_color_array(cell_colors_all)
        array_name = 'descriptor cell colors'
        cell_colors_vtk.SetName(array_name)
        self._poly_data.GetCellData().AddArray(cell_colors_vtk)
        self._poly_data.GetCellData().SetActiveScalars(array_name)
        self._poly_data_item.mapper.ScalarVisibilityOn()

    def color_mesh_using_clipped_descriptors(self):

        cell_valid = self._descriptor_stats['cell_valid']
        print "num valid cells", cell_valid.size
        #
        # descriptor_stats_file = self._scene_structure.mesh_descriptor_statistics_filename()
        # descriptor_stats = np.load(descriptor_stats_file)
        #
        # # normalize data
        # num_cells = self._poly_data_item.polyData.GetNumberOfCells()
        # cell_valid = descriptor_stats['cell_valid']
        # self._cell_valid = cell_valid
        # print "cell_valid\n", cell_valid
        # cell_descriptor_mean_all = descriptor_stats['cell_descriptor_mean']
        #
        # D = cell_descriptor_mean_all.shape[1]
        # if not D == 3:
        #     raise ValueError("descriptor dimension must be 3, it is %d" % D)
        #
        # cell_descriptor_mean_valid = cell_descriptor_mean_all[cell_valid, :]
        #
        # # now the data has zero mean, unit std_dev
        # cell_descriptor_norm = scale(cell_descriptor_mean_valid, axis=0)
        #
        # cell_colors = 1 / std_dev_scale_factor * cell_descriptor_norm
        # cell_colors = np.clip(cell_colors, -1, 1)
        # cell_colors = (cell_colors + 1) / 2 * 255  # now it is in range [0,255]
        # cell_colors = cell_colors.astype(np.int64)  # should now be ints in [0,255]
        #
        # cell_colors_all = np.zeros([num_cells, 3], dtype=np.int64)
        # cell_colors_all[cell_valid, :] = cell_colors
        #
        # cell_colors_vtk = DescriptorMeshColor.make_vtk_color_array(cell_colors_all)
        # array_name = 'descriptor cell colors'
        # cell_colors_vtk.SetName(array_name)
        # self._poly_data.GetCellData().AddArray(cell_colors_vtk)
        # self._poly_data.GetCellData().SetActiveScalars(array_name)
        # self._poly_data_item.mapper.ScalarVisibilityOn()

    @staticmethod
    def make_vtk_color_array(color_array_np):
        """
        Converts a numpy array of colors to a vtk array that can be used for cells
        :param color_array_np: [num_cells, 3] of int64 in range [0,255]
        :type color_array_np:
        :return:
        :rtype:
        """
        a = vtk.vtkUnsignedCharArray()
        a.SetNumberOfComponents(3)
        num_cells = color_array_np.shape[0]
        a.SetNumberOfTuples(num_cells)

        for i in xrange(0, num_cells):
            color = color_array_np[i,:].tolist()
            a.InsertTuple(i, color)

        return a



class MeshRender(object):

    def __init__(self, view, view_options, fusion_reconstruction, data_folder):
        """
        app is the director app
        :param view: app.view
        :type view:
        :param self._view_options: app.self._view_options
        :type self._view_options:
        """

        self._view = view
        self._view_options = view_options
        self._fusion_reconstruction = fusion_reconstruction
        self._poly_data_item = self._fusion_reconstruction.vis_obj

        # list of vis objects to control the lighting on
        self._vis_objects = []
        self._vis_objects.append(self._poly_data_item)


        self._data_folder = data_folder
        self._dataset_structure = SceneStructure(data_folder)

        self.initialize()

    @property
    def poly_data_item(self):
        return self._poly_data_item

    def initialize(self):
        """
        Visualizes the fusion reconstruction.
        Sets the camera intrinsics etc.
        :return:
        :rtype:
        """
        self.set_camera_intrinsics()


    def colorize_mesh(self):
        # colorize the mesh
        self._mesh_colorizer = MeshColorizer(self._poly_data_item)
        self._mesh_colorizer.add_colors_to_mesh()



    def set_camera_intrinsics(self):
        """
        Sets the camera intrinsics, and locks the view size
        :return:
        :rtype:
        """

        self._camera_intrinsics = utils.CameraIntrinsics.from_yaml_file(self._dataset_structure.camera_info_file)
        director_utils.setCameraIntrinsics(self._view, self._camera_intrinsics, lockViewSize=True)

    def set_camera_transform(self, camera_transform):
        """
        Sets the camera transform and forces a render
        :param camera_transform: vtkTransform
        :type camera_transform:
        :return:
        :rtype:
        """
        director_utils.setCameraTransform(self._view.camera(), camera_transform)
        self._view.forceRender()

    def render_images(self):
        """
        Render images of the colorized mesh
        Creates files processed/rendered_images/000000_mesh_cells.png
        :return:
        :rtype:
        """
        self.colorize_mesh()
        self.disable_lighting()

        image_idx_list = self._fusion_reconstruction.get_image_indices()
        num_poses = len(image_idx_list)


        for counter, idx in enumerate(image_idx_list):
            print "Rendering mask for pose %d of %d" % (counter + 1, num_poses)
            camera_to_world = self._fusion_reconstruction.get_camera_to_world(idx)
            self.set_camera_transform(camera_to_world)
            filename = self._dataset_structure.mesh_cells_image_filename(idx)

            img, img_vtk = self.render_image(filename=filename)


    def disable_lighting(self):
        """
        Disables the lighting.
        Sets the background to black (i.e. (0,0,0))
        :return:
        :rtype:
        """
        self._view_options.setProperty('Gradient background', False)
        self._view_options.setProperty('Orientation widget', False)
        self._view_options.setProperty('Background color', [0, 0, 0])

        self._view.renderer().TexturedBackgroundOff()

        for obj in self._vis_objects:
            obj.actor.GetProperty().LightingOff()

        self._view.forceRender()

    def render_image(self, filename=None):
        """
        Make sure you call disable_lighting() BEFORE calling this function
        :return:
        :rtype:
        """

        if filename is not None:
            shouldWrite = True
        else:
            shouldWrite = False

        img_vtk = sgp.saveScreenshot(self._view, filename, shouldRender=True, shouldWrite=shouldWrite)

        img = vnp.getNumpyFromVtk(img_vtk, arrayName='ImageScalars')
        assert img.dtype == np.uint8

        img.shape = (img_vtk.GetDimensions()[1], img_vtk.GetDimensions()[0], 3)
        img = np.flipud(img)

        # if filename is not None:
        #     cv2.imwrite(filename, img)

        return img, img_vtk

    def test(self):
        camera_to_world = self._fusion_reconstruction.get_camera_to_world(0)
        self.set_camera_transform(camera_to_world)

        self.disable_lighting()
        img, img_vtk = self.render_image()

        num_cells = self._poly_data_item.polyData.GetNumberOfCells()
        print "num_cells", num_cells

        idx_img = MeshColorizer.rgb_img_to_idx_img(img)
        print "np.max(idx_img)", np.max(idx_img)

        return img, idx_img

    def test2(self):

        camera_to_world = self._fusion_reconstruction.get_camera_to_world(0)
        self.set_camera_transform(camera_to_world)

        self.disable_lighting()
        img, img_vtk = self.render_image()

        red_invalid = np.logical_and(img[:,:,0] > 0, img[:,:,0] < 100)
        idx_red_invalid = np.where(red_invalid == True)
        print "idx_red_invalid[0].size", idx_red_invalid[0].size
        print "val", img[idx_red_invalid[0][0], idx_red_invalid[1][0]]

        return img, idx_red_invalid

    def test3(self):
        self.disable_lighting()
        img, img_vtk = self.render_image()
        red_invalid = np.logical_and(img[:, :, 0] > 0, img[:, :, 0] < 100)
        idx_red_invalid = np.where(red_invalid == True)
        print "idx_red_invalid[0].size", idx_red_invalid[0].size
        print "val", img[idx_red_invalid[0][0], idx_red_invalid[1][0]]
        return img





    @staticmethod
    def from_data_folder(data_folder, config=None):
        """
        Creates the director app
        Creates a MeshRender object from the given folder

        :param data_folder:
        :type data_folder:
        :type config:
        :return:
        :rtype:
        """

        # dict to store objects that are created
        obj_dict = dict()

        # create the director app
        # make sure we disable anti-aliasing

        vis.setAntiAliasing(False)
        app = mainwindowapp.construct()
        app.gridObj.setProperty('Visible', False)
        app.viewOptions.setProperty('Orientation widget', False)
        app.viewOptions.setProperty('View angle', 30)
        app.sceneBrowserDock.setVisible(False)
        app.propertiesDock.setVisible(False)
        app.mainWindow.setWindowTitle('Mesh Rendering')

        app.mainWindow.show()
        app.mainWindow.resize(920, 600)
        app.mainWindow.move(0, 0)

        foreground_reconstruction = TSDFReconstruction.from_data_folder(data_folder, load_foreground_mesh=True)
        foreground_reconstruction.visualize_reconstruction(app.view)

        mesh_render = MeshRender(app.view, app.viewOptions, foreground_reconstruction, data_folder)


        obj_dict['app'] = app
        obj_dict['foreground_reconstruction'] = foreground_reconstruction
        obj_dict['mesh_render'] = mesh_render

        return obj_dict