# system
import numpy as np

# director
import director.vtkAll as vtk
import director.vtkNumpy as vnp

"""
Class to colorize the mesh for later rendering
"""

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
        self._poly_data_item.setProperty('Surface Mode', 'Surface with edges')
        self._poly_data_item._renderAllViews() # render all the views just to be safe

    @staticmethod
    def index_to_color(idx):
        """
        Converts an integer idx into a base 255 representation
        Can handle numbers up to 255**3 - 1 = 16581374

        :param idx: The integer index to convert
        :type idx: int or long
        :return:
        :rtype:
        """
        base = 2
        idx_str = np.base_repr(idx, base=base)

        # 24 because it is 3 * 8, and 255 = 2**8
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
        The color is the representation of the index in base 255.

        Note 65025 = 255**2
        :param color: (r,g,b) color representation
        :type color: list(int) with length 3
        :return: int
        :rtype:
        """


        idx = 65025 * color[0] + 255 * color[1] + color[2]
        return idx

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
    def make_vtk_color_array(num_cells):
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
            a.InsertTuple(i, MeshColorizer.index_to_color(i))

        return a