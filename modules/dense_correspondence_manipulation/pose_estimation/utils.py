# system
import numpy as np

# director
import director.vtkAll as vtk

"""
Utilities related to pose estimation using dense descriptors
"""

def get_cell_center(vtk_triangle):
    """
    Return the center of the triangle in world coordinates
    :param vtk_triangle: vtkTriangle (maybe also works for vtkCell?)
    :type vtk_triangle:
    :return: The center of the cell in world coordinates
    :rtype:
    """

    pcoords = [0]*3
    center = [0]*3
    subid = vtk.mutable(0)
    weights = [0.0] # pointer to a double for passing to vtk

    vtk_triangle.GetParametricCenter(pcoords)
    vtk_triangle.EvaluateLocation(subid, pcoords, center, weights)

    return center


def compute_cell_locations(poly_data, cell_ids):
    """
    Computes the cell locations for the given cell_ids.

    N = cell_ids.size()
    :param poly_data: the vtkPolyData
    :type poly_data: vtkPolyData
    :param cell_ids: numpy array, shape [N,]
    :type cell_ids:
    :return: numpy array of cell locations in world frame, shape is [N,3]
    :rtype:
    """

    N = cell_ids.size
    cell_locations = np.zeros([N, 3])
    for k in xrange(0, N):
        id = cell_ids[k]
        vtk_cell = poly_data.GetCell(id)
        center = get_cell_center(vtk_cell)
        cell_locations[k, :] = np.array(center)

    return cell_locations


