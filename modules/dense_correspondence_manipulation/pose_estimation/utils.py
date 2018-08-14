# system
import numpy as np

# director
import director.vtkAll as vtk
from director import vtkNumpy as vnp

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

def transform_3D_points(transform, points):
    """

    :param transform: homogeneous transform
    :type transform:  4 x 4 numpy array
    :param points:
    :type points: numpy array, [N,3]
    :return: numpy array [N,3]
    :rtype:
    """

    N = points.shape[0]
    points_homog = np.append(np.transpose(points), np.ones([1,N]), axis=0) # shape [4,N]
    transformed_points_homog = transform.dot(points_homog)

    transformed_points = np.transpose(transformed_points_homog[0:3, :]) # shape [N, 3]
    return transformed_points

def compute_landmark_transform(sourcePoints, targetPoints):

    '''
    Returns a vtkTransform for the transform sourceToTarget
    that can be used to transform the source points to the target.

    :return: vtkTransform that aligns source to target

    '''
    sourcePoints = vnp.getVtkPointsFromNumpy(np.array(sourcePoints))
    targetPoints = vnp.getVtkPointsFromNumpy(np.array(targetPoints))

    f = vtk.vtkLandmarkTransform()
    f.SetSourceLandmarks(sourcePoints)
    f.SetTargetLandmarks(targetPoints)
    f.SetModeToRigidBody()
    f.Update()

    mat = f.GetMatrix()
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.SetMatrix(mat)
    return t

