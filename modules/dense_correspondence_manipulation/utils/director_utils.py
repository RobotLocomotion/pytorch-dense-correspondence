import os
import numpy as np
import scipy.misc
import matplotlib.cm as cm
import yaml

from director import vtkNumpy as vnp
from director import ioUtils
from director import vtkAll as vtk
from director import actionhandlers
from director import screengrabberpanel as sgp
from director import transformUtils
from director import visualization as vis
from director import objectmodel as om

class CameraIntrinsics(object):
    def __init__(self, cx, cy, fx, fy, width, height):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.width = width
        self.height = height

    
def getCameraTransform(camera):
    """
    Camera transform has X - right, Y - down, Z-forward
    Note that in VTK GetViewUp and ForwardDir are NOT necessarily orthogonal
    """
    focal_point = np.array(camera.GetFocalPoint())
    position = np.array(camera.GetPosition())
    view_up = np.array(camera.GetViewUp())


    forward_dir = focal_point - position
    if np.linalg.norm(forward_dir) < 1e-8:
        print "forward_dir norm was very small, setting to [1,0,0]"
        forward_dir = [1.0, 0.0, 0.0]

    up_dir = np.array(view_up)

    yaxis = -up_dir
    zaxis = forward_dir
    xaxis = np.cross(yaxis, zaxis)
    yaxis = np.cross(zaxis, xaxis)


    # normalize the axes
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis /= np.linalg.norm(zaxis)


    return transformUtils.getTransformFromAxesAndOrigin(xaxis, yaxis, zaxis, position)

def setCameraTransform(camera, transform):
    """
    Camera transform is of the Right-Down-Forward (XYZ) convention. 
    See http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node2.html
    """
    origin = np.array(transform.GetPosition())
    axes = transformUtils.getAxesFromTransform(transform)
    camera.SetPosition(origin)
    camera.SetFocalPoint(origin+axes[2])
    camera.SetViewUp(-axes[1])
    camera.Modified()

def focalLengthToViewAngle(focalLength, imageHeight):
    '''Returns a view angle in degrees that can be set on a vtkCamera'''
    return np.degrees(2.0 * np.arctan2(imageHeight/2.0, focalLength))


def viewAngleToFocalLength(viewAngle, imageHeight):
    '''Returns the focal length given a view angle in degrees from a vtkCamera'''
    return (imageHeight/2.0)/np.tan(np.radians(viewAngle/2.0))


def setCameraIntrinsics(view, cameraIntrinsics, lockViewSize=True):
    '''
    Note, call this function after setting the view dimensions
    Parameters:
        cameraIntrinsics: CameraIntrinsics objects

    '''
    cx = cameraIntrinsics.cx
    cy = cameraIntrinsics.cy
    fx = cameraIntrinsics.fx
    fy = cameraIntrinsics.fy
    imageWidth = cameraIntrinsics.width
    imageHeight = cameraIntrinsics.height

    if lockViewSize:
        view.setFixedHeight(imageHeight)
        view.setFixedWidth(imageWidth)

    # make sure view.height and cameraIntrinsics.height match
    if (view.height != imageHeight) or (view.width != imageWidth):
        raise ValueError('CameraIntrinsics image dimensions (%d, %d) must match view image dimensions (%d,%d)' %(imageWidth, imageHeight, view.width, view.height))

    wcx = -2*(cx - float(imageWidth)/2) / imageWidth
    wcy =  2*(cy - float(imageHeight)/2) / imageHeight
    viewAngle = focalLengthToViewAngle(fy, imageHeight)

    camera = view.camera()
    camera.SetWindowCenter(wcx, wcy)
    camera.SetViewAngle(viewAngle)


def setCameraInstrinsicsAsus(view):
    principalX = 320.0
    principalY = 240.0
    focalLength = 528.0
    setCameraIntrinsics(view, principalX, principalY, focalLength)

#######################################################################################

