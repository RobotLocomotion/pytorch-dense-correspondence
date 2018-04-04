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
from director.segmentationroutines import *
import director.filterUtils as filterUtils


import dense_correspondence_manipulation.utils.utils as utils


def transformFromPose(d):
    """
    Returns a transform from a standard encoding in dict format
    :param d:
    :return:
    """
    pos = [0]*3
    pos[0] = d['translation']['x']
    pos[1] = d['translation']['y']
    pos[2] = d['translation']['z']

    quatDict = utils.getQuaternionFromDict(d)
    quat = [0]*4
    quat[0] = quatDict['w']
    quat[1] = quatDict['x']
    quat[2] = quatDict['y']
    quat[3] = quatDict['z']

    return transformUtils.transformFromPose(pos, quat)
    
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

    # set the focal length fx
    # followed the last comment here https://stackoverflow.com/questions/18019732/opengl-vtk-setting-camera-intrinsic-parameters
    m = np.eye(4)
    m[0, 0] = 1.0 * fx / fy
    t = vtk.vtkTransform()
    t.SetMatrix(m.flatten())
    camera.SetUserTransform(t)


def setCameraInstrinsicsAsus(view):
    principalX = 320.0
    principalY = 240.0
    focalLength = 528.0
    setCameraIntrinsics(view, principalX, principalY, focalLength)


def cropToLineSegment(polyData, point1, point2, data_type='cells'):

    line = np.array(point2) - np.array(point1)
    length = np.linalg.norm(line)
    axis = line / length

    polyData = labelPointDistanceAlongAxis(polyData, axis, origin=point1, resultArrayName='dist_along_line')

    if data_type == "cells":
        return filterUtils.thresholdCells(polyData, 'dist_along_line', [0.0, length], arrayType="points")

    elif data_type == "points":
        return filterUtils.thresholdPoints(polyData, 'dist_along_line', [0.0, length])
    else:
        raise ValueError("unknown data_type = %s" %(data_type))

def cropToBox(polyData, transform, dimensions, data_type='cells'):
    '''
    dimensions is length 3 describing box dimensions
    '''
    origin = np.array(transform.GetPosition())
    axes = transformUtils.getAxesFromTransform(transform)

    for axis, length in zip(axes, dimensions):
        cropAxis = np.array(axis)*(length/2.0)
        polyData = cropToLineSegment(polyData, origin - cropAxis, origin + cropAxis)

    return polyData

#######################################################################################


