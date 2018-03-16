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

def setCameraTransform(camera, transform):
    '''Set camera transform so that view direction is +Z and view up is -Y'''
    origin = np.array(transform.GetPosition())
    axes = transformUtils.getAxesFromTransform(transform)
    camera.SetPosition(origin)
    camera.SetFocalPoint(origin+axes[2])
    camera.SetViewUp(-axes[1])