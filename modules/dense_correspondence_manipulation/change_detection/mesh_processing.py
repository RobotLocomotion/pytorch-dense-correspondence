import numpy as np
import os


from director import imageview
from director import vtkAll as vtk
from director import transformUtils
from director import visualization as vis
from director import viewbehaviors
from director import vtkNumpy as vnp
from director.debugVis import DebugData
from director.timercallback import TimerCallback
from director import ioUtils
import PythonQt


from director import mainwindowapp
from PythonQt import QtCore, QtGui

import cv2

import dense_correspondence_manipulation.change_detection.change_detection as change_detection
from dense_correspondence_manipulation.utils.constants import *
import dense_correspondence_manipulation.utils.utils as utils
import dense_correspondence_manipulation.utils.director_utils as director_utils
from dense_correspondence_manipulation.fusion.fusion_reconstruction import FusionReconstruction, TSDFReconstruction
from dense_correspondence_manipulation.mesh_processing.mesh_render import MeshColorizer


CONFIG = utils.getDictFromYamlFilename(CHANGE_DETECTION_CONFIG_FILE)

class ReconstructionProcessing(object):

    def __init__(self):
        pass

    def spawnCropBox(self, dims=None):
        if dims is None:
            dim_x = CONFIG['crop_box']['dimensions']['x']
            dim_y = CONFIG['crop_box']['dimensions']['y']
            dim_z = CONFIG['crop_box']['dimensions']['z']
            dims = [dim_x, dim_y, dim_z]

        transform = director_utils.transformFromPose(CONFIG['crop_box']['transform'])
        d = DebugData()
        d.addCube(dims, [0,0,0], color=[0,1,0])
        self.cube_vis = vis.updatePolyData(d.getPolyData(), 'Crop Cube', colorByName='RGB255')
        vis.addChildFrame(self.cube_vis)
        self.cube_vis.getChildFrame().copyFrame(transform)
        self.cube_vis.setProperty('Alpha', 0.3)

    def getCropBoxFrame(self):
        transform = self.cube_vis.getChildFrame().transform
        pos, quat = transformUtils.poseFromTransform(transform)
        print (pos,quat)


def createApp(globalsDict=None):

    from director import mainwindowapp
    from PythonQt import QtCore, QtGui

    app = mainwindowapp.construct()
    app.gridObj.setProperty('Visible', True)
    app.viewOptions.setProperty('Orientation widget', True)
    app.viewOptions.setProperty('View angle', 30)
    app.sceneBrowserDock.setVisible(True)
    app.propertiesDock.setVisible(False)
    app.mainWindow.setWindowTitle('Depth Scanner')
    app.mainWindow.show()
    app.mainWindow.resize(920,600)
    app.mainWindow.move(0,0)

    view = app.view


    globalsDict['view'] = view
    globalsDict['app'] = app



def main(globalsDict, data_folder):
    createApp(globalsDict)
    view = globalsDict['view']
    app = globalsDict['app']

    reconstruction = TSDFReconstruction.from_data_folder(data_folder, config=CONFIG)
    reconstruction.visualize_reconstruction(view, vis_uncropped=False)
    rp = ReconstructionProcessing()

    mesh_colorizer = MeshColorizer(reconstruction.vis_obj)


    globalsDict['r'] = reconstruction
    globalsDict['reconstruction'] = reconstruction
    globalsDict['rp'] = rp

    globalsDict['mc'] = mesh_colorizer
    globalsDict['app'] = app

    # rp.spawnCropBox()

    return globalsDict

    


if __name__ == '__main__':
    main(globals())