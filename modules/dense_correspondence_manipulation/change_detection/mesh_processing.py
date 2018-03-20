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


class ReconstructionProcessing(object):

    def __init__(self):
        pass

    def spawnCropBox(self, dims=CROP_BOX_DATA['dimensions']):
        d = DebugData()
        d.addCube(dims, [0,0,0], color=[0,1,0])

        self.cube_vis = vis.updatePolyData(d.getPolyData(), 'Crop Cube', colorByName='RGB255')
        vis.addChildFrame(self.cube_vis)
        self.cube_vis.getChildFrame().copyFrame(change_detection.CROP_BOX_DATA['transform'])
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
    app.sceneBrowserDock.setVisible(False)
    app.propertiesDock.setVisible(False)
    app.mainWindow.setWindowTitle('Depth Scanner')
    app.mainWindow.show()
    app.mainWindow.resize(920,600)
    app.mainWindow.move(0,0)

    view = app.view


    globalsDict['view'] = view
    globalsDict['app'] = app



def main(globalsDict):
    createApp(globalsDict)
    view = globalsDict['view']
    app = globalsDict['app']

    reconstruction = change_detection.loadDefaultBackground()
    reconstruction.visualize_reconstruction(view)

    rf = change_detection.loadDefaultForeground()
    rf.visualize_reconstruction(view)

    rp = ReconstructionProcessing()
    globalsDict['r'] = reconstruction
    globalsDict['rp'] = rp
    globalsDict['rf'] = rf


    # rp.spawnCropBox()
    # vis.updatePolyData(reconstruction.poly_data_uncropped, 'Reconstruction Uncropped', colorByName='RGB')

    app.app.start(restoreWindow=True)

    


if __name__ == '__main__':
    main(globals())