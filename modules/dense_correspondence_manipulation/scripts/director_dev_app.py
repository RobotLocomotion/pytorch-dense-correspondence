#!/usr/bin/env directorPython

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


def load_polydata():
    filename = "/home/manuelli/code/modules/dense_correspondence_manipulation/scripts/test_reconstruction.ply"
    poly_data = ioUtils.readPolyData(filename)
    vis.showPolyData(poly_data, 'test')

def main(globalsDict):
    createApp(globalsDict)
    view = globalsDict['view']
    app = globalsDict['app']


    load_polydata()

    app.app.start(restoreWindow=True)


if __name__ == "__main__":
    main(globals())