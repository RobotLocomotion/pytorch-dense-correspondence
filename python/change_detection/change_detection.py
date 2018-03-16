from director import imageview
from director import vtkAll as vtk
from director import transformUtils
from director import visualization as vis
from director import viewbehaviors
from director import vtkNumpy as vnp
from director.debugVis import DebugData
from director.timercallback import TimerCallback
import PythonQt
import numpy as np

from director import mainwindowapp
from PythonQt import QtCore, QtGui

import cv2
from depthscanner import DepthScanner
import director_utils

cameraPoseTest = ([ 7.53674834e-01, -8.55423154e-19,  6.92103873e-01], [-0.34869616,  0.6090305 ,  0.62659914, -0.33891938])

cameraToWorld = transformUtils.transformFromPose(cameraPoseTest[0], cameraPoseTest[1])

class ChangeDetection(object):

    def __init__(self, app, view):
        self.app = app
        self.views = dict()
        self.vis = True
        self.views['foreground'] = view
        self.views['background'] = self.createBackgroundView()


        self.depthScanners = dict()
        self.depthScanners['foreground'] = initDepthScanner(self.views['foreground'], widgetArea=QtCore.Qt.RightDockWidgetArea)
        self.depthScanners['background'] = initDepthScanner(self.views['background'], widgetArea=QtCore.Qt.LeftDockWidgetArea)

        

    def createBackgroundView(self):
        view = PythonQt.dd.ddQVTKWidgetView()
        view.orientationMarkerWidget().Off()

        if self.vis:
            dock = app.app.addWidgetToDock(view, QtCore.Qt.LeftDockWidgetArea)
            dock.setMinimumWidth(300)
            dock.setMinimumHeight(300)

        return view

    def testCreateBackground(self):
        d = DebugData()
        d.addCube((1,1,0.1), (0,0,0), color=[1,1,1])
        self.backgroundPolyData = d.getPolyData()

        if self.vis:
            view = self.views['background']
            vis.updatePolyData(self.backgroundPolyData, 'background', view=view, colorByName='RGB255')
            view.resetCamera()

    def testCreateForeground(self):
        d = DebugData()
        dims = np.array([0.2,0.2,0.2])
        center = (0,0,dims[2]/2.0)
        d.addCube(dims, center, color=[0,1,0])

        d.addPolyData(self.backgroundPolyData)
        

        self.foregroundPolyData = d.getPolyData()

        if self.vis:
            view=self.views['foreground']
            vis.updatePolyData(self.foregroundPolyData, 'foreground', view=view, colorByName='RGB255')

            view.resetCamera()

    def getDepthImage(self, scene="foreground"):
        return self.depthScanners[scene].getDepthImageAsNumpyArray()

    def test(self):
        self.testCreateBackground()
        self.testCreateForeground()

    def testRenderDepthImages(self):
        pass

    def testSetCameraPose(self):
        for viewType, view in self.views.iteritems():
            camera = view.camera()
            director_utils.setCameraTransform(camera, cameraToWorld)
            view.forceRender()
    
    def setCameraPose():
        pass

# create app
def createChangeDetectionApp(globalsDict=None):

    from director import mainwindowapp
    from PythonQt import QtCore, QtGui

    app = mainwindowapp.construct()
    app.gridObj.setProperty('Visible', True)
    app.viewOptions.setProperty('Orientation widget', False)
    app.viewOptions.setProperty('View angle', 30)
    app.sceneBrowserDock.setVisible(False)
    app.propertiesDock.setVisible(False)
    app.mainWindow.setWindowTitle('Depth Scanner')
    app.mainWindow.show()
    app.mainWindow.resize(920,600)
    app.mainWindow.move(0,0)

    view = app.view
    view.setParent(None)
    mdiArea = QtGui.QMdiArea()
    app.mainWindow.setCentralWidget(mdiArea)
    subWindow = mdiArea.addSubWindow(view)
    subWindow.setMinimumSize(300,300)
    subWindow.setWindowTitle('Camera image')
    subWindow.resize(640, 480)
    mdiArea.tileSubWindows()

    globalsDict['view'] = view
    globalsDict['app'] = app

def initDepthScanner(view, widgetArea=QtCore.Qt.RightDockWidgetArea):
    depthScanner = DepthScanner(view)
    depthScanner.update()

    dock = app.app.addWidgetToDock(depthScanner.imageView.view, widgetArea)
    dock.setMinimumWidth(300)
    dock.setMinimumHeight(300)

    dock = app.app.addWidgetToDock(depthScanner.pointCloudView, widgetArea)
    dock.setMinimumWidth(300)
    dock.setMinimumHeight(300)

    return depthScanner


def main(globalsDict):
    createChangeDetectionApp(globalsDict)
    view = globalsDict['view']
    app = globalsDict['app']
    changeDetection = ChangeDetection(app, view)
    changeDetection.test()

    globalsDict['changeDetection'] = changeDetection
    globalsDict['cd'] = changeDetection
    app.app.start(restoreWindow=False)


if __name__ == '__main__':
    main(globals())