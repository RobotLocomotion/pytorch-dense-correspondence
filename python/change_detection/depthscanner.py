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

import cv2

def vtk_image_to_numpy_array(vtk_image):
      w, h, _ = vtk_image.GetDimensions()
      scalars = vnp.getNumpyFromVtk(vtk_image, vtk_image.GetPointData().GetScalars().GetName())
      img = scalars.reshape(h, w, -1)
      img = np.flipud(img) # you might need to flip the image rows, vtk has a difference row ordering convention that numpy/opencv

      return img, scalars


def computeDepthImageAndPointCloud(depthBuffer, colorBuffer, camera):
    '''
    Input args are an OpenGL depth buffer and color buffer as vtkImageData objects,
    and the vtkCamera instance that was used to render the scene.  The function returns
    returns a depth image and a point cloud as vtkImageData and vtkPolyData.
    '''
    depthImage = vtk.vtkImageData()
    pts = vtk.vtkPoints()
    ptColors = vtk.vtkUnsignedCharArray()
    vtk.vtkDepthImageUtils.DepthBufferToDepthImage(depthBuffer, colorBuffer, camera, depthImage, pts, ptColors)

    pts = vnp.numpy_support.vtk_to_numpy(pts.GetData())
    polyData = vnp.numpyToPolyData(pts, createVertexCells=True)
    ptColors.SetName('rgb')
    polyData.GetPointData().AddArray(ptColors)

    return depthImage, polyData


class DepthScanner(object):

    def __init__(self, view):
        self.view = view

        self.depthImage = None
        self.pointCloudObj = None
        self.renderObserver = None

        self.windowToDepthBuffer = vtk.vtkWindowToImageFilter()
        self.windowToDepthBuffer.SetInput(self.view.renderWindow())
        self.windowToDepthBuffer.SetInputBufferTypeToZBuffer()
        self.windowToDepthBuffer.ShouldRerenderOff()

        self.windowToColorBuffer = vtk.vtkWindowToImageFilter()
        self.windowToColorBuffer.SetInput(self.view.renderWindow())
        self.windowToColorBuffer.SetInputBufferTypeToRGB()
        self.windowToColorBuffer.ShouldRerenderOff()

        useBackBuffer = False
        if useBackBuffer:
            self.windowToDepthBuffer.ReadFrontBufferOff()
            self.windowToColorBuffer.ReadFrontBufferOff()

        self.initDepthImageView()
        self.initPointCloudView()

        self._block = False
        self.singleShotTimer = TimerCallback()
        self.singleShotTimer.callback = self.update

    def getDepthBufferImage(self):
        return self.windowToDepthBuffer.GetOutput()

    def getDepthImage(self):
        return self.depthScaleFilter.GetOutput()

    def getDepthImageAsNumpyArray(self):
        vtk_image = self.getDepthImage()
        return vtk_image_to_numpy_array(vtk_image)

    def getColorBufferImage(self):
        return self.windowToColorBuffer.GetOutput()

    def updateBufferImages(self):
        for f in [self.windowToDepthBuffer, self.windowToColorBuffer]:
            f.Modified()
            f.Update()

    def initDepthImageView(self):

        self.depthImageColorByRange = [0.0, 4.0]

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        lut.SetHueRange(0, 0.667) # red to blue
        lut.SetRange(self.depthImageColorByRange) # map red (near) to blue (far)
        lut.SetRampToLinear()
        lut.Build()

        self.depthScaleFilter = vtk.vtkImageShiftScale()
        self.depthScaleFilter.SetScale(1000)
        self.depthScaleFilter.SetOutputScalarTypeToUnsignedShort()

        self.depthImageLookupTable = lut
        self.imageMapToColors = vtk.vtkImageMapToColors()
        self.imageMapToColors.SetLookupTable(self.depthImageLookupTable)
        self.imageMapToColors.SetInputConnection(self.depthScaleFilter.GetOutputPort())

        self.imageView = imageview.ImageView()
        self.imageView.view.setWindowTitle('Depth image')
        self.imageView.setImage(self.imageMapToColors.GetOutput())

    def initPointCloudView(self):
        self.pointCloudView = PythonQt.dd.ddQVTKWidgetView()
        self.pointCloudView.setWindowTitle('Pointcloud')
        self.pointCloudViewBehaviors = viewbehaviors.ViewBehaviors(self.pointCloudView)

    def update(self):

        if not self.renderObserver:
            def onEndRender(obj, event):
                if self._block:
                    return
                if not self.singleShotTimer.singleShotTimer.isActive():
                    self.singleShotTimer.singleShot(0)
            self.renderObserver = self.view.renderWindow().AddObserver('EndEvent', onEndRender)

        self._block = True
        self.view.forceRender()
        self.updateBufferImages()
        self._block = False

        depthImage, polyData = computeDepthImageAndPointCloud(self.getDepthBufferImage(), self.getColorBufferImage(), self.view.camera())

        self.depthScaleFilter.SetInputData(depthImage)
        self.depthScaleFilter.Update()

        self.depthImageLookupTable.SetRange(self.depthScaleFilter.GetOutput().GetScalarRange())
        self.imageMapToColors.Update()
        self.imageView.resetCamera()
        #self.imageView.view.render()

        if not self.pointCloudObj:
            self.pointCloudObj = vis.showPolyData(polyData, 'point cloud', colorByName='rgb', view=self.pointCloudView)
        else:
            self.pointCloudObj.setPolyData(polyData)
        self.pointCloudView.render()

    def test(self):
        img, scalars = self.getDepthImageAsNumpyArray()


def main(globalsDict=None):

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

    #affordanceManager = affordancemanager.AffordanceObjectModelManager(view)

    depthScanner = DepthScanner(view)
    depthScanner.update()

    dock = app.app.addWidgetToDock(depthScanner.imageView.view, QtCore.Qt.RightDockWidgetArea)
    dock.setMinimumWidth(300)
    dock.setMinimumHeight(300)

    dock = app.app.addWidgetToDock(depthScanner.pointCloudView, QtCore.Qt.RightDockWidgetArea)
    dock.setMinimumWidth(300)
    dock.setMinimumHeight(300)


    # add some test data
    def addTestData():
        d = DebugData()
        # d.addSphere((0,0,0), radius=0.5)
        d.addArrow((0,0,0), (0,0,1), color=[1,0,0])
        d.addArrow((0,0,1), (0,.5,1), color=[0,1,0])
        vis.showPolyData(d.getPolyData(), 'debug data', colorByName='RGB255')
        view.resetCamera()

    addTestData()

    # xvfb command
    # /usr/bin/Xvfb  :99 -ac -screen 0 1280x1024x16

    if globalsDict is not None:
        globalsDict.update(dict(app=app, view=view, depthScanner=depthScanner))

    app.app.start(restoreWindow=False)



if __name__ == '__main__':
    main(globals())
