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
from director import transformUtils
from director import filterUtils
import PythonQt


from director import mainwindowapp
from PythonQt import QtCore, QtGui

import cv2
from dense_correspondence_manipulation.change_detection.depthscanner import DepthScanner
import dense_correspondence_manipulation.utils.director_utils as director_utils
import dense_correspondence_manipulation.utils.utils as utils
import dense_correspondence_manipulation.utils.segmentation as segmentation

cameraPoseTest = ([ 7.53674834e-01, -8.55423154e-19,  6.92103873e-01], [-0.34869616,  0.6090305 ,  0.62659914, -0.33891938])

CAMERA_TO_WORLD = transformUtils.transformFromPose(cameraPoseTest[0], cameraPoseTest[1])
DEPTH_IM_RESCALE = 4000.0

CROP_BOX_DATA = dict()
CROP_BOX_DATA['dimensions'] = [0.5, 0.7, 0.4]
CROP_BOX_DATA['transform'] = transformUtils.transformFromPose([0.66757267, 0, 0.18108], [1., 0., 0., 0.])

class DepthImageVisualizer(object):
    def __init__(self, window_title='Depth Image', scale=1):
        
        self.scale = scale

        depthImageColorByRange = [0.0, 4.0]

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)
        lut.SetHueRange(0, 0.667) # red to blue
        lut.SetRange(depthImageColorByRange) # map red (near) to blue (far)
        lut.SetRampToLinear()
        lut.Build()

        self.depthScaleFilter = vtk.vtkImageShiftScale()
        self.depthScaleFilter.SetScale(scale)
        self.depthScaleFilter.SetOutputScalarTypeToUnsignedShort()

        self.depthImageLookupTable = lut
        self.imageMapToColors = vtk.vtkImageMapToColors()
        self.imageMapToColors.SetLookupTable(self.depthImageLookupTable)
        self.imageMapToColors.SetInputConnection(self.depthScaleFilter.GetOutputPort())

        self.imageView = imageview.ImageView()
        self.imageView.view.setWindowTitle(window_title)
        self.imageView.setImage(self.imageMapToColors.GetOutput())

    def setImage(self, depthImage):
        self.depthScaleFilter.SetInputData(depthImage)
        self.depthScaleFilter.Update()

        self.depthImageLookupTable.SetRange(self.depthScaleFilter.GetOutput().GetScalarRange())
        self.imageMapToColors.Update()
        self.imageView.resetCamera()

    def setImageNumpy(self, img):
        vtkImg = vnp.numpyToImageData(img)
        self.setImage(vtkImg)


class FusionReconstruction(object):
    """
    A utility class for storing information about a 3D reconstruction

    e.g. reconstruction produced by ElasticFusion
    """

    def __init__(self):
        self._crop_box_data = None
        pass


    def setup(self):
        self.load_poly_data()

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value):
        self._data_dir = value

    @property
    def pose_data(self):
        return self._pose_data

    @pose_data.setter
    def pose_data(self, value):
        self._pose_data = value

    @property
    def pose_data(self):
        return self._pose_data

    @pose_data.setter
    def pose_data(self, value):
        self._pose_data = value

    @property
    def camera_info(self):
        return self._camera_info

    @camera_info.setter
    def camera_info(self, value):
        self._camera_info = value

    @property
    def reconstruction_filename(self):
        return self._reconstruction_filename

    @reconstruction_filename.setter
    def reconstruction_filename(self, value):
        self._reconstruction_filename = value

    @staticmethod
    def from_data_folder(data_folder):
        fr = FusionReconstruction()
        fr.data_dir = data_folder

        pose_data_filename = os.path.join(data_folder, 'images', 'pose_data.yaml')
        camera_info_filename = os.path.join(data_folder, 'images', 'camera_info.yaml')

        fr.pose_data = utils.getDictFromYamlFilename(pose_data_filename)
        fr.camera_info = utils.getDictFromYamlFilename(camera_info_filename)

        fr.reconstruction_filename = os.path.join(fr.data_dir, 'reconstruction.vtp')
        fr._crop_box_data = CROP_BOX_DATA
        fr.setup()
        return fr

        # fr.mesh_filename =

    def get_camera_pose(self, idx):
        camera_pose_dict = self.pose_data[idx]['camera_to_world']
        return director_utils.transformFromPose(camera_pose_dict)

    def load_poly_data(self):
        cameraToWorld = self.get_camera_pose(0)
        self.poly_data_raw = ioUtils.readPolyData(self.reconstruction_filename)
        self.poly_data = filterUtils.transformPolyData(self.poly_data_raw, cameraToWorld)
        self.crop_poly_data()

    def crop_poly_data(self):
        if self._crop_box_data is None:
            print "_crop_box_data is empty, skipping"
            return

        dimensions = self._crop_box_data['dimensions']
        transform = self._crop_box_data['transform']

        # store the old poly data
        self.poly_data_uncropped = self.poly_data

        self.poly_data = segmentation.cropToBox(self.poly_data, transform, dimensions)

    def visualize_reconstruction(self, view, point_size=3):
        self.reconstruction_vis_obj = vis.showPolyData(self.poly_data, 'Fusion Reconstruction',
                                                       view=view, colorByName='RGB')
        self.reconstruction_vis_obj.setProperty('Point Size', point_size)


class ChangeDetection(object):

    def __init__(self, app, view, cameraIntrinsics=None, debug=True,
        data_directory=None):

        self.app = app
        self.views = dict()
        self.vis = True
        self.debug = debug
        self.views['foreground'] = view
        self.views['background'] = self.createBackgroundView()

        self.threshold = 10
        self.loadConfig()

        if cameraIntrinsics is None:
            self.cameraIntrinsics = makeDefaultCameraIntrinsics()
        else:
            self.cameraIntrinsics = cameraIntrinsics

        # set the views to use the camera intrinsics
        for _, view in self.views.iteritems():
            director_utils.setCameraIntrinsics(view, self.cameraIntrinsics, lockViewSize=True)

        self.depthScanners = dict()
        self.depthScanners['foreground'] = initDepthScanner(self.views['foreground'], widgetArea=QtCore.Qt.RightDockWidgetArea)
        self.depthScanners['background'] = initDepthScanner(self.views['background'], widgetArea=QtCore.Qt.LeftDockWidgetArea)

        
        # self.changeDetectionPointCloudView = PythonQt.dd.ddQVTKWidgetView()
        # self.changeDetectionPointCloudView.setWindowTitle('Change Detection Pointcloud')
        # dock = self.app.app.addWidgetToDock(self.changeDetectionPointCloudView, QtCore.Qt.BottomDockWidgetArea)
        # dock.setMinimumWidth(300)
        # dock.setMinimumHeight(300)

        # used for visualizing the depth image/mask coming from change detection
        # self.changeDetectionImageVisualizer = imageview.ImageView()
        # self.changeDetectionImageVisualizer.view.setWindowTitle('Depth Image Change')
        # dock = self.app.app.addWidgetToDock(self.changeDetectionImageVisualizer.view, QtCore.Qt.BottomDockWidgetArea)
        #
        # dock.setMinimumWidth(300)
        # dock.setMinimumHeight(300)

        if data_directory is not None:
            self.data_dir = data_directory
            self.loadData()

    def loadConfig(self):
        self.config = dict()
        self.config['point_size'] = 3

    def loadData(self):
        """
        Load the relevant data from the folder.

        Key pieces of data are 
        - background reconstruction
        - foreground reconstruction
        - images (depth and rgb)
        - camera pose data
        - camera intrinsics
        """
        self.image_dir = os.path.join(self.data_dir, 'images')

    @property
    def background_reconstruction(self):
        return self._background_reconstruction

    @background_reconstruction.setter
    def background_reconstruction(self, value):
        assert isinstance(value, FusionReconstruction)

        view = self.views['background']
        self._background_reconstruction = value
        self._background_reconstruction.visualize_reconstruction(view, point_size=self.config['point_size'])

    @property
    def foreground_reconstruction(self):
        return self._foreground_reconstruction

    @foreground_reconstruction.setter
    def foreground_reconstruction(self, value):
        assert isinstance(value, FusionReconstruction)
        self._foreground_reconstruction = value

        view = self.views['foreground']
        self._foreground_reconstruction.visualize_reconstruction(view, point_size=self.config['point_size'])
        

    def createBackgroundView(self):
        """
        Creates a new view for use in rendering the background scene
        """
        view = PythonQt.dd.ddQVTKWidgetView()
        view.orientationMarkerWidget().Off()

        if self.vis:
            dock = app.app.addWidgetToDock(view, QtCore.Qt.LeftDockWidgetArea)
            dock.setMinimumWidth(300)
            dock.setMinimumHeight(300)

        return view

    def updateDepthScanners(self):
        """
        Updates all the DepthScanner objects to make sure we are rendering
        against the current scene
        """
        for _, scanner in self.depthScanners.iteritems():
            scanner.update()

    @staticmethod
    def drawNumpyImage(title, img):
        """
        Draws a new image using ImageView
        """
        imageView = imageview.ImageView()
        imageView.view.setWindowTitle(title)
        imageView.showNumpyImage(img)
        imageView.show()

        return imageView

    @staticmethod
    def drawDepthImageCV2(title, img):
        cv2.imshow(title, img.squeeze()/DEPTH_IM_RESCALE)


    def setCameraTransform(self, cameraToWorld):
        """
        Sets the camera location in both foreground and background scenes

        Parameters:
            cameraToWorld: vtkTransform
                specifies the transform from camera frame to world frame
                CameraFrame is encoded as right-down-forward
        """

        for _, view in self.views.iteritems():
            director_utils.setCameraTransform(view.camera(), cameraToWorld)
            view.forceRender()

        
    def computeForegroundMask(self, visualize=True):
        """
        Computes the foreground mask. The camera location of the foreground view should
        already be set correctly. Steps of the pipeline are

        1. Set the background camera to same location as foreground camera
        2. Render depth images from foreground and background scenes
        3. Compute change detection based on pair of depth images
        """

        # assume that foreground view is already in the correct place
        view_f = self.views['foreground']
        view_b = self.views['background']

        view_f.forceRender()
        cameraToWorld = director_utils.getCameraTransform(view_f.camera())

        director_utils.setCameraTransform(view_f.camera(), cameraToWorld)
        view_f.forceRender()

        director_utils.setCameraTransform(view_b.camera(), cameraToWorld)
        view_b.forceRender()

        self.updateDepthScanners()

        # get the depth images
        # make sure to convert them to int32 since we want to avoid wrap-around issues when using
        # uint16 types
        depth_img_f = np.array(self.depthScanners['foreground'].getDepthImageAsNumpyArray(), dtype=np.int32)
        depth_img_b = np.array(self.depthScanners['background'].getDepthImageAsNumpyArray(), dtype=np.int32)

        # zero values are interpreted as max-range measurements
        depth_img_b[depth_img_b == 0] = np.iinfo(np.int32).max
        depth_img_f[depth_img_f == 0] = np.iinfo(np.int32).max

        idx, mask = ChangeDetection.computeForegroundMaskFromDepthImagePair(depth_img_f, depth_img_b, self.threshold)

        depth_img_change = mask * depth_img_f

        returnData = dict()
        returnData['idx'] = idx
        returnData['mask'] = mask
        returnData['depth_img_foreground'] = depth_img_f
        returnData['depth_img_background'] = depth_img_b
        returnData['depth_img_change'] = depth_img_change


        if visualize:
            # make sure to remap the mask to [0,255], otherwise it will be all black
            ChangeDetection.drawNumpyImage('Change Detection Mask', mask*255)

        return returnData

    @staticmethod
    def computeForegroundMaskFromDepthImagePair(img_f, img_b, threshold):

        """
        Computes the foreground mask given a pair of depth images. Depths that are closer in
        img_f compared to img_b, by at least 'threshold' are classified as foreground pixels

        Be careful with
        wraparound if you are usint uint16 type.
        """

        
        idx = (img_b - img_f) > threshold
        mask = np.zeros(np.shape(img_f))
        mask[idx] = 1
        return idx, mask

    def showCameraTransform(self):
        t = director_utils.getCameraTransform(self.views['foreground'].camera())
        vis.updateFrame(t, 'camera transform', scale=0.15)


    ##################### DEBUGGING FUNCTIONS #################
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

    def drawDebug(self):
        ChangeDetection.drawDepthImageCV2('foreground', self.data['depth_img_foreground'])
        ChangeDetection.drawDepthImageCV2('background', self.data['depth_img_background'])
        ChangeDetection.drawDepthImageCV2('change', self.data['depth_img_change'])

        cv2.imshow('mask', self.data['mask']*255.0)

    def testComputeForegroundMask(self):
        data = self.computeForegroundMask()
        depth_img_change = data['depth_img_change']
        self.img = depth_img_change
        self.data = data

    def testIdentity(self):
        view = self.views['foreground']
        cameraToWorld = director_utils.getCameraTransform(view.camera())
        director_utils.setCameraTransform(view.camera(), cameraToWorld)
        view.forceRender()

    def testGetCameraTransforms(self):
        vis.updateFrame(vtk.vtkTransform(), "origin frame", view=self.views['background'])
        for viewType, view in self.views.iteritems():
            print "\n\nviewType: ", viewType
            cameraTransform = director_utils.getCameraTransform(view.camera())
            pos, quat = transformUtils.poseFromTransform(cameraTransform)
            print "pos: ", pos
            print "quat: ", quat


            vis.updateFrame(cameraTransform, viewType + "_frame")

    def test(self):
        self.testCreateBackground()
        self.testCreateForeground()


    def testSetCameraPose(self):
        for viewType, view in self.views.iteritems():
            camera = view.camera()
            director_utils.setCameraTransform(camera, CAMERA_TO_WORLD)
            view.forceRender()


    def testMoveCameraAndGetDepthImage(self):
        self.testSetCameraPose()
        self.depthScanners['foreground'].update()
        self.img = self.getDepthImage(visualize=False)
        # self.changeDetectionImageVisualizer.setImage(self.img)

    def testDepthImageToPointcloud(self):
        ds = self.depthScanners['foreground']
        camera = ds.view.camera()
        depth_img = ds.getDepthImage()
        depth_img_raw, _, _ = ds.getDepthImageAndPointCloud()
        depth_img_numpy = ds.getDepthImageAsNumpyArray()

        print "depth min %.3f, depth max = %.3f" %(np.min(depth_img_numpy), np.max(depth_img_numpy))

        # self.changeDetectionImageVisualizer.setImage(depth_img)


        f = vtk.vtkDepthImageToPointCloud()
        f.SetCamera(camera)
        f.SetInputData(depth_img)
        f.Update()
        polyData = f.GetOutput()

        print "polyData.GetNumberOfPoints() = ", polyData.GetNumberOfPoints()
        view = self.changeDetectionPointCloudView
        vis.updatePolyData(polyData, 'depth_im_to_pointcloud')

    def testShowBackground(self):
        view = self.views['foreground']
        self.background_reconstruction.visualize_reconstruction(view)



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

def makeDefaultCameraIntrinsics():
    fx = 533.6422696034836
    cx = 319.4091030774892
    fy = 534.7824445233571
    cy = 236.4374299691866

    width = 640
    height = 480

    return director_utils.CameraIntrinsics(cx, cy, fx, fy, width, height)\

def loadDefaultBackground():
    data_folder = '/home/manuelli/code/data_volume/sandbox/kuka_scene_background'
    reconstruction = FusionReconstruction.from_data_folder(data_folder)
    return reconstruction

def loadDefaultForeground():
    data_folder = '/home/manuelli/code/data_volume/sandbox/kuka_scene_foreground'
    reconstruction = FusionReconstruction.from_data_folder(data_folder)
    return reconstruction


def main(globalsDict):
    createChangeDetectionApp(globalsDict)
    view = globalsDict['view']
    app = globalsDict['app']
    changeDetection = ChangeDetection(app, view, cameraIntrinsics=makeDefaultCameraIntrinsics())


    DEBUG = False
    if DEBUG:
        changeDetection.test()
    else:
        changeDetection.background_reconstruction = loadDefaultBackground()
        changeDetection.foreground_reconstruction = loadDefaultForeground()
        changeDetection.testShowBackground()


    globalsDict['changeDetection'] = changeDetection
    globalsDict['cd'] = changeDetection
    app.app.start(restoreWindow=False)


if __name__ == '__main__':
    main(globals())