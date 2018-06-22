import numpy as np
import os
import time


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
from dense_correspondence_manipulation.fusion.fusion_reconstruction import FusionReconstruction, TSDFReconstruction
from dense_correspondence_manipulation.utils.constants import *

import dense_correspondence_manipulation.utils.utils as utils
import dense_correspondence_manipulation.utils.segmentation as segmentation


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




class ChangeDetection(object):

    def __init__(self, app, view, cameraIntrinsics=None, debug=True,
        data_directory=None):
        """
        This shouldn't be called directly. Should only be called via the staticmethod from_data_folder

        :param app:
        :type app:
        :param view:
        :type view:
        :param cameraIntrinsics:
        :type cameraIntrinsics:
        :param debug:
        :type debug:
        :param data_directory: The 'processed' subfolder of a top level log folder
        :type data_directory:
        """

        self.app = app
        self.views = dict()
        self.vis = True
        self.debug = debug
        self.views['foreground'] = view
        self.views['background'] = self.createBackgroundView()

        self.threshold = 10

        if cameraIntrinsics is None:
            self.cameraIntrinsics = makeDefaultCameraIntrinsics()
        else:
            self.cameraIntrinsics = cameraIntrinsics

        # set the views to use the camera intrinsics
        for _, view in self.views.iteritems():
            director_utils.setCameraIntrinsics(view, self.cameraIntrinsics, lockViewSize=True)

        self.depthScanners = dict()


        self.depthScanners['foreground'] = initDepthScanner(self.app, self.views['foreground'],
                                                            widgetArea=QtCore.Qt.RightDockWidgetArea)

        self.depthScanners['background'] = initDepthScanner(self.app, self.views['background'],
                                                            widgetArea=QtCore.Qt.LeftDockWidgetArea)

        
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

    @property
    def background_reconstruction(self):
        return self._background_reconstruction

    @background_reconstruction.setter
    def background_reconstruction(self, value):
        assert isinstance(value, FusionReconstruction)

        view = self.views['background']
        self._background_reconstruction = value
        self._background_reconstruction.name = "background"
        self._background_reconstruction.visualize_reconstruction(view)

    @property
    def foreground_reconstruction(self):
        return self._foreground_reconstruction

    @foreground_reconstruction.setter
    def foreground_reconstruction(self, value):
        assert isinstance(value, FusionReconstruction)
        self._foreground_reconstruction = value
        self._foreground_reconstruction.name = "foreground"

        view = self.views['foreground']
        self._foreground_reconstruction.visualize_reconstruction(view)
        

    def createBackgroundView(self):
        """
        Creates a new view for use in rendering the background scene
        """
        view = PythonQt.dd.ddQVTKWidgetView()
        view.orientationMarkerWidget().Off()

        if self.vis:
            dock = self.app.app.addWidgetToDock(view, QtCore.Qt.LeftDockWidgetArea)
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

        self.updateDepthScanners()

        
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
        depth_img_foreground_raw = self.depthScanners['foreground'].getDepthImageAsNumpyArray()
        depth_img_f = np.array(depth_img_foreground_raw, dtype=np.int32)
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
        returnData['depth_img_foreground_raw'] = depth_img_foreground_raw
        returnData['depth_img_change'] = depth_img_change


        if visualize:
            # make sure to remap the mask to [0,255], otherwise it will be all black
            ChangeDetection.drawNumpyImage('Change Detection Mask', mask*255)

        return returnData

    def computeForegroundMaskUsingCropStrategy(self, visualize=True):
        """
        Computes the foreground mask. The camera location of the foreground view should
        already be set correctly. Steps of the pipeline are

        1. Render depth image from cropped foreground
        2. Everything in the "crop" should be foreground for now
        """

        # assume that foreground view is already in the correct place
        view_f = self.views['foreground']
        view_b = self.views['background']

        view_f.forceRender()
        cameraToWorld = director_utils.getCameraTransform(view_f.camera())
        self.updateDepthScanners()

        # get the depth images
        # make sure to convert them to int32 since we want to avoid wrap-around issues when using
        # uint16 types
        depth_img_foreground_raw = self.depthScanners['foreground'].getDepthImageAsNumpyArray()
        depth_img_f = np.array(depth_img_foreground_raw, dtype=np.int32)


        idx = depth_img_f > 0
        mask = np.zeros(np.shape(depth_img_f))
        mask[idx] = 1

        returnData = dict()
        returnData['idx'] = idx
        returnData['mask'] = mask
        returnData['depth_img_foreground_raw'] = depth_img_foreground_raw
        returnData['depth_img_foreground'] = depth_img_f

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


    def run(self, output_dir=None, rendered_images_dir=None):
        """
        Run the mask generation algorithm
        :return:
        """

        if output_dir is None:
            output_dir = os.path.join(self.foreground_reconstruction.data_dir, 'image_masks')

        if rendered_images_dir is None:
            rendered_images_dir = os.path.join(self.foreground_reconstruction.data_dir, 'rendered_images')

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if not os.path.isdir(rendered_images_dir):
            os.makedirs(rendered_images_dir)

        start_time = time.time()

        # read in each image in the log
        image_dir = self.foreground_reconstruction.image_dir
        camera_pose_data = self.foreground_reconstruction.kinematics_pose_data
        img_file_extension = 'png'

        num_poses = self.foreground_reconstruction.kinematics_pose_data.num_poses()

        logging_rate = 50
        counter = 0

        for idx, value in camera_pose_data.pose_dict.iteritems():
            if (counter % logging_rate) == 0:
                print "Rendering mask for pose %d of %d" %(counter + 1, num_poses)

            mask_image_filename = utils.getPaddedString(idx) + "_mask" + "." + img_file_extension
            mask_image_full_filename = os.path.join(output_dir, mask_image_filename)

            camera_to_world = self.foreground_reconstruction.get_camera_to_world(idx)
            self.setCameraTransform(camera_to_world)
            d = self.computeForegroundMaskUsingCropStrategy(visualize=False)

            mask = d['mask']
            visible_mask = mask*255

            visible_mask_filename = os.path.join(output_dir, utils.getPaddedString(idx) + '_visible_mask'
                                                 + "." + img_file_extension)

            depth_img_filename = os.path.join(rendered_images_dir, utils.getPaddedString(idx) + '_depth_cropped'
                                                 + "." + img_file_extension)

            # save the images
            cv2.imwrite(mask_image_full_filename, mask)
            cv2.imwrite(visible_mask_filename, visible_mask)

            # make sure to save this as uint16
            depth_img = d['depth_img_foreground_raw']
            cv2.imwrite(depth_img_filename, depth_img)

            counter += 1

        end_time = time.time()

        print "rendering masks took %d seconds" %(end_time - start_time)

    def render_depth_images(self, output_dir=None, rendered_images_dir=None):
        """
        Run the mask generation algorithm
        :return:
        """

        if output_dir is None:
            output_dir = os.path.join(self.foreground_reconstruction.data_dir, 'image_masks')

        if rendered_images_dir is None:
            rendered_images_dir = os.path.join(self.foreground_reconstruction.data_dir, 'rendered_images')

        start_time = time.time()

        # read in each image in the log
        image_dir = self.foreground_reconstruction.image_dir
        camera_pose_data = self.foreground_reconstruction.kinematics_pose_data
        img_file_extension = 'png'

        num_poses = self.foreground_reconstruction.kinematics_pose_data.num_poses()

        logging_rate = 50
        counter = 0

        for idx, value in camera_pose_data.pose_dict.iteritems():
            if (counter % logging_rate) == 0:
                print "Rendering depth image for pose %d of %d" % (counter + 1, num_poses)


            camera_to_world = self.foreground_reconstruction.get_camera_to_world(idx)
            self.setCameraTransform(camera_to_world)
            depth_img = self.depthScanners['foreground'].getDepthImageAsNumpyArray()
            depth_img_filename = os.path.join(rendered_images_dir, utils.getPaddedString(idx) + '_depth'
                                              + "." + img_file_extension)

            cv2.imwrite(depth_img_filename, depth_img)

            counter += 1

        end_time = time.time()

        print "rendering depth images took %d seconds" % (end_time - start_time)


    @staticmethod
    def create_change_detection_app(globalsDict=None):
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
        app.mainWindow.resize(920, 600)
        app.mainWindow.move(0, 0)

        view = app.view
        view.setParent(None)
        mdiArea = QtGui.QMdiArea()
        app.mainWindow.setCentralWidget(mdiArea)
        subWindow = mdiArea.addSubWindow(view)
        subWindow.setMinimumSize(300, 300)
        subWindow.setWindowTitle('Camera image')
        subWindow.resize(640, 480)
        mdiArea.tileSubWindows()

        globalsDict['view'] = view
        globalsDict['app'] = app

    @staticmethod
    def from_data_folder(data_folder, config=None, globalsDict=None, background_data_folder=None):
        """
        Creates a ChangeDetection object from a data_folder which contains the
        3D reconstruction and the image files

        :param data_folder:
        :param config:
        :param globalsDict:
        :return:
        """
        foreground_reconstruction = TSDFReconstruction.from_data_folder(data_folder, config=config)

        if background_data_folder is None:
            background_data_folder = data_folder

        print "background folder: ", background_data_folder

        background_reconstruction_placeholder = TSDFReconstruction.from_data_folder(background_data_folder, config=config)


        if globalsDict is None:
            globalsDict = dict()

        ChangeDetection.create_change_detection_app(globalsDict)
        view = globalsDict['view']
        app = globalsDict['app']


        camera_info_file = os.path.join(data_folder, 'images', 'camera_info.yaml')
        camera_intrinsics = utils.CameraIntrinsics.from_yaml_file(camera_info_file)
        changeDetection = ChangeDetection(app, view, cameraIntrinsics=camera_intrinsics)
        changeDetection.foreground_reconstruction = foreground_reconstruction
        changeDetection.background_reconstruction = background_reconstruction_placeholder

        globalsDict['changeDetection'] = changeDetection
        return changeDetection, globalsDict



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
        self.setCameraTransform(cameraToWorld)
        self.updateDepthScanners()

    def testGetCameraTransforms(self):
        vis.updateFrame(vtk.vtkTransform(), "origin frame", view=self.views['background'])
        for viewType, view in self.views.iteritems():
            print "\n\nviewType: ", viewType
            cameraTransform = director_utils.getCameraTransform(view.camera())
            pos, quat = transformUtils.poseFromTransform(cameraTransform)
            print "pos: ", pos
            print "quat: ", quat


            vis.updateFrame(cameraTransform, viewType + "_frame")


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

    def testRenderMask(self, idx=0, save_mask=True):
        camera_to_world = self.foreground_reconstruction.fusion_pose_data.get_camera_to_world_pose(idx)
        self.setCameraTransform(camera_to_world)
        d = self.computeForegroundMaskUsingCropStrategy(visualize=True)
        mask = d['mask']

        mask_filename = "mask.png"
        cv2.imwrite(mask_filename, mask)

    def testGetDepthImage(self, type='foreground'):
        self.testIdentity()
        self.updateDepthScanners()
        depth_img = self.depthScanners[type].getDepthImageAsNumpyArray()
        depth_img_vis = depth_img*255.0/4000.0


        # make sure to remap the mask to [0,255], otherwise it will be all black
        ChangeDetection.drawNumpyImage('Depth Image ' + type , depth_img_vis)
        return depth_img

    def test(self):
        self.updateDepthScanners()
        self.testGetDepthImage('foreground')
        self.testGetDepthImage('background')




def initDepthScanner(app, view, widgetArea=QtCore.Qt.RightDockWidgetArea):
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

    return utils.CameraIntrinsics(cx, cy, fx, fy, width, height)\

def loadDefaultBackground():
    data_folder = '/home/manuelli/code/data_volume/sandbox/drill_scenes/00_background'
    reconstruction = FusionReconstruction.from_data_folder(data_folder)
    return reconstruction

def loadDefaultForeground():
    data_folder = '/home/manuelli/code/data_volume/sandbox/drill_scenes/01_drill'
    reconstruction = FusionReconstruction.from_data_folder(data_folder)
    return reconstruction


def main(globalsDict, data_folder=None):
    if data_folder is None:
        data_folder = '/home/manuelli/code/data_volume/sandbox/04_drill_long_downsampled'
    changeDetection, globalsDict = ChangeDetection.from_data_folder(data_folder, globalsDict=globalsDict)

    globalsDict['cd'] = changeDetection
    app = globalsDict['app']

    app.app.start(restoreWindow=False)


if __name__ == '__main__':
    main(globals())