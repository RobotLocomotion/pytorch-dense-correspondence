import os
import sys
import math
import vtk
import time
import functools
import traceback
import PythonQt
from PythonQt import QtCore, QtGui
import director.applogic as app
from director import objectmodel as om
from director import transformUtils
from director import visualization as vis
from director.transformUtils import getTransformFromAxes
from director.timercallback import TimerCallback
from director import affordancemanager
from director.affordanceitems import *
from director.visualization import *
from director.filterUtils import *
from director.fieldcontainer import FieldContainer
from director.segmentationroutines import *
from director import cameraview

# from thirdparty import qhull_2d
# from thirdparty import min_bounding_rect

import numpy as np
from director import vtkNumpy
from director.debugVis import DebugData
from director.shallowCopy import shallowCopy
from director import ioUtils
from director.uuidutil import newUUID

DRILL_TRIANGLE_BOTTOM_LEFT = 'bottom left'
DRILL_TRIANGLE_BOTTOM_RIGHT = 'bottom right'
DRILL_TRIANGLE_TOP_LEFT = 'top left'
DRILL_TRIANGLE_TOP_RIGHT = 'top right'

# # prefer drc plane segmentation instead of PCL
# try:
#     planeSegmentationFilter = vtk.vtkPlaneSegmentation
# except AttributeError:
#     planeSegmentationFilter = vtk.vtkPCLSACSegmentationPlane

_defaultSegmentationView = None


def getSegmentationView():
    return _defaultSegmentationView or app.getViewManager().findView('Segmentation View')


def getDRCView():
    return app.getDRCView()


def switchToView(viewName):
    app.getViewManager().switchToView(viewName)


def getCurrentView():
    return app.getCurrentRenderView()


def initAffordanceManager(view):
    '''
    Normally the affordance manager is initialized by the application.
    This function can be called from scripts and tests to initialize the manager.
    '''
    global affordanceManager
    affordanceManager = affordancemanager.AffordanceObjectModelManager(view)


def cropToLineSegment(polyData, point1, point2):
    line = np.array(point2) - np.array(point1)
    length = np.linalg.norm(line)
    axis = line / length

    polyData = labelPointDistanceAlongAxis(polyData, axis, origin=point1, resultArrayName='dist_along_line')
    return thresholdPoints(polyData, 'dist_along_line', [0.0, length])


'''
icp programmable filter

import vtkFiltersGeneralPython as filtersGeneral

points = inputs[0]
block = inputs[1]

print points.GetNumberOfPoints()
print block.GetNumberOfPoints()

if points.GetNumberOfPoints() < block.GetNumberOfPoints():
    block, points = points, block

icp = vtk.vtkIterativeClosestPointTransform()
icp.SetSource(points.VTKObject)
icp.SetTarget(block.VTKObject)
icp.GetLandmarkTransform().SetModeToRigidBody()
icp.Update()

t = filtersGeneral.vtkTransformPolyDataFilter()
t.SetInput(points.VTKObject)
t.SetTransform(icp)
t.Update()

output.ShallowCopy(t.GetOutput())
'''


def computeAToB(a, b):
    t = vtk.vtkTransform()
    t.PostMultiply()
    t.Concatenate(b)
    t.Concatenate(a.GetLinearInverse())
    tt = vtk.vtkTransform()
    tt.SetMatrix(t.GetMatrix())
    return tt


def lockAffordanceToHand(aff, hand='l_hand'):
    linkFrame = getLinkFrame(hand)
    affT = aff.actor.GetUserTransform()

    if not hasattr(aff, 'handToAffT') or not aff.handToAffT:
        aff.handToAffT = computeAToB(linkFrame, affT)

    t = vtk.vtkTransform()
    t.PostMultiply()
    t.Concatenate(aff.handToAffT)
    t.Concatenate(linkFrame)
    aff.actor.GetUserTransform().SetMatrix(t.GetMatrix())


handAffUpdater = None


def lockToHandOn():
    aff = getDefaultAffordanceObject()
    if not aff:
        return

    global handAffUpdater
    if handAffUpdater is None:
        handAffUpdater = TimerCallback()
        handAffUpdater.targetFps = 30

    handAffUpdater.callback = functools.partial(lockAffordanceToHand, aff)
    handAffUpdater.start()


def lockToHandOff():
    aff = getDefaultAffordanceObject()
    if not aff:
        return

    handAffUpdater.stop()
    aff.handToAffT = None


class DisparityPointCloudItem(vis.PolyDataItem):
    def __init__(self, name, imagesChannel, cameraName, imageManager):
        vis.PolyDataItem.__init__(self, name, vtk.vtkPolyData(), view=None)

        self.addProperty('Channel', imagesChannel)
        self.addProperty('Camera name', cameraName)

        self.addProperty('Decimation', 0, attributes=om.PropertyAttributes(enumNames=['1', '2', '4', '8', '16']))
        self.addProperty('Remove Size', 1000,
                         attributes=om.PropertyAttributes(decimals=0, minimum=0, maximum=100000.0, singleStep=1000))
        self.addProperty('Target FPS', 1.0,
                         attributes=om.PropertyAttributes(decimals=1, minimum=0.1, maximum=30.0, singleStep=0.1))
        self.addProperty('Max Range', 2.0,
                         attributes=om.PropertyAttributes(decimals=2, minimum=0., maximum=30.0, singleStep=0.25))

        self.timer = TimerCallback()
        self.timer.callback = self.update
        self.lastUtime = 0
        self.imageManager = imageManager
        self.cameraName = cameraName
        self.setProperty('Visible', False)
        self.addProperty('Remove Stale Data', False)
        self.addProperty('Stale Data Timeout', 5.0,
                         attributes=om.PropertyAttributes(decimals=1, minimum=0.1, maximum=30.0, singleStep=0.1))
        self.lastDataReceivedTime = time.time()

    def _onPropertyChanged(self, propertySet, propertyName):
        vis.PolyDataItem._onPropertyChanged(self, propertySet, propertyName)

        if propertyName == 'Visible':
            if self.getProperty(propertyName):
                self.timer.start()
            else:
                self.timer.stop()

        elif propertyName in ('Decimation', 'Remove outliers', 'Max Range'):
            self.lastUtime = 0

    def onRemoveFromObjectModel(self):
        vis.PolyDataItem.onRemoveFromObjectModel(self)
        self.timer.stop()

    def update(self):
        utime = self.imageManager.queue.getCurrentImageTime(self.cameraName)
        if utime == self.lastUtime:
            if self.getProperty('Remove Stale Data') and (
                (time.time() - self.lastDataReceivedTime) > self.getProperty('Stale Data Timeout')):
                if self.polyData.GetNumberOfPoints() > 0:
                    self.setPolyData(vtk.vtkPolyData())
            return

        if (utime < self.lastUtime):
            temp = 0  # dummy
        elif (utime - self.lastUtime < 1E6 / self.getProperty('Target FPS')):
            return

        decimation = int(self.properties.getPropertyEnumValue('Decimation'))
        removeSize = int(self.properties.getProperty('Remove Size'))
        rangeThreshold = float(self.properties.getProperty('Max Range'))
        polyData = getDisparityPointCloud(decimation, imagesChannel=self.getProperty('Channel'),
                                          cameraName=self.getProperty('Camera name'),
                                          removeOutliers=False, removeSize=removeSize, rangeThreshold=rangeThreshold)

        self.setPolyData(polyData)
        self.lastDataReceivedTime = time.time()

        if polyData.GetNumberOfPoints() > 0 and not self.lastUtime:
            self.setProperty('Color By', 'rgb_colors')

        self.lastUtime = utime


def extractLargestCluster(polyData, **kwargs):
    '''
    Calls applyEuclideanClustering and then extracts the first (largest) cluster.
    The given keyword arguments are passed into the applyEuclideanClustering function.
    '''
    polyData = applyEuclideanClustering(polyData, **kwargs)
    return thresholdPoints(polyData, 'cluster_labels', [1, 1])


def segmentGround(polyData, groundThickness=0.02, sceneHeightFromGround=0.05):
    ''' A More complex ground removal algorithm. Works when plane isn't
    preceisely flat. First clusters on z to find approx ground height, then fits a plane there
    '''

    searchRegionThickness = 0.5

    zvalues = vtkNumpy.getNumpyFromVtk(polyData, 'Points')[:, 2]
    groundHeight = np.percentile(zvalues, 5)

    vtkNumpy.addNumpyToVtk(polyData, zvalues.copy(), 'z')
    searchRegion = thresholdPoints(polyData, 'z', [groundHeight - searchRegionThickness / 2.0,
                                                   groundHeight + searchRegionThickness / 2.0])

    updatePolyData(searchRegion, 'ground search region', parent=getDebugFolder(), colorByName='z', visible=False)

    _, origin, normal = applyPlaneFit(searchRegion, distanceThreshold=0.02, expectedNormal=[0, 0, 1],
                                      perpendicularAxis=[0, 0, 1], returnOrigin=True)

    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    dist = np.dot(points - origin, normal)
    vtkNumpy.addNumpyToVtk(polyData, dist, 'dist_to_plane')

    groundPoints = thresholdPoints(polyData, 'dist_to_plane', [-groundThickness / 2.0, groundThickness / 2.0])
    scenePoints = thresholdPoints(polyData, 'dist_to_plane', [sceneHeightFromGround, 100])

    return origin, normal, groundPoints, scenePoints


def segmentGroundPlane():
    inputObj = om.findObjectByName('pointcloud snapshot')
    inputObj.setProperty('Visible', False)
    polyData = shallowCopy(inputObj.polyData)

    zvalues = vtkNumpy.getNumpyFromVtk(polyData, 'Points')[:, 2]
    groundHeight = np.percentile(zvalues, 5)
    searchRegion = thresholdPoints(polyData, 'z', [groundHeight - 0.3, groundHeight + 0.3])

    updatePolyData(searchRegion, 'ground search region', parent=getDebugFolder(), colorByName='z', visible=False)

    _, origin, normal = applyPlaneFit(searchRegion, distanceThreshold=0.02, expectedNormal=[0, 0, 1],
                                      perpendicularAxis=[0, 0, 1], returnOrigin=True)

    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    dist = np.dot(points - origin, normal)
    vtkNumpy.addNumpyToVtk(polyData, dist, 'dist_to_plane')

    groundPoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    scenePoints = thresholdPoints(polyData, 'dist_to_plane', [0.05, 10])

    updatePolyData(groundPoints, 'ground points', alpha=0.3)
    updatePolyData(scenePoints, 'scene points', alpha=0.3)

    # scenePoints = applyEuclideanClustering(scenePoints, clusterTolerance=0.10, minClusterSize=100, maxClusterSize=1e6)
    # updatePolyData(scenePoints, 'scene points', colorByName='cluster_labels')


def applyLocalPlaneFit(polyData, searchPoint, searchRadius, searchRadiusEnd=None, removeGroundFirst=True):
    useVoxelGrid = True
    voxelGridSize = 0.03
    distanceToPlaneThreshold = 0.02

    if useVoxelGrid:
        polyData = applyVoxelGrid(polyData, leafSize=voxelGridSize)

    if removeGroundFirst:
        _, polyData = removeGround(polyData, groundThickness=0.02, sceneHeightFromGround=0.04)

    cropped = cropToSphere(polyData, searchPoint, searchRadius)
    updatePolyData(cropped, 'crop to sphere', visible=False, colorByName='distance_to_point')

    polyData, normal = applyPlaneFit(polyData, distanceToPlaneThreshold, searchOrigin=searchPoint,
                                     searchRadius=searchRadius)

    if searchRadiusEnd is not None:
        polyData, normal = applyPlaneFit(polyData, distanceToPlaneThreshold, perpendicularAxis=normal,
                                         angleEpsilon=math.radians(30), searchOrigin=searchPoint,
                                         searchRadius=searchRadiusEnd)

    fitPoints = thresholdPoints(polyData, 'dist_to_plane', [-distanceToPlaneThreshold, distanceToPlaneThreshold])

    updatePolyData(fitPoints, 'fitPoints', visible=False)

    fitPoints = labelDistanceToPoint(fitPoints, searchPoint)
    clusters = extractClusters(fitPoints, clusterTolerance=0.05, minClusterSize=3)
    clusters.sort(key=lambda x: vtkNumpy.getNumpyFromVtk(x, 'distance_to_point').min())
    fitPoints = clusters[0]

    return fitPoints, normal

    normalEstimationSearchRadius = 0.065

    f = vtk.vtkPCLNormalEstimation()
    f.SetSearchRadius(normalEstimationSearchRadius)
    f.SetInput(polyData)
    f.Update()
    scenePoints = shallowCopy(f.GetOutput())

    normals = vtkNumpy.getNumpyFromVtk(scenePoints, 'normals')
    normalsDotPlaneNormal = np.abs(np.dot(normals, normal))
    vtkNumpy.addNumpyToVtk(scenePoints, normalsDotPlaneNormal, 'normals_dot_plane_normal')

    showPolyData(scenePoints, 'scene_with_normals', parent=getDebugFolder(), colorByName='normals_dot_plane_normal')

    surfaces = thresholdPoints(scenePoints, 'normals_dot_plane_normal', [0.95, 1.0])

    clusters = extractClusters(surfaces, clusterTolerance=0.1, minClusterSize=5)
    clusters = clusters[:10]

    for i, cluster in enumerate(clusters):
        showPolyData(cluster, 'plane cluster %i' % i, parent=getDebugFolder(), visible=False)

    return fitPoints


def orientToMajorPlane(polyData, pickedPoint):
    '''
    Find the largest plane and transform the cloud to align that plane
    Use the given point as the origin
    '''
    distanceToPlaneThreshold = 0.02
    searchRadius = 0.5

    planePoints, origin, normal = applyPlaneFit(polyData, distanceToPlaneThreshold, searchOrigin=pickedPoint,
                                                searchRadius=searchRadius, returnOrigin=True)
    vis.updatePolyData(planePoints, 'local plane fit', color=[0, 1, 0], parent=getDebugFolder(), visible=False)

    planeFrame = transformUtils.getTransformFromOriginAndNormal(pickedPoint, normal)
    vis.updateFrame(planeFrame, 'plane frame', scale=0.15, parent=getDebugFolder(), visible=False)

    polyData = transformPolyData(polyData, planeFrame.GetLinearInverse())

    # if the mean point is below the horizontal plane, flip the cloud
    zvalues = vtkNumpy.getNumpyFromVtk(polyData, 'Points')[:, 2]
    midCloudHeight = np.mean(zvalues)
    if (midCloudHeight < 0):
        flipTransform = transformUtils.frameFromPositionAndRPY([0, 0, 0], [0, 180, 0])
        polyData = transformPolyData(polyData, flipTransform)

    return polyData, planeFrame


def getMajorPlanes(polyData, useVoxelGrid=True):
    voxelGridSize = 0.01
    distanceToPlaneThreshold = 0.02

    if useVoxelGrid:
        polyData = applyVoxelGrid(polyData, leafSize=voxelGridSize)

    polyDataList = []

    minClusterSize = 100

    while len(polyDataList) < 25:

        f = planeSegmentationFilter()
        f.SetInput(polyData)
        f.SetDistanceThreshold(distanceToPlaneThreshold)
        f.Update()
        polyData = shallowCopy(f.GetOutput())

        outliers = thresholdPoints(polyData, 'ransac_labels', [0, 0])
        inliers = thresholdPoints(polyData, 'ransac_labels', [1, 1])
        largestCluster = extractLargestCluster(inliers)

        # i = len(polyDataList)
        # showPolyData(inliers, 'inliers %d' % i, color=getRandomColor(), parent='major planes')
        # showPolyData(outliers, 'outliers %d' % i, color=getRandomColor(), parent='major planes')
        # showPolyData(largestCluster, 'cluster %d' % i, color=getRandomColor(), parent='major planes')

        if largestCluster.GetNumberOfPoints() > minClusterSize:
            polyDataList.append(largestCluster)
            polyData = outliers
        else:
            break

    return polyDataList


def showMajorPlanes(polyData=None):
    if not polyData:
        inputObj = om.findObjectByName('pointcloud snapshot')
        inputObj.setProperty('Visible', False)
        polyData = inputObj.polyData

    om.removeFromObjectModel(om.findObjectByName('major planes'))
    folderObj = om.findObjectByName('segmentation')
    folderObj = om.getOrCreateContainer('major planes', folderObj)

    origin = SegmentationContext.getGlobalInstance().getViewFrame().GetPosition()
    polyData = labelDistanceToPoint(polyData, origin)
    polyData = thresholdPoints(polyData, 'distance_to_point', [1, 4])

    polyDataList = getMajorPlanes(polyData)

    for i, polyData in enumerate(polyDataList):
        obj = showPolyData(polyData, 'plane %d' % i, color=getRandomColor(), visible=True, parent='major planes')
        obj.setProperty('Point Size', 3)


def cropToBox(polyData, transform, dimensions):
    '''
    dimensions is length 3 describing box dimensions
    '''
    origin = np.array(transform.GetPosition())
    axes = transformUtils.getAxesFromTransform(transform)

    for axis, length in zip(axes, dimensions):
        cropAxis = np.array(axis) * (length / 2.0)
        polyData = cropToLineSegment(polyData, origin - cropAxis, origin + cropAxis)

    return polyData


def cropToBounds(polyData, transform, bounds):
    '''
    bounds is a 2x3 containing the min/max values along the transform axes to use for cropping
    '''
    origin = np.array(transform.GetPosition())
    axes = transformUtils.getAxesFromTransform(transform)

    for axis, bound in zip(axes, bounds):
        axis = np.array(axis) / np.linalg.norm(axis)
        polyData = cropToLineSegment(polyData, origin + axis * bound[0], origin + axis * bound[1])

    return polyData


def cropToSphere(polyData, origin, radius):
    polyData = labelDistanceToPoint(polyData, origin)
    return thresholdPoints(polyData, 'distance_to_point', [0, radius])


def applyPlaneFit(polyData, distanceThreshold=0.02, expectedNormal=None, perpendicularAxis=None, angleEpsilon=0.2,
                  returnOrigin=False, searchOrigin=None, searchRadius=None):
    expectedNormal = expectedNormal if expectedNormal is not None else [-1, 0, 0]

    fitInput = polyData
    if searchOrigin is not None:
        assert searchRadius
        fitInput = cropToSphere(fitInput, searchOrigin, searchRadius)

    # perform plane segmentation
    f = planeSegmentationFilter()
    f.SetInput(fitInput)
    f.SetDistanceThreshold(distanceThreshold)
    if perpendicularAxis is not None:
        f.SetPerpendicularConstraintEnabled(True)
        f.SetPerpendicularAxis(perpendicularAxis)
        f.SetAngleEpsilon(angleEpsilon)
    f.Update()
    origin = f.GetPlaneOrigin()
    normal = np.array(f.GetPlaneNormal())

    # flip the normal if needed
    if np.dot(normal, expectedNormal) < 0:
        normal = -normal

    # for each point, compute signed distance to plane

    polyData = shallowCopy(polyData)
    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    dist = np.dot(points - origin, normal)
    vtkNumpy.addNumpyToVtk(polyData, dist, 'dist_to_plane')

    if returnOrigin:
        return polyData, origin, normal
    else:
        return polyData, normal


def flipNormalsWithViewDirection(polyData, viewDirection):
    normals = vnp.getNumpyFromVtk(polyData, 'normals')
    normals[np.dot(normals, viewDirection) > 0] *= -1


def normalEstimation(dataObj, searchCloud=None, searchRadius=0.05, useVoxelGrid=False, voxelGridLeafSize=0.05):
    f = vtk.vtkPCLNormalEstimation()
    f.SetSearchRadius(searchRadius)
    f.SetInput(dataObj)
    if searchCloud:
        f.SetInput(1, searchCloud)
    elif useVoxelGrid:
        f.SetInput(1, applyVoxelGrid(dataObj, voxelGridLeafSize))
    f.Update()
    dataObj = shallowCopy(f.GetOutput())
    dataObj.GetPointData().SetNormals(dataObj.GetPointData().GetArray('normals'))

    return dataObj


def addCoordArraysToPolyData(polyData):
    polyData = shallowCopy(polyData)
    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    vtkNumpy.addNumpyToVtk(polyData, points[:, 0].copy(), 'x')
    vtkNumpy.addNumpyToVtk(polyData, points[:, 1].copy(), 'y')
    vtkNumpy.addNumpyToVtk(polyData, points[:, 2].copy(), 'z')

    viewFrame = SegmentationContext.getGlobalInstance().getViewFrame()
    viewOrigin = viewFrame.TransformPoint([0.0, 0.0, 0.0])
    viewX = viewFrame.TransformVector([1.0, 0.0, 0.0])
    viewY = viewFrame.TransformVector([0.0, 1.0, 0.0])
    viewZ = viewFrame.TransformVector([0.0, 0.0, 1.0])
    polyData = labelPointDistanceAlongAxis(polyData, viewX, origin=viewOrigin, resultArrayName='distance_along_view_x')
    polyData = labelPointDistanceAlongAxis(polyData, viewY, origin=viewOrigin, resultArrayName='distance_along_view_y')
    polyData = labelPointDistanceAlongAxis(polyData, viewZ, origin=viewOrigin, resultArrayName='distance_along_view_z')

    return polyData


def getDebugRevolutionData():
    # dataDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../drc-data'))
    # filename = os.path.join(dataDir, 'valve_wall.vtp')
    # filename = os.path.join(dataDir, 'bungie_valve.vtp')
    # filename = os.path.join(dataDir, 'cinder-blocks.vtp')
    # filename = os.path.join(dataDir, 'cylinder_table.vtp')
    # filename = os.path.join(dataDir, 'firehose.vtp')
    # filename = os.path.join(dataDir, 'debris.vtp')
    # filename = os.path.join(dataDir, 'rev1.vtp')
    # filename = os.path.join(dataDir, 'drill-in-hand.vtp')

    filename = os.path.expanduser('~/Desktop/scans/debris-scan.vtp')

    return addCoordArraysToPolyData(ioUtils.readPolyData(filename))


def getCurrentScanBundle(useVoxelGrid=False):
    obj = om.findObjectByName('SCANS_HALF_SWEEP')
    if not obj:
        return None

    revPolyData = obj.polyData
    if not revPolyData or not revPolyData.GetNumberOfPoints():
        return None

    if useVoxelGrid:
        revPolyData = applyVoxelGrid(revPolyData, leafSize=0.015)

    return addCoordArraysToPolyData(revPolyData)


def getCurrentRevolutionData(useVoxelGrid=False):
    from director import perception
    revPolyData = perception._multisenseItem.model.revPolyData
    if not revPolyData or not revPolyData.GetNumberOfPoints():
        return getCurrentScanBundle()

    if useVoxelGrid:
        revPolyData = applyVoxelGrid(revPolyData, leafSize=0.015)

    return addCoordArraysToPolyData(revPolyData)


def getDisparityPointCloud(decimation=4, removeOutliers=True, removeSize=0, rangeThreshold=-1,
                           imagesChannel='MULTISENSE_CAMERA', cameraName='CAMERA_LEFT'):
    p = cameraview.getStereoPointCloud(decimation, imagesChannel=imagesChannel, cameraName=cameraName,
                                       removeSize=removeSize, rangeThreshold=rangeThreshold)
    if not p:
        return None

    if removeOutliers:
        # attempt to scale outlier filtering, best tuned for decimation of 2 or 4
        scaling = (10 * 16) / (decimation * decimation)
        p = labelOutliers(p, searchRadius=0.06, neighborsInSearchRadius=scaling)
        p = thresholdPoints(p, 'is_outlier', [0.0, 0.0])

    return p


def getCurrentMapServerData():
    mapServer = om.findObjectByName('Map Server')
    polyData = None
    if mapServer and mapServer.getProperty('Visible'):
        polyData = mapServer.source.polyData

    if not polyData or not polyData.GetNumberOfPoints():
        return None

    return addCoordArraysToPolyData(polyData)


def segmentGroundPlanes():
    objs = []
    for obj in om.getObjects():
        name = obj.getProperty('Name')
        if name.startswith('pointcloud snapshot'):
            objs.append(obj)

    objs = sorted(objs, key=lambda x: x.getProperty('Name'))

    d = DebugData()

    prevHeadAxis = None
    for obj in objs:
        name = obj.getProperty('Name')
        print '----- %s---------' % name
        print  'head axis:', obj.headAxis
        origin, normal, groundPoints, _ = segmentGround(obj.polyData)
        print 'ground normal:', normal
        showPolyData(groundPoints, name + ' ground points', visible=False)
        a = np.array([0, 0, 1])
        b = np.array(normal)
        diff = math.degrees(math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        if diff > 90:
            print 180 - diff
        else:
            print diff

        if prevHeadAxis is not None:
            a = prevHeadAxis
            b = np.array(obj.headAxis)
            diff = math.degrees(math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
            if diff > 90:
                print 180 - diff
            else:
                print diff
        prevHeadAxis = np.array(obj.headAxis)

        d.addLine([0, 0, 0], normal)

    updatePolyData(d.getPolyData(), 'normals')


def extractCircle(polyData, distanceThreshold=0.04, radiusLimit=None):
    circleFit = vtk.vtkPCLSACSegmentationCircle()
    circleFit.SetDistanceThreshold(distanceThreshold)
    circleFit.SetInput(polyData)
    if radiusLimit is not None:
        circleFit.SetRadiusLimit(radiusLimit)
        circleFit.SetRadiusConstraintEnabled(True)
    circleFit.Update()

    polyData = thresholdPoints(circleFit.GetOutput(), 'ransac_labels', [1.0, 1.0])
    return polyData, circleFit


def removeMajorPlane(polyData, distanceThreshold=0.02):
    # perform plane segmentation
    f = planeSegmentationFilter()
    f.SetInput(polyData)
    f.SetDistanceThreshold(distanceThreshold)
    f.Update()

    polyData = thresholdPoints(f.GetOutput(), 'ransac_labels', [0.0, 0.0])
    return polyData, f


def removeGroundSimple(polyData, groundThickness=0.02, sceneHeightFromGround=0.05):
    ''' Simple ground plane removal algorithm. Uses ground height
        and does simple z distance filtering.
        Suitable for noisy data e.g. kinect/stereo camera
        (Default args should be relaxed, filtering simplfied)
    '''
    groundHeight = SegmentationContext.getGlobalInstance().getGroundHeight()
    origin = [0, 0, groundHeight]
    normal = [0, 0, 1]

    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    dist = np.dot(points - origin, normal)
    vtkNumpy.addNumpyToVtk(polyData, dist, 'dist_to_plane')

    groundPoints = thresholdPoints(polyData, 'dist_to_plane', [-groundThickness / 2.0, groundThickness / 2.0])
    scenePoints = thresholdPoints(polyData, 'dist_to_plane', [sceneHeightFromGround, 100])

    return groundPoints, scenePoints


def removeGround(polyData, groundThickness=0.02, sceneHeightFromGround=0.05):
    origin, normal, groundPoints, scenePoints = segmentGround(polyData, groundThickness, sceneHeightFromGround)
    return groundPoints, scenePoints


def generateFeetForValve():
    aff = om.findObjectByName('valve affordance')
    assert aff

    params = aff.params

    origin = np.array(params['origin'])
    origin[2] = 0.0

    xaxis = -params['axis']
    zaxis = np.array([0, 0, 1])
    yaxis = np.cross(zaxis, xaxis)
    xaxis = np.cross(yaxis, zaxis)

    stanceWidth = 0.2
    stanceRotation = 25.0
    stanceOffset = [-1.0, -0.5, 0.0]

    valveFrame = getTransformFromAxes(xaxis, yaxis, zaxis)
    valveFrame.PostMultiply()
    valveFrame.Translate(origin)

    stanceFrame, lfootFrame, rfootFrame = getFootFramesFromReferenceFrame(valveFrame, stanceWidth, stanceRotation,
                                                                          stanceOffset)

    showFrame(boardFrame, 'board ground frame', parent=aff, scale=0.15, visible=False)
    showFrame(lfootFrame, 'lfoot frame', parent=aff, scale=0.15)
    showFrame(rfootFrame, 'rfoot frame', parent=aff, scale=0.15)

    # d = DebugData()
    # d.addLine(valveFrame.GetPosition(), stanceFrame.GetPosition())
    # updatePolyData(d.getPolyData(), 'stance debug')
    # publishSteppingGoal(lfootFrame, rfootFrame)


def generateFeetForDebris():
    aff = om.findObjectByName('board A')
    if not aff:
        return

    params = aff.params

    origin = np.array(params['origin'])

    origin = origin + params['zaxis'] * params['zwidth'] / 2.0 - params['xaxis'] * params['xwidth'] / 2.0
    origin[2] = 0.0

    yaxis = params['zaxis']
    zaxis = np.array([0, 0, 1])
    xaxis = np.cross(yaxis, zaxis)

    stanceWidth = 0.35
    stanceRotation = 0.0
    stanceOffset = [-0.48, -0.08, 0]

    boardFrame = getTransformFromAxes(xaxis, yaxis, zaxis)
    boardFrame.PostMultiply()
    boardFrame.Translate(origin)

    stanceFrame, lfootFrame, rfootFrame = getFootFramesFromReferenceFrame(boardFrame, stanceWidth, stanceRotation,
                                                                          stanceOffset)

    showFrame(boardFrame, 'board ground frame', parent=aff, scale=0.15, visible=False)
    lfoot = showFrame(lfootFrame, 'lfoot frame', parent=aff, scale=0.15)
    rfoot = showFrame(rfootFrame, 'rfoot frame', parent=aff, scale=0.15)

    for obj in [lfoot, rfoot]:
        obj.addToView(app.getDRCView())

        # d = DebugData()
        # d.addLine(valveFrame.GetPosition(), stanceFrame.GetPosition())
        # updatePolyData(d.getPolyData(), 'stance debug')
        # publishSteppingGoal(lfootFrame, rfootFrame)


def generateFeetForWye():
    aff = om.findObjectByName('wye points')
    if not aff:
        return

    params = aff.params

    origin = np.array(params['origin'])
    origin[2] = 0.0

    yaxis = params['xaxis']
    xaxis = -params['zaxis']
    zaxis = np.cross(xaxis, yaxis)

    stanceWidth = 0.20
    stanceRotation = 0.0
    stanceOffset = [-0.48, -0.08, 0]

    affGroundFrame = getTransformFromAxes(xaxis, yaxis, zaxis)
    affGroundFrame.PostMultiply()
    affGroundFrame.Translate(origin)

    stanceFrame, lfootFrame, rfootFrame = getFootFramesFromReferenceFrame(affGroundFrame, stanceWidth, stanceRotation,
                                                                          stanceOffset)

    showFrame(affGroundFrame, 'affordance ground frame', parent=aff, scale=0.15, visible=False)
    lfoot = showFrame(lfootFrame, 'lfoot frame', parent=aff, scale=0.15)
    rfoot = showFrame(rfootFrame, 'rfoot frame', parent=aff, scale=0.15)

    for obj in [lfoot, rfoot]:
        obj.addToView(app.getDRCView())


def getFootFramesFromReferenceFrame(referenceFrame, stanceWidth, stanceRotation, stanceOffset):
    footHeight = 0.0745342

    ref = vtk.vtkTransform()
    ref.SetMatrix(referenceFrame.GetMatrix())

    stanceFrame = vtk.vtkTransform()
    stanceFrame.PostMultiply()
    stanceFrame.RotateZ(stanceRotation)
    stanceFrame.Translate(stanceOffset)
    stanceFrame.Concatenate(ref)

    lfootFrame = vtk.vtkTransform()
    lfootFrame.PostMultiply()
    lfootFrame.Translate(0, stanceWidth / 2.0, footHeight)
    lfootFrame.Concatenate(stanceFrame)

    rfootFrame = vtk.vtkTransform()
    rfootFrame.PostMultiply()
    rfootFrame.Translate(0, -stanceWidth / 2.0, footHeight)
    rfootFrame.Concatenate(stanceFrame)

    return stanceFrame, lfootFrame, rfootFrame


def poseFromFrame(frame):
    import bot_core as lcmbotcore

    pos, quat = transformUtils.poseFromTransform(frame)
    trans = lcmbotcore.vector_3d_t()
    trans.x, trans.y, trans.z = pos

    quatMsg = lcmbotcore.quaternion_t()
    quatMsg.w, quatMsg.x, quatMsg.y, quatMsg.z = quat

    pose = lcmbotcore.position_3d_t()
    pose.translation = trans
    pose.rotation = quatMsg
    return pose


def cropToPlane(polyData, origin, normal, threshold):
    polyData = shallowCopy(polyData)
    normal = normal / np.linalg.norm(normal)
    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    dist = np.dot(points - origin, normal)
    vtkNumpy.addNumpyToVtk(polyData, dist, 'dist_to_plane')
    cropped = thresholdPoints(polyData, 'dist_to_plane', threshold)
    return cropped, polyData


def createLine(blockDimensions, p1, p2):
    sliceWidth = np.array(blockDimensions).max() / 2.0 + 0.02
    sliceThreshold = [-sliceWidth, sliceWidth]

    # require p1 to be point on left
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    _, worldPt1 = getRayFromDisplayPoint(app.getCurrentRenderView(), p1)
    _, worldPt2 = getRayFromDisplayPoint(app.getCurrentRenderView(), p2)

    cameraPt = np.array(app.getCurrentRenderView().camera().GetPosition())

    leftRay = worldPt1 - cameraPt
    rightRay = worldPt2 - cameraPt
    middleRay = (leftRay + rightRay) / 2.0

    d = DebugData()
    d.addLine(cameraPt, worldPt1)
    d.addLine(cameraPt, worldPt2)
    d.addLine(worldPt1, worldPt2)
    d.addLine(cameraPt, cameraPt + middleRay)
    updatePolyData(d.getPolyData(), 'line annotation', parent=getDebugFolder(), visible=False)

    inputObj = om.findObjectByName('pointcloud snapshot')
    if inputObj:
        polyData = shallowCopy(inputObj.polyData)
    else:
        polyData = getCurrentRevolutionData()

    origin = cameraPt

    normal = np.cross(rightRay, leftRay)
    leftNormal = np.cross(normal, leftRay)
    rightNormal = np.cross(rightRay, normal)

    normal /= np.linalg.norm(normal)
    leftNormal /= np.linalg.norm(leftNormal)
    rightNormal /= np.linalg.norm(rightNormal)
    middleRay /= np.linalg.norm(middleRay)

    cropped, polyData = cropToPlane(polyData, origin, normal, sliceThreshold)

    updatePolyData(polyData, 'slice dist', parent=getDebugFolder(), colorByName='dist_to_plane',
                   colorByRange=[-0.5, 0.5], visible=False)
    updatePolyData(cropped, 'slice', parent=getDebugFolder(), colorByName='dist_to_plane', visible=False)

    cropped, _ = cropToPlane(cropped, origin, leftNormal, [-1e6, 0])
    cropped, _ = cropToPlane(cropped, origin, rightNormal, [-1e6, 0])

    updatePolyData(cropped, 'slice segment', parent=getDebugFolder(), colorByName='dist_to_plane', visible=False)

    planePoints, planeNormal = applyPlaneFit(cropped, distanceThreshold=0.005, perpendicularAxis=middleRay,
                                             angleEpsilon=math.radians(60))
    planePoints = thresholdPoints(planePoints, 'dist_to_plane', [-0.005, 0.005])
    updatePolyData(planePoints, 'board segmentation', parent=getDebugFolder(), color=getRandomColor(), visible=False)

    '''
    names = ['board A', 'board B', 'board C', 'board D', 'board E', 'board F', 'board G', 'board H', 'board I']
    for name in names:
        if not om.findObjectByName(name):
            break
    else:
        name = 'board'
    '''
    name = 'board'

    segmentBlockByTopPlane(planePoints, blockDimensions, expectedNormal=-middleRay, expectedXAxis=middleRay,
                           edgeSign=-1, name=name)


def updateBlockAffordances(polyData=None):
    for obj in om.getObjects():
        if isinstance(obj, BoxAffordanceItem):
            if 'refit' in obj.getProperty('Name'):
                om.removeFromObjectModel(obj)

    for obj in om.getObjects():
        if isinstance(obj, BoxAffordanceItem):
            updateBlockFit(obj, polyData)


def updateBlockFit(affordanceObj, polyData=None):
    affordanceObj.updateParamsFromActorTransform()

    name = affordanceObj.getProperty('Name') + ' refit'
    origin = affordanceObj.params['origin']
    normal = affordanceObj.params['yaxis']
    edgePerpAxis = affordanceObj.params['xaxis']
    blockDimensions = [affordanceObj.params['xwidth'], affordanceObj.params['ywidth']]

    if polyData is None:
        inputObj = om.findObjectByName('pointcloud snapshot')
        polyData = shallowCopy(inputObj.polyData)

    cropThreshold = 0.1
    cropped = polyData
    cropped, _ = cropToPlane(cropped, origin, normal, [-cropThreshold, cropThreshold])
    cropped, _ = cropToPlane(cropped, origin, edgePerpAxis, [-cropThreshold, cropThreshold])

    updatePolyData(cropped, 'refit search region', parent=getDebugFolder(), visible=False)

    cropped = extractLargestCluster(cropped)

    planePoints, planeNormal = applyPlaneFit(cropped, distanceThreshold=0.005, perpendicularAxis=normal,
                                             angleEpsilon=math.radians(10))
    planePoints = thresholdPoints(planePoints, 'dist_to_plane', [-0.005, 0.005])
    updatePolyData(planePoints, 'refit board segmentation', parent=getDebugFolder(), visible=False)

    refitObj = segmentBlockByTopPlane(planePoints, blockDimensions, expectedNormal=normal, expectedXAxis=edgePerpAxis,
                                      edgeSign=-1, name=name)

    refitOrigin = np.array(refitObj.params['origin'])
    refitLength = refitObj.params['zwidth']
    refitZAxis = refitObj.params['zaxis']
    refitEndPoint1 = refitOrigin + refitZAxis * refitLength / 2.0

    originalLength = affordanceObj.params['zwidth']
    correctedOrigin = refitEndPoint1 - refitZAxis * originalLength / 2.0
    originDelta = correctedOrigin - refitOrigin

    refitObj.params['zwidth'] = originalLength
    refitObj.polyData.DeepCopy(affordanceObj.polyData)
    refitObj.actor.GetUserTransform().Translate(originDelta)
    refitObj.updateParamsFromActorTransform()


def startInteractiveLineDraw(blockDimensions):
    picker = LineDraw(app.getCurrentRenderView())
    addViewPicker(picker)
    picker.enabled = True
    picker.start()
    picker.annotationFunc = functools.partial(createLine, blockDimensions)


def startLeverValveSegmentation():
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentLeverValve)


def refitValveAffordance(aff, point1, origin, normal):
    xaxis = aff.params['xaxis']
    yaxis = aff.params['yaxis']
    zaxis = aff.params['zaxis']
    origin = aff.params['origin']

    zaxis = normal
    xaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(origin)

    aff.actor.GetUserTransform().SetMatrix(t.GetMatrix())
    aff.updateParamsFromActorTransform()


def segmentValve(expectedValveRadius, point1, point2):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())

    polyData, _, wallNormal = applyPlaneFit(polyData, expectedNormal=viewPlaneNormal, searchOrigin=point1,
                                            searchRadius=0.2, angleEpsilon=0.7, returnOrigin=True)

    wallPoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(wallPoints, 'wall points', parent=getDebugFolder(), visible=False)

    polyData, _, _ = applyPlaneFit(polyData, expectedNormal=wallNormal, searchOrigin=point2,
                                   searchRadius=expectedValveRadius, angleEpsilon=0.2, returnOrigin=True)
    valveCluster = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    valveCluster = cropToSphere(valveCluster, point2, expectedValveRadius * 2)
    valveCluster = extractLargestCluster(valveCluster, minClusterSize=1)
    updatePolyData(valveCluster, 'valve cluster', parent=getDebugFolder(), visible=False)
    origin = np.average(vtkNumpy.getNumpyFromVtk(valveCluster, 'Points'), axis=0)

    zaxis = wallNormal
    xaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(origin)

    zwidth = 0.03
    radius = expectedValveRadius

    d = DebugData()
    d.addLine(np.array([0, 0, -zwidth / 2.0]), np.array([0, 0, zwidth / 2.0]), radius=radius)

    name = 'valve affordance'
    obj = showPolyData(d.getPolyData(), name, cls=FrameAffordanceItem, parent='affordances', color=[0, 1, 0])
    obj.actor.SetUserTransform(t)
    obj.addToView(app.getDRCView())
    refitWallCallbacks.append(functools.partial(refitValveAffordance, obj))

    params = dict(axis=zaxis, radius=radius, length=zwidth, origin=origin, xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
                  xwidth=radius, ywidth=radius, zwidth=zwidth,
                  otdf_type='steering_cyl', friendly_name='valve')

    obj.setAffordanceParams(params)
    obj.updateParamsFromActorTransform()

    frameObj = showFrame(obj.actor.GetUserTransform(), name + ' frame', parent=obj, scale=radius, visible=False)
    frameObj.addToView(app.getDRCView())


def segmentValveByBoundingBox(polyData, searchPoint):
    viewDirection = SegmentationContext.getGlobalInstance().getViewDirection()

    polyData = cropToSphere(polyData, searchPoint, radius=0.6)
    polyData = applyVoxelGrid(polyData, leafSize=0.015)

    # extract tube search region
    polyData = labelDistanceToLine(polyData, searchPoint, np.array(searchPoint) + np.array([0, 0, 1]))
    searchRegion = thresholdPoints(polyData, 'distance_to_line', [0.0, 0.2])
    updatePolyData(searchRegion, 'valve tube search region', parent=getDebugFolder(), color=[1, 0, 0], visible=False)

    # guess valve plane
    _, origin, normal = applyPlaneFit(searchRegion, distanceThreshold=0.01, perpendicularAxis=viewDirection,
                                      angleEpsilon=math.radians(30), expectedNormal=-viewDirection, returnOrigin=True)

    # extract plane search region
    polyData = labelPointDistanceAlongAxis(polyData, normal, origin)
    searchRegion = thresholdPoints(polyData, 'distance_along_axis', [-0.05, 0.05])
    updatePolyData(searchRegion, 'valve plane search region', parent=getDebugFolder(),
                   colorByName='distance_along_axis', visible=False)

    valvePoints = extractLargestCluster(searchRegion, minClusterSize=1)
    updatePolyData(valvePoints, 'valve cluster', parent=getDebugFolder(), color=[0, 1, 0], visible=False)

    valvePoints, _ = applyPlaneFit(valvePoints, expectedNormal=normal, perpendicularAxis=normal, distanceThreshold=0.01)
    valveFit = thresholdPoints(valvePoints, 'dist_to_plane', [-0.01, 0.01])

    updatePolyData(valveFit, 'valve cluster', parent=getDebugFolder(), color=[0, 1, 0], visible=False)

    points = vtkNumpy.getNumpyFromVtk(valveFit, 'Points')
    zvalues = points[:, 2].copy()
    minZ = np.min(zvalues)
    maxZ = np.max(zvalues)

    tubeRadius = 0.017
    radius = float((maxZ - minZ) / 2.0) - tubeRadius

    fields = makePolyDataFields(valveFit)
    origin = np.array(fields.frame.GetPosition())

    # origin = computeCentroid(valveFit)

    zaxis = [0, 0, 1]
    xaxis = normal
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(origin)

    pose = transformUtils.poseFromTransform(t)
    desc = dict(classname='CapsuleRingAffordanceItem', Name='valve', uuid=newUUID(), pose=pose, Color=[0, 1, 0],
                Radius=radius, Segments=20)
    desc['Tube Radius'] = tubeRadius

    obj = affordanceManager.newAffordanceFromDescription(desc)
    obj.params = dict(radius=radius)

    return obj


def segmentDoorPlane(polyData, doorPoint, stanceFrame):
    doorPoint = np.array(doorPoint)
    doorBand = 1.5

    polyData = cropToLineSegment(polyData, doorPoint + [0.0, 0.0, doorBand / 2], doorPoint - [0.0, 0.0, doorBand / 2])
    fitPoints, normal = applyLocalPlaneFit(polyData, doorPoint, searchRadius=0.2, searchRadiusEnd=1.0,
                                           removeGroundFirst=False)

    updatePolyData(fitPoints, 'door points', visible=False, color=[0, 1, 0])

    viewDirection = SegmentationContext.getGlobalInstance().getViewDirection()
    if np.dot(normal, viewDirection) > 0:
        normal = -normal

    origin = computeCentroid(fitPoints)
    groundHeight = stanceFrame.GetPosition()[2]
    origin = [origin[0], origin[1], groundHeight]

    xaxis = -normal
    zaxis = [0, 0, 1]

    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(origin)

    return t


def segmentValveByRim(polyData, rimPoint1, rimPoint2):
    viewDirection = SegmentationContext.getGlobalInstance().getViewDirection()

    yaxis = np.array(rimPoint2) - np.array(rimPoint1)
    zaxis = [0, 0, 1]
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)

    # flip xaxis to be with view direction
    if np.dot(xaxis, viewDirection) < 0:
        xaxis = -xaxis

    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)

    origin = (np.array(rimPoint2) + np.array(rimPoint1)) / 2.0

    polyData = labelPointDistanceAlongAxis(polyData, xaxis, origin)
    polyData = thresholdPoints(polyData, 'distance_along_axis', [-0.05, 0.05])
    updatePolyData(polyData, 'valve plane region', parent=getDebugFolder(), colorByName='distance_along_axis',
                   visible=False)

    polyData = cropToSphere(polyData, origin, radius=0.4)
    polyData = applyVoxelGrid(polyData, leafSize=0.015)

    updatePolyData(polyData, 'valve search region', parent=getDebugFolder(), color=[1, 0, 0], visible=False)

    valveFit = extractLargestCluster(polyData, minClusterSize=1)
    updatePolyData(valveFit, 'valve cluster', parent=getDebugFolder(), color=[0, 1, 0], visible=False)

    points = vtkNumpy.getNumpyFromVtk(valveFit, 'Points')
    zvalues = points[:, 2].copy()
    minZ = np.min(zvalues)
    maxZ = np.max(zvalues)

    tubeRadius = 0.017
    radius = float((maxZ - minZ) / 2.0) - tubeRadius

    fields = makePolyDataFields(valveFit)
    origin = np.array(fields.frame.GetPosition())
    vis.updatePolyData(transformPolyData(fields.box, fields.frame), 'valve cluster bounding box', visible=False)

    # origin = computeCentroid(valveFit)

    '''
    zaxis = [0,0,1]
    xaxis = normal
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    '''

    radius = np.max(fields.dims) / 2.0 - tubeRadius

    proj = [np.abs(np.dot(xaxis, axis)) for axis in fields.axes]
    xaxisNew = fields.axes[np.argmax(proj)]
    if np.dot(xaxisNew, xaxis) < 0:
        xaxisNew = -xaxisNew

    xaxis = xaxisNew

    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(origin)

    pose = transformUtils.poseFromTransform(t)
    desc = dict(classname='CapsuleRingAffordanceItem', Name='valve', uuid=newUUID(), pose=pose, Color=[0, 1, 0],
                Radius=float(radius), Segments=20)
    desc['Tube Radius'] = tubeRadius

    obj = affordanceManager.newAffordanceFromDescription(desc)
    obj.params = dict(radius=radius)

    return obj


def segmentValveByWallPlane(expectedValveRadius, point1, point2):
    centerPoint = (point1 + point2) / 2.0

    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    _, polyData = removeGround(polyData)

    viewDirection = SegmentationContext.getGlobalInstance().getViewDirection()
    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=-viewDirection, returnOrigin=True)

    perpLine = np.cross(point2 - point1, normal)
    # perpLine /= np.linalg.norm(perpLine)
    # perpLine * np.linalg.norm(point2 - point1)/2.0
    point3, point4 = centerPoint + perpLine / 2.0, centerPoint - perpLine / 2.0

    d = DebugData()
    d.addLine(point1, point2)
    d.addLine(point3, point4)
    updatePolyData(d.getPolyData(), 'crop lines', parent=getDebugFolder(), visible=False)

    wallPoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(wallPoints, 'valve wall', parent=getDebugFolder(), visible=False)

    searchRegion = thresholdPoints(polyData, 'dist_to_plane', [0.05, 0.5])
    searchRegion = cropToLineSegment(searchRegion, point1, point2)
    searchRegion = cropToLineSegment(searchRegion, point3, point4)

    updatePolyData(searchRegion, 'valve search region', parent=getDebugFolder(), color=[1, 0, 0], visible=False)

    searchRegionSpokes = shallowCopy(searchRegion)

    searchRegion, origin, _ = applyPlaneFit(searchRegion, expectedNormal=normal, perpendicularAxis=normal,
                                            returnOrigin=True)
    searchRegion = thresholdPoints(searchRegion, 'dist_to_plane', [-0.015, 0.015])

    updatePolyData(searchRegion, 'valve search region 2', parent=getDebugFolder(), color=[0, 1, 0], visible=False)

    largestCluster = extractLargestCluster(searchRegion, minClusterSize=1)

    updatePolyData(largestCluster, 'valve cluster', parent=getDebugFolder(), color=[0, 1, 0], visible=False)

    radiusLimit = [expectedValveRadius - 0.01, expectedValveRadius + 0.01] if expectedValveRadius else None
    # radiusLimit = None

    polyData, circleFit = extractCircle(largestCluster, distanceThreshold=0.01, radiusLimit=radiusLimit)
    updatePolyData(polyData, 'circle fit', parent=getDebugFolder(), visible=False)

    # polyData, circleFit = extractCircle(polyData, distanceThreshold=0.01)
    # showPolyData(polyData, 'circle fit', colorByName='z')


    radius = circleFit.GetCircleRadius()
    origin = np.array(circleFit.GetCircleOrigin())
    circleNormal = np.array(circleFit.GetCircleNormal())
    circleNormal = circleNormal / np.linalg.norm(circleNormal)

    if np.dot(circleNormal, normal) < 0:
        circleNormal *= -1

    # force use of the plane normal
    circleNormal = normal
    radius = expectedValveRadius

    d = DebugData()
    d.addLine(origin - normal * radius, origin + normal * radius)
    d.addCircle(origin, circleNormal, radius)
    updatePolyData(d.getPolyData(), 'valve axes', parent=getDebugFolder(), visible=False)

    zaxis = -circleNormal
    xaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    # t = getTransformFromAxes(xaxis, yaxis, zaxis) # this was added to be consistent with segmentValveByRim
    t = getTransformFromAxes(zaxis, -yaxis, xaxis)  # this was added to be consistent with segmentValveByRim
    t.PostMultiply()
    t.Translate(origin)

    # Spoke angle fitting:
    if (1 == 0):  # disabled jan 2015
        # extract the relative positon of the points to the valve axis:
        searchRegionSpokes = labelDistanceToLine(searchRegionSpokes, origin, [origin + circleNormal])
        searchRegionSpokes = thresholdPoints(searchRegionSpokes, 'distance_to_line', [0.05, radius - 0.04])
        updatePolyData(searchRegionSpokes, 'valve spoke search', parent=getDebugFolder(), visible=False)
        searchRegionSpokesLocal = transformPolyData(searchRegionSpokes, t.GetLinearInverse())
        points = vtkNumpy.getNumpyFromVtk(searchRegionSpokesLocal, 'Points')

        spoke_angle = findValveSpokeAngle(points)
    else:
        spoke_angle = 0

    spokeAngleTransform = transformUtils.frameFromPositionAndRPY([0, 0, 0], [0, 0, spoke_angle])
    spokeTransform = transformUtils.copyFrame(t)
    spokeAngleTransform.Concatenate(spokeTransform)
    spokeObj = showFrame(spokeAngleTransform, 'spoke frame', parent=getDebugFolder(), visible=False, scale=radius)
    spokeObj.addToView(app.getDRCView())
    t = spokeAngleTransform

    tubeRadius = 0.017

    pose = transformUtils.poseFromTransform(t)
    desc = dict(classname='CapsuleRingAffordanceItem', Name='valve', uuid=newUUID(), pose=pose, Color=[0, 1, 0],
                Radius=float(radius), Segments=20)
    desc['Tube Radius'] = tubeRadius

    obj = affordanceManager.newAffordanceFromDescription(desc)
    obj.params = dict(radius=radius)


def showHistogram(polyData, arrayName, numberOfBins=100):
    import matplotlib.pyplot as plt

    x = vnp.getNumpyFromVtk(polyData, arrayName)
    hist, bins = np.histogram(x, bins=numberOfBins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

    return bins[np.argmax(hist)] + (bins[1] - bins[0]) / 2.0


def showTable(table, parent):
    '''
    explictly draw a table and its frames
    '''
    pose = transformUtils.poseFromTransform(table.frame)
    desc = dict(classname='MeshAffordanceItem', Name='table', Color=[0, 1, 0], pose=pose)
    aff = affordanceManager.newAffordanceFromDescription(desc)
    aff.setPolyData(table.mesh)

    tableBox = vis.showPolyData(table.box, 'table box', parent=aff, color=[0, 1, 0], visible=False)
    tableBox.actor.SetUserTransform(table.frame)


def applyKmeansLabel(polyData, arrayName, numberOfClusters, whiten=False):
    import scipy.cluster
    ar = vnp.getNumpyFromVtk(polyData, arrayName).copy()

    if whiten:
        scipy.cluster.vq.whiten(ar)

    codes, disturbances = scipy.cluster.vq.kmeans(ar, numberOfClusters)

    if arrayName == 'normals' and numberOfClusters == 2:
        v1 = codes[0]
        v2 = codes[1]
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        angle = np.arccos(np.dot(v1, v2))
        print 'angle between normals:', np.degrees(angle)

    code, distance = scipy.cluster.vq.vq(ar, codes)

    polyData = shallowCopy(polyData)
    vnp.addNumpyToVtk(polyData, code, '%s_kmeans_label' % arrayName)
    return polyData


def findValveSpokeAngle(points):
    '''
    Determine the location of the valve spoke angle
    By binning the spoke returns. returns angle in degrees
    '''

    # np.savetxt("/home/mfallon/Desktop/spoke_points.csv", points, delimiter=",")


    # convert all points to degrees in range [0,120]
    angle = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    qq = np.where(angle < 0)[0]
    angle[qq] += 360
    angle = np.mod(angle, 120)

    # find the spoke as the max of a histogram:
    bins = range(0, 130, 10)  # 0,10,...130
    freq, bins = np.histogram(angle, bins)
    amax = np.argmax(freq)
    spoke_angle = bins[amax] + 5  # correct for 5deg offset

    return spoke_angle


def findWallCenter(polyData, removeGroundMethod=removeGround):
    '''
    Find a frame at the center of the valve wall
    X&Y: average of points on the wall plane
    Z: 4 feet off the ground (determined using robot's feet
    Orientation: z-normal into plane, y-axis horizontal
    '''

    _, polyData = removeGroundMethod(polyData)

    viewDirection = SegmentationContext.getGlobalInstance().getViewDirection()
    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=-viewDirection, returnOrigin=True)

    wallPoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    wallPoints = applyVoxelGrid(wallPoints, leafSize=0.03)
    wallPoints = extractLargestCluster(wallPoints, minClusterSize=100)

    updatePolyData(wallPoints, 'auto valve wall', parent=getDebugFolder(), visible=False)

    xvalues = vtkNumpy.getNumpyFromVtk(wallPoints, 'Points')[:, 0]
    yvalues = vtkNumpy.getNumpyFromVtk(wallPoints, 'Points')[:, 1]

    # median or mid of max or min?
    # xcenter = np.median(xvalues)
    # ycenter = np.median(yvalues)
    xcenter = (np.max(xvalues) + np.min(xvalues)) / 2
    ycenter = (np.max(yvalues) + np.min(yvalues)) / 2

    # not used, not very reliable
    # zvalues = vtkNumpy.getNumpyFromVtk(wallPoints, 'Points')[:,2]
    # zcenter = np.median(zvalues)
    zcenter = SegmentationContext.getGlobalInstance().getGroundHeight() + 1.2192  # valves are 4ft from ground
    point1 = np.array([xcenter, ycenter, zcenter])  # center of the valve wall

    zaxis = -normal
    xaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(point1)

    normalObj = showFrame(t, 'valve wall frame', parent=getDebugFolder(), visible=False)  # z direction out of wall
    normalObj.addToView(app.getDRCView())

    return t


def segmentValveWallAuto(expectedValveRadius=.195, mode='both', removeGroundMethod=removeGround):
    '''
    Automatically segment a valve hanging in front of the wall at the center
    '''

    # find the valve wall and its center
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    t = findWallCenter(polyData, removeGroundMethod)

    valve_point1 = [0, 0.6, 0]
    valveTransform1 = transformUtils.frameFromPositionAndRPY(valve_point1, [0, 0, 0])
    valveTransform1.Concatenate(t)
    point1 = np.array(valveTransform1.GetPosition())  # left of wall

    valve_point2 = [0, -0.6, 0]
    valveTransform2 = transformUtils.frameFromPositionAndRPY(valve_point2, [0, 0, 0])
    valveTransform2.Concatenate(t)
    point2 = np.array(valveTransform2.GetPosition())  # left of wall

    valve_point3 = [0, 1.0, 0]  # lever can over hang
    valveTransform3 = transformUtils.frameFromPositionAndRPY(valve_point3, [0, 0, 0])
    valveTransform3.Concatenate(t)
    point3 = valveTransform3.GetPosition()  # right of wall

    d = DebugData()
    d.addSphere(point2, radius=0.01)
    d.addSphere(point1, radius=0.03)
    d.addSphere(point3, radius=0.01)
    updatePolyData(d.getPolyData(), 'auto wall points', parent=getDebugFolder(), visible=False)

    if (mode == 'valve'):
        segmentValveByWallPlane(expectedValveRadius, point1, point2)
    elif (mode == 'lever'):
        segmentLeverByWallPlane(point1, point3)
    elif (mode == 'both'):
        segmentValveByWallPlane(expectedValveRadius, point1, point2)
        segmentLeverByWallPlane(point1, point3)
    else:
        raise Exception('unexpected segmentation mode: ' + mode)


def segmentLeverByWallPlane(point1, point2):
    '''
    determine the position (including rotation of a lever near a wall
    input is as for the valve - to points on the wall either side of the lever
    '''

    # 1. determine the wall plane and normal
    centerPoint = (point1 + point2) / 2.0

    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    viewDirection = SegmentationContext.getGlobalInstance().getViewDirection()
    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=-viewDirection, returnOrigin=True)

    # 2. Crop the cloud down to the lever only using the wall plane
    perpLine = np.cross(point2 - point1, -normal)
    # perpLine /= np.linalg.norm(perpLine)
    # perpLine * np.linalg.norm(point2 - point1)/2.0
    point3, point4 = centerPoint + perpLine / 2.0, centerPoint - perpLine / 2.0

    d = DebugData()
    d.addLine(point1, point2)
    d.addLine(point3, point4)
    updatePolyData(d.getPolyData(), 'lever crop lines', parent=getDebugFolder(), visible=False)

    wallPoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(wallPoints, 'lever valve wall', parent=getDebugFolder(), visible=False)

    searchRegion = thresholdPoints(polyData, 'dist_to_plane', [0.12, 0.2])  # very tight threshold
    searchRegion = cropToLineSegment(searchRegion, point1, point2)
    searchRegion = cropToLineSegment(searchRegion, point3, point4)
    updatePolyData(searchRegion, 'lever search region', parent=getDebugFolder(), color=[1, 0, 0], visible=False)

    # 3. fit line to remaining points - all assumed to be the lever
    linePoint, lineDirection, _ = applyLineFit(searchRegion, distanceThreshold=0.02)
    # if np.dot(lineDirection, forwardDirection) < 0:
    #    lineDirection = -lineDirection

    d = DebugData()
    d.addSphere(linePoint, radius=0.02)
    updatePolyData(d.getPolyData(), 'lever point', parent=getDebugFolder(), visible=False)

    pts = vtkNumpy.getNumpyFromVtk(searchRegion, 'Points')
    dists = np.dot(pts - linePoint, lineDirection)
    lever_center = linePoint + lineDirection * np.min(dists)
    lever_tip = linePoint + lineDirection * np.max(dists)

    # 4. determine which lever point is closest to the lower left of the wall. That's the lever_center point
    zaxis = -normal
    xaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(point1)

    # a distant point down and left from wall
    wall_point_lower_left = [-20, -20.0, 0]
    wall_point_lower_left_Transform = transformUtils.frameFromPositionAndRPY(wall_point_lower_left, [0, 0, 0])
    wall_point_lower_left_Transform.Concatenate(t)
    wall_point_lower_left = wall_point_lower_left_Transform.GetPosition()
    d1 = np.sqrt(np.sum((wall_point_lower_left - projectPointToPlane(lever_center, origin, normal)) ** 2))
    d2 = np.sqrt(np.sum((wall_point_lower_left - projectPointToPlane(lever_tip, origin, normal)) ** 2))

    if (d2 < d1):  # flip the points to match variable names
        p_temp = lever_center
        lever_center = lever_tip
        lever_tip = p_temp
        lineDirection = -lineDirection

    # 5. compute the rotation angle of the lever and, using that, its frame
    zaxis = -normal
    xaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(lever_center)  # nominal frame at lever center

    rotationAngle = -computeSignedAngleBetweenVectors(lineDirection, [0, 0, 1], -normal)
    t_lever = transformUtils.frameFromPositionAndRPY([0, 0, 0], [0, 0, math.degrees(rotationAngle)])
    t_lever.PostMultiply()
    t_lever.Concatenate(t)

    d = DebugData()
    # d.addSphere( point1 , radius=0.1)
    d.addSphere(wall_point_lower_left, radius=0.1)
    d.addSphere(lever_center, radius=0.04)
    d.addSphere(lever_tip, radius=0.01)
    d.addLine(lever_center, lever_tip)
    updatePolyData(d.getPolyData(), 'lever end points', color=[0, 1, 0], parent=getDebugFolder(), visible=False)

    radius = 0.01
    length = np.sqrt(np.sum((lever_tip - lever_center) ** 2))

    d = DebugData()
    d.addLine([0, 0, 0], [length, 0, 0], radius=radius)
    d.addSphere([0, 0, 0], 0.02)
    geometry = d.getPolyData()

    obj = showPolyData(geometry, 'valve lever', cls=FrameAffordanceItem, parent='affordances', color=[0, 1, 0],
                       visible=True)
    obj.actor.SetUserTransform(t_lever)
    obj.addToView(app.getDRCView())
    frameObj = showFrame(t_lever, 'lever frame', parent=obj, visible=False)
    frameObj.addToView(app.getDRCView())

    otdfType = 'lever_valve'
    params = dict(origin=np.array(t_lever.GetPosition()), xaxis=xaxis, yaxis=yaxis, zaxis=zaxis, xwidth=0.1, ywidth=0.1,
                  zwidth=0.1, radius=radius, length=length, friendly_name=otdfType, otdf_type=otdfType)
    obj.setAffordanceParams(params)
    obj.updateParamsFromActorTransform()


def applyICP(source, target):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.Update()
    t = vtk.vtkTransform()
    t.SetMatrix(icp.GetMatrix())
    return t


def applyDiskGlyphs(polyData, computeNormals=True):
    voxelGridLeafSize = 0.03
    normalEstimationSearchRadius = 0.05
    diskRadius = 0.015
    diskResolution = 12

    if computeNormals:
        scanInput = polyData

        pd = applyVoxelGrid(scanInput, leafSize=voxelGridLeafSize)

        pd = labelOutliers(pd, searchRadius=normalEstimationSearchRadius, neighborsInSearchRadius=3)
        pd = thresholdPoints(pd, 'is_outlier', [0, 0])

        pd = normalEstimation(pd, searchRadius=normalEstimationSearchRadius, searchCloud=scanInput)
    else:
        pd = polyData

    assert polyData.GetPointData().GetNormals()

    disk = vtk.vtkDiskSource()
    disk.SetOuterRadius(diskRadius)
    disk.SetInnerRadius(0.0)
    disk.SetRadialResolution(0)
    disk.SetCircumferentialResolution(diskResolution)
    disk.Update()

    t = vtk.vtkTransform()
    t.RotateY(90)
    disk = transformPolyData(disk.GetOutput(), t)

    glyph = vtk.vtkGlyph3D()
    glyph.ScalingOff()
    glyph.OrientOn()
    glyph.SetSource(disk)
    glyph.SetInput(pd)
    glyph.SetVectorModeToUseNormal()
    glyph.Update()

    return shallowCopy(glyph.GetOutput())


def applyArrowGlyphs(polyData, computeNormals=True, voxelGridLeafSize=0.03, normalEstimationSearchRadius=0.05,
                     arrowSize=0.02):
    if computeNormals:
        polyData = applyVoxelGrid(polyData, leafSize=0.02)
        voxelData = applyVoxelGrid(polyData, leafSize=voxelGridLeafSize)
        polyData = normalEstimation(polyData, searchRadius=normalEstimationSearchRadius, searchCloud=voxelData)
        polyData = removeNonFinitePoints(polyData, 'normals')
        flipNormalsWithViewDirection(polyData, SegmentationContext.getGlobalInstance().getViewDirection())

    assert polyData.GetPointData().GetNormals()

    arrow = vtk.vtkArrowSource()
    arrow.Update()

    glyph = vtk.vtkGlyph3D()
    glyph.SetScaleFactor(arrowSize)
    glyph.SetSource(arrow.GetOutput())
    glyph.SetInput(polyData)
    glyph.SetVectorModeToUseNormal()
    glyph.Update()

    return shallowCopy(glyph.GetOutput())


def segmentLeverValve(point1, point2):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())
    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=viewPlaneNormal, searchOrigin=point1,
                                             searchRadius=0.2, angleEpsilon=0.7, returnOrigin=True)

    wallPoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(wallPoints, 'wall points', parent=getDebugFolder(), visible=False)

    radius = 0.01
    length = 0.33

    normal = -normal  # set z to face into wall
    zaxis = normal
    xaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(point2)

    leverP1 = point2
    leverP2 = point2 + xaxis * length
    d = DebugData()
    d.addLine([0, 0, 0], [length, 0, 0], radius=radius)
    d.addSphere([0, 0, 0], 0.02)
    geometry = d.getPolyData()

    obj = showPolyData(geometry, 'valve lever', cls=FrameAffordanceItem, parent='affordances', color=[0, 1, 0],
                       visible=True)
    obj.actor.SetUserTransform(t)
    obj.addToView(app.getDRCView())
    frameObj = showFrame(t, 'lever frame', parent=obj, visible=False)
    frameObj.addToView(app.getDRCView())

    otdfType = 'lever_valve'
    params = dict(origin=np.array(t.GetPosition()), xaxis=xaxis, yaxis=yaxis, zaxis=zaxis, xwidth=0.1, ywidth=0.1,
                  zwidth=0.1, radius=radius, length=length, friendly_name=otdfType, otdf_type=otdfType)
    obj.setAffordanceParams(params)
    obj.updateParamsFromActorTransform()


def segmentWye(point1, point2):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())

    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=viewPlaneNormal, searchOrigin=point1,
                                             searchRadius=0.2, angleEpsilon=0.7, returnOrigin=True)

    wallPoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(wallPoints, 'wall points', parent=getDebugFolder(), visible=False)

    wyeMesh = ioUtils.readPolyData(os.path.join(app.getDRCBase(), 'software/models/otdf/wye.obj'))

    wyeMeshPoint = np.array([0.0, 0.0, 0.005])
    wyeMeshLeftHandle = np.array([0.032292, 0.02949, 0.068485])

    xaxis = -normal
    zaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PreMultiply()
    t.Translate(-wyeMeshPoint)
    t.PostMultiply()
    t.Translate(point2)

    d = DebugData()
    d.addSphere(point2, radius=0.005)
    updatePolyData(d.getPolyData(), 'wye pick point', parent=getDebugFolder(), visible=False)

    wyeObj = showPolyData(wyeMesh, 'wye', cls=FrameAffordanceItem, color=[0, 1, 0], visible=True)
    wyeObj.actor.SetUserTransform(t)
    wyeObj.addToView(app.getDRCView())
    frameObj = showFrame(t, 'wye frame', parent=wyeObj, visible=False)
    frameObj.addToView(app.getDRCView())

    params = dict(origin=np.array(t.GetPosition()), xaxis=xaxis, yaxis=yaxis, zaxis=zaxis, xwidth=0.1, ywidth=0.1,
                  zwidth=0.1, friendly_name='wye', otdf_type='wye')
    wyeObj.setAffordanceParams(params)
    wyeObj.updateParamsFromActorTransform()


def segmentDoorHandle(otdfType, point1, point2):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())

    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=viewPlaneNormal, searchOrigin=point1,
                                             searchRadius=0.2, angleEpsilon=0.7, returnOrigin=True)

    wallPoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(wallPoints, 'wall points', parent=getDebugFolder(), visible=False)

    handlePoint = np.array([0.005, 0.065, 0.011])

    xaxis = -normal
    zaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)

    xwidth = 0.01
    ywidth = 0.13
    zwidth = 0.022
    cube = vtk.vtkCubeSource()
    cube.SetXLength(xwidth)
    cube.SetYLength(ywidth)
    cube.SetZLength(zwidth)
    cube.Update()
    cube = shallowCopy(cube.GetOutput())

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    # t.PreMultiply()
    # t.Translate(-handlePoint)
    t.PostMultiply()
    t.Translate(point2)

    name = 'door handle'
    obj = showPolyData(cube, name, cls=FrameAffordanceItem, parent='affordances')
    obj.actor.SetUserTransform(t)
    obj.addToView(app.getDRCView())

    params = dict(origin=origin, xwidth=xwidth, ywidth=ywidth, zwidth=zwidth, xaxis=xaxis, yaxis=yaxis, zaxis=zaxis,
                  friendly_name=name, otdf_type=otdfType)
    obj.setAffordanceParams(params)
    obj.updateParamsFromActorTransform()

    frameObj = showFrame(obj.actor.GetUserTransform(), name + ' frame', parent=obj, visible=False)
    frameObj.addToView(app.getDRCView())


def segmentTruss(point1, point2):
    edge = point2 - point1
    edgeLength = np.linalg.norm(edge)

    stanceOffset = [-0.42, 0.0, 0.0]
    stanceYaw = 0.0

    d = DebugData()
    p1 = [0.0, 0.0, 0.0]
    p2 = -np.array([0.0, -1.0, 0.0]) * edgeLength
    d.addSphere(p1, radius=0.02)
    d.addSphere(p2, radius=0.02)
    d.addLine(p1, p2)

    stanceTransform = vtk.vtkTransform()
    stanceTransform.PostMultiply()
    stanceTransform.Translate(stanceOffset)
    # stanceTransform.RotateZ(stanceYaw)

    geometry = transformPolyData(d.getPolyData(), stanceTransform.GetLinearInverse())

    yaxis = edge / edgeLength
    zaxis = [0.0, 0.0, 1.0]
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)

    xwidth = 0.1
    ywidth = edgeLength
    zwidth = 0.1

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PreMultiply()
    t.Concatenate(stanceTransform)
    t.PostMultiply()
    t.Translate(point1)

    name = 'truss'
    otdfType = 'robot_knees'
    obj = showPolyData(geometry, name, cls=FrameAffordanceItem, parent='affordances')
    obj.actor.SetUserTransform(t)
    obj.addToView(app.getDRCView())

    params = dict(origin=t.GetPosition(), xwidth=xwidth, ywidth=ywidth, zwidth=zwidth, xaxis=xaxis, yaxis=yaxis,
                  zaxis=zaxis, friendly_name=name, otdf_type=otdfType)
    obj.setAffordanceParams(params)
    obj.updateParamsFromActorTransform()

    frameObj = showFrame(obj.actor.GetUserTransform(), name + ' frame', parent=obj, visible=False)
    frameObj.addToView(app.getDRCView())


def segmentHoseNozzle(point1):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    searchRegion = cropToSphere(polyData, point1, 0.10)
    updatePolyData(searchRegion, 'nozzle search region', parent=getDebugFolder(), visible=False)

    xaxis = [1, 0, 0]
    yaxis = [0, -1, 0]
    zaxis = [0, 0, -1]
    origin = point1

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(point1)

    nozzleRadius = 0.0266
    nozzleLength = 0.042
    nozzleTipRadius = 0.031
    nozzleTipLength = 0.024

    d = DebugData()
    d.addLine(np.array([0, 0, -nozzleLength / 2.0]), np.array([0, 0, nozzleLength / 2.0]), radius=nozzleRadius)
    d.addLine(np.array([0, 0, nozzleLength / 2.0]), np.array([0, 0, nozzleLength / 2.0 + nozzleTipLength]),
              radius=nozzleTipRadius)

    obj = showPolyData(d.getPolyData(), 'hose nozzle', cls=FrameAffordanceItem, color=[0, 1, 0], visible=True)
    obj.actor.SetUserTransform(t)
    obj.addToView(app.getDRCView())
    frameObj = showFrame(t, 'nozzle frame', parent=obj, visible=False)
    frameObj.addToView(app.getDRCView())

    params = dict(origin=origin, xaxis=xaxis, yaxis=yaxis, zaxis=zaxis, xwidth=0.1, ywidth=0.1, zwidth=0.1,
                  friendly_name='firehose', otdf_type='firehose')
    obj.setAffordanceParams(params)
    obj.updateParamsFromActorTransform()


def segmentDrillWall(point1, point2, point3):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    points = [point1, point2, point3]

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())
    expectedNormal = np.cross(point2 - point1, point3 - point1)
    expectedNormal /= np.linalg.norm(expectedNormal)
    if np.dot(expectedNormal, viewPlaneNormal) < 0:
        expectedNormal *= -1.0

    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=expectedNormal,
                                             searchOrigin=(point1 + point2 + point3) / 3.0, searchRadius=0.3,
                                             angleEpsilon=0.3, returnOrigin=True)

    points = [projectPointToPlane(point, origin, normal) for point in points]

    xaxis = -normal
    zaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(points[0])

    d = DebugData()
    pointsInWallFrame = []
    for p in points:
        pp = np.zeros(3)
        t.GetLinearInverse().TransformPoint(p, pp)
        pointsInWallFrame.append(pp)
        d.addSphere(pp, radius=0.02)

    for a, b in zip(pointsInWallFrame, pointsInWallFrame[1:] + [pointsInWallFrame[0]]):
        d.addLine(a, b, radius=0.015)

    aff = showPolyData(d.getPolyData(), 'drill target', cls=FrameAffordanceItem, color=[0, 1, 0], visible=True)
    aff.actor.SetUserTransform(t)
    showFrame(t, 'drill target frame', parent=aff, visible=False)
    refitWallCallbacks.append(functools.partial(refitDrillWall, aff))

    params = dict(origin=points[0], xaxis=xaxis, yaxis=yaxis, zaxis=zaxis, xwidth=0.1, ywidth=0.1, zwidth=0.1,
                  p1y=pointsInWallFrame[0][1], p1z=pointsInWallFrame[0][2],
                  p2y=pointsInWallFrame[1][1], p2z=pointsInWallFrame[1][2],
                  p3y=pointsInWallFrame[2][1], p3z=pointsInWallFrame[2][2],
                  friendly_name='drill_wall', otdf_type='drill_wall')

    aff.setAffordanceParams(params)
    aff.updateParamsFromActorTransform()
    aff.addToView(app.getDRCView())


refitWallCallbacks = []


def refitWall(point1):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())

    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=viewPlaneNormal, searchOrigin=point1,
                                             searchRadius=0.2, angleEpsilon=0.7, returnOrigin=True)

    wallPoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(wallPoints, 'wall points', parent=getDebugFolder(), visible=False)

    for func in refitWallCallbacks:
        func(point1, origin, normal)


def refitDrillWall(aff, point1, origin, normal):
    t = aff.actor.GetUserTransform()

    targetOrigin = np.array(t.GetPosition())

    projectedOrigin = projectPointToPlane(targetOrigin, origin, normal)
    projectedOrigin[2] = targetOrigin[2]

    xaxis = -normal
    zaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(projectedOrigin)
    aff.actor.GetUserTransform().SetMatrix(t.GetMatrix())


# this should be depreciated!
def getGroundHeightFromFeet():
    rfoot = getLinkFrame(drcargs.getDirectorConfig()['rightFootLink'])
    return np.array(rfoot.GetPosition())[2] - 0.0745342


# this should be depreciated!
def getTranslationRelativeToFoot(t):
    rfoot = getLinkFrame(drcargs.getDirectorConfig()['rightFootLink'])


def segmentDrillWallConstrained(rightAngleLocation, point1, point2):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())
    expectedNormal = np.cross(point2 - point1, [0.0, 0.0, 1.0])
    expectedNormal /= np.linalg.norm(expectedNormal)
    if np.dot(expectedNormal, viewPlaneNormal) < 0:
        expectedNormal *= -1.0

    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=expectedNormal, searchOrigin=point1,
                                             searchRadius=0.3, angleEpsilon=0.3, returnOrigin=True)

    triangleOrigin = projectPointToPlane(point2, origin, normal)

    xaxis = -normal
    zaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(triangleOrigin)

    createDrillWall(rightAngleLocation, t)


def createDrillWall(rightAngleLocation, trianglePose):
    # recover the origin and axes from the pose:
    triangleOrigin = trianglePose.GetPosition()
    xaxis, yaxis, zaxis = transformUtils.getAxesFromTransform(trianglePose)

    # 0.6096 = 24 * .0254 (m = feet)
    # 0.3048 = 12 * .0254 (m = feet)
    edgeRight = np.array([0.0, -1.0, 0.0]) * (0.6)
    edgeUp = np.array([0.0, 0.0, 1.0]) * (0.3)

    pointsInWallFrame = np.zeros((3, 3))

    if rightAngleLocation == DRILL_TRIANGLE_BOTTOM_LEFT:
        pointsInWallFrame[1] = edgeUp
        pointsInWallFrame[2] = edgeRight

    elif rightAngleLocation == DRILL_TRIANGLE_BOTTOM_RIGHT:
        pointsInWallFrame[1] = edgeUp  # edgeRight +edgeUp
        pointsInWallFrame[2] = -edgeRight  # edgeRight

    elif rightAngleLocation == DRILL_TRIANGLE_TOP_LEFT:
        pointsInWallFrame[1] = edgeRight
        pointsInWallFrame[2] = -edgeUp

    elif rightAngleLocation == DRILL_TRIANGLE_TOP_RIGHT:
        pointsInWallFrame[1] = edgeRight
        pointsInWallFrame[2] = edgeRight - edgeUp
    else:
        raise Exception('unexpected value for right angle location: ', + rightAngleLocation)

    center = pointsInWallFrame.sum(axis=0) / 3.0
    shrinkFactor = 1  # 0.90
    shrinkPoints = (pointsInWallFrame - center) * shrinkFactor + center

    d = DebugData()
    for p in pointsInWallFrame:
        d.addSphere(p, radius=0.015)

    for a, b in zip(pointsInWallFrame, np.vstack((pointsInWallFrame[1:], pointsInWallFrame[0]))):
        d.addLine(a, b, radius=0.005)  # 01)

    for a, b in zip(shrinkPoints, np.vstack((shrinkPoints[1:], shrinkPoints[0]))):
        d.addLine(a, b, radius=0.005)  # 0.025

    folder = om.getOrCreateContainer('affordances')

    wall = om.findObjectByName('wall')
    om.removeFromObjectModel(wall)

    aff = showPolyData(d.getPolyData(), 'wall', cls=FrameAffordanceItem, color=[0, 1, 0], visible=True, parent=folder)
    aff.actor.SetUserTransform(trianglePose)
    aff.addToView(app.getDRCView())

    refitWallCallbacks.append(functools.partial(refitDrillWall, aff))

    frameObj = showFrame(trianglePose, 'wall frame', parent=aff, scale=0.2, visible=False)
    frameObj.addToView(app.getDRCView())

    params = dict(origin=triangleOrigin, xaxis=xaxis, yaxis=yaxis, zaxis=zaxis, xwidth=0.1, ywidth=0.1, zwidth=0.1,
                  p1y=shrinkPoints[0][1], p1z=shrinkPoints[0][2],
                  p2y=shrinkPoints[1][1], p2z=shrinkPoints[1][2],
                  p3y=shrinkPoints[2][1], p3z=shrinkPoints[2][2],
                  friendly_name='drill_wall', otdf_type='drill_wall')

    aff.setAffordanceParams(params)
    aff.updateParamsFromActorTransform()

    '''
    rfoot = getLinkFrame(drcargs.getDirectorConfig()['rightFootLink'])
    tt = getTransformFromAxes(xaxis, yaxis, zaxis)
    tt.PostMultiply()
    tt.Translate(rfoot.GetPosition())
    showFrame(tt, 'rfoot with wall orientation')
    aff.footToAffTransform = computeAToB(tt, trianglePose)

    footToAff = list(aff.footToAffTransform.GetPosition())
    tt.TransformVector(footToAff, footToAff)

    d = DebugData()
    d.addSphere(tt.GetPosition(), radius=0.02)
    d.addLine(tt.GetPosition(), np.array(tt.GetPosition()) + np.array(footToAff))
    showPolyData(d.getPolyData(), 'rfoot debug')
    '''


def getDrillAffordanceParams(origin, xaxis, yaxis, zaxis, drillType="dewalt_button"):
    if (drillType == "dewalt_button"):
        params = dict(origin=origin, xaxis=xaxis, yaxis=yaxis, zaxis=zaxis, xwidth=0.1, ywidth=0.1, zwidth=0.1,
                      button_x=0.007,
                      button_y=-0.035,
                      button_z=-0.06,
                      button_roll=-90.0,
                      button_pitch=-90.0,
                      button_yaw=0.0,
                      bit_x=-0.01,
                      bit_y=0.0,
                      bit_z=0.15,
                      bit_roll=0,
                      bit_pitch=-90,
                      bit_yaw=0,
                      friendly_name='dewalt_button', otdf_type='dewalt_button')
    else:
        params = dict(origin=origin, xaxis=xaxis, yaxis=yaxis, zaxis=zaxis, xwidth=0.1, ywidth=0.1, zwidth=0.1,
                      button_x=0.007,
                      button_y=-0.035,
                      button_z=-0.06,
                      button_roll=0.0,
                      button_pitch=0.0,
                      button_yaw=0.0,
                      bit_x=0.18,
                      bit_y=0.0,
                      bit_z=0.13,
                      bit_roll=0,
                      bit_pitch=0,
                      bit_yaw=0,
                      friendly_name='dewalt_barrel', otdf_type='dewalt_barrel')

    return params


def getDrillMesh(applyBitOffset=False):
    button = np.array([0.007, -0.035, -0.06])
    drillMesh = ioUtils.readPolyData(os.path.join(app.getDRCBase(), 'software/models/otdf/dewalt_button.obj'))

    if applyBitOffset:
        t = vtk.vtkTransform()
        t.Translate(0.01, 0.0, 0.0)
        drillMesh = transformPolyData(drillMesh, t)

    d = DebugData()
    d.addPolyData(drillMesh)
    d.addSphere(button, radius=0.005, color=[0, 1, 0])
    d.addLine([0.0, 0.0, 0.155], [0.0, 0.0, 0.14], radius=0.001, color=[0, 1, 0])

    return shallowCopy(d.getPolyData())


def getDrillBarrelMesh():
    return ioUtils.readPolyData(os.path.join(app.getDRCBase(), 'software/models/otdf/dewalt.ply'), computeNormals=True)


def segmentDrill(point1, point2, point3):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())

    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=viewPlaneNormal, searchOrigin=point1,
                                             searchRadius=0.2, angleEpsilon=0.7, returnOrigin=True)

    tablePoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(tablePoints, 'table plane points', parent=getDebugFolder(), visible=False)

    searchRegion = thresholdPoints(polyData, 'dist_to_plane', [0.03, 0.4])
    searchRegion = cropToSphere(searchRegion, point2, 0.30)
    drillPoints = extractLargestCluster(searchRegion)

    drillToTopPoint = np.array([-0.002904, -0.010029, 0.153182])

    zaxis = normal
    yaxis = point3 - point2
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PreMultiply()
    t.Translate(-drillToTopPoint)
    t.PostMultiply()
    t.Translate(point2)

    drillMesh = getDrillMesh()

    aff = showPolyData(drillMesh, 'drill', cls=FrameAffordanceItem, visible=True)
    aff.actor.SetUserTransform(t)
    showFrame(t, 'drill frame', parent=aff, visible=False).addToView(app.getDRCView())

    params = getDrillAffordanceParams(origin, xaxis, yaxis, zaxis)
    aff.setAffordanceParams(params)
    aff.updateParamsFromActorTransform()
    aff.addToView(app.getDRCView())


def makePolyDataFields(pd):
    mesh = computeDelaunay3D(pd)

    if not mesh.GetNumberOfPoints():
        return None

    origin, edges, wireframe = getOrientedBoundingBox(mesh)

    edgeLengths = np.array([np.linalg.norm(edge) for edge in edges])
    axes = [edge / np.linalg.norm(edge) for edge in edges]

    # find axis nearest to the +/- up vector
    upVector = [0, 0, 1]
    dotProducts = [np.abs(np.dot(axe, upVector)) for axe in axes]
    zAxisIndex = np.argmax(dotProducts)

    # re-index axes and edge lengths so that the found axis is the z axis
    axesInds = [(zAxisIndex + 1) % 3, (zAxisIndex + 2) % 3, zAxisIndex]
    axes = [axes[i] for i in axesInds]
    edgeLengths = [edgeLengths[i] for i in axesInds]

    # flip if necessary so that z axis points toward up
    if np.dot(axes[2], upVector) < 0:
        axes[1] = -axes[1]
        axes[2] = -axes[2]

    boxCenter = computeCentroid(wireframe)

    t = getTransformFromAxes(axes[0], axes[1], axes[2])
    t.PostMultiply()
    t.Translate(boxCenter)

    pd = transformPolyData(pd, t.GetLinearInverse())
    wireframe = transformPolyData(wireframe, t.GetLinearInverse())
    mesh = transformPolyData(mesh, t.GetLinearInverse())

    return FieldContainer(points=pd, box=wireframe, mesh=mesh, frame=t, dims=edgeLengths, axes=axes)


def makeMovable(obj, initialTransform=None):
    '''
    Adds a child frame to the given PolyDataItem.  If initialTransform is not
    given, then an origin frame is computed for the polydata using the
    center and orientation of the oriented bounding of the polydata.  The polydata
    is transformed using the inverse of initialTransform and then a child frame
    is assigned to the object to reposition it.
    '''
    pd = obj.polyData
    t = initialTransform

    if t is None:
        origin, edges, wireframe = getOrientedBoundingBox(pd)
        edgeLengths = np.array([np.linalg.norm(edge) for edge in edges])
        axes = [edge / np.linalg.norm(edge) for edge in edges]
        boxCenter = computeCentroid(wireframe)
        t = getTransformFromAxes(axes[0], axes[1], axes[2])
        t.PostMultiply()
        t.Translate(boxCenter)

    pd = transformPolyData(pd, t.GetLinearInverse())
    obj.setPolyData(pd)

    frame = obj.getChildFrame()
    if frame:
        frame.copyFrame(t)
    else:
        frame = vis.showFrame(t, obj.getProperty('Name') + ' frame', parent=obj, scale=0.2, visible=False)
        obj.actor.SetUserTransform(t)


def segmentTable(polyData, searchPoint):
    '''
    NB: If you wish to use the table frame use segmentTableAndFrame instead
    ##################
    Segment a horizontal table surface (perpendicular to +Z) in the given polyData
    Input:
    - polyData
    - search point on plane

    Output:
    - polyData, tablePoints, origin, normal
    - polyData is the input polyData with a new 'dist_to_plane' attribute.
    '''
    expectedNormal = np.array([0.0, 0.0, 1.0])
    tableNormalEpsilon = 0.4

    polyData = applyVoxelGrid(polyData, leafSize=0.01)

    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=expectedNormal, perpendicularAxis=expectedNormal,
                                             searchOrigin=searchPoint, searchRadius=0.3,
                                             angleEpsilon=tableNormalEpsilon, returnOrigin=True)
    tablePoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])

    tablePoints = labelDistanceToPoint(tablePoints, searchPoint)
    tablePointsClusters = extractClusters(tablePoints, minClusterSize=10, clusterTolerance=0.1)
    tablePointsClusters.sort(key=lambda x: vtkNumpy.getNumpyFromVtk(x, 'distance_to_point').min())

    tablePoints = tablePointsClusters[0]

    updatePolyData(tablePoints, 'table plane points', parent=getDebugFolder(), visible=False)
    updatePolyData(tablePoints, 'table points', parent=getDebugFolder(), visible=False)

    return polyData, tablePoints, origin, normal


def filterClusterObjects(clusters):
    result = []
    for cluster in clusters:

        if np.abs(np.dot(cluster.axes[2], [0, 0, 1])) < 0.5:
            continue

        if cluster.dims[2] < 0.1:
            continue

        result.append(cluster)
    return result


def segmentTableScene(polyData, searchPoint, filterClustering=True):
    objectClusters, tableData = segmentTableSceneClusters(polyData, searchPoint)

    clusters = [makePolyDataFields(cluster) for cluster in objectClusters]
    clusters = [cluster for cluster in clusters if cluster is not None]

    # Add an additional frame to these objects which has z-axis aligned upwards
    # but rotated to have the x-axis facing away from the robot
    table_axes = transformUtils.getAxesFromTransform(tableData.frame)
    for cluster in clusters:
        cluster_axes = transformUtils.getAxesFromTransform(cluster.frame)

        zaxis = cluster_axes[2]
        xaxis = table_axes[0]
        yaxis = np.cross(zaxis, xaxis)
        xaxis = np.cross(yaxis, zaxis)
        xaxis /= np.linalg.norm(xaxis)
        yaxis /= np.linalg.norm(yaxis)
        orientedFrame = transformUtils.getTransformFromAxesAndOrigin(xaxis, yaxis, zaxis, cluster.frame.GetPosition())
        cluster._add_fields(oriented_frame=orientedFrame)

    if (filterClustering):
        clusters = filterClusterObjects(clusters)

    return FieldContainer(table=tableData, clusters=clusters)


def segmentTableSceneClusters(polyData, searchPoint, clusterInXY=False):
    ''' Given a point cloud of a table with some objects on it
        and a point on that table
        determine the plane of the table and
        extract clusters above the table
    '''

    tableData, polyData = segmentTableAndFrame(polyData, searchPoint)

    searchRegion = thresholdPoints(polyData, 'dist_to_plane', [0.02, 0.5])
    # TODO: replace with 'all points above the table':
    searchRegion = cropToSphere(searchRegion, tableData.frame.GetPosition(), 0.5)  # was 1.0

    showFrame(tableData.frame, 'tableFrame', visible=False, parent=getDebugFolder(), scale=0.15)
    showPolyData(searchRegion, 'searchRegion', color=[1, 0, 0], visible=False, parent=getDebugFolder())

    objectClusters = extractClusters(searchRegion, clusterInXY, clusterTolerance=0.02, minClusterSize=10)

    # print 'got %d clusters' % len(objectClusters)
    for i, c in enumerate(objectClusters):
        name = "cluster %d" % i
        showPolyData(c, name, color=getRandomColor(), visible=False, parent=getDebugFolder())

    return objectClusters, tableData


def segmentTableAndFrame(polyData, searchPoint):
    '''
    Segment a table using a searchPoint on the table top
    and then recover its coordinate frame, facing away from the robot
    Objects/points on the table are ignored

    Input: polyData and searchPoint on the table

    Output: FieldContainer with:
    - all relevent details about the table (only)

    '''

    polyData, tablePoints, _, _ = segmentTable(polyData, searchPoint)
    tableMesh = computeDelaunay3D(tablePoints)

    viewFrame = SegmentationContext.getGlobalInstance().getViewFrame()
    viewDirection = SegmentationContext.getGlobalInstance().getViewDirection()
    robotYaw = math.atan2(viewDirection[1], viewDirection[0]) * 180.0 / np.pi
    linkFrame = transformUtils.frameFromPositionAndRPY(viewFrame.GetPosition(), [0, 0, robotYaw])

    # Function returns corner point that is far right from the robot
    cornerTransform, rectDepth, rectWidth, _ = findMinimumBoundingRectangle(tablePoints, linkFrame)
    rectHeight = 0.02  # arbitrary table width

    # recover mid point
    t = transformUtils.copyFrame(cornerTransform)
    t.PreMultiply()
    table_center = [-rectDepth / 2, rectWidth / 2, 0]
    t3 = transformUtils.frameFromPositionAndRPY(table_center, [0, 0, 0])
    t.Concatenate(t3)

    # Create required outputs
    edgeLengths = [rectDepth, rectWidth, rectHeight]
    tableXAxis, tableYAxis, tableZAxis = transformUtils.getAxesFromTransform(t)
    axes = tableXAxis, tableYAxis, tableZAxis
    wf = vtk.vtkOutlineSource()
    wf.SetBounds([-rectDepth / 2, rectDepth / 2, -rectWidth / 2, rectWidth / 2, -rectHeight / 2, rectHeight / 2])
    # wf.SetBoxTypeToOriented()
    # cube =[0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,1]
    # wf.SetCorners(cube)
    wireframe = wf.GetOutput()

    tablePoints = transformPolyData(tablePoints, t.GetLinearInverse())
    # wireframe = transformPolyData(wireframe, t.GetLinearInverse())
    tableMesh = transformPolyData(tableMesh, t.GetLinearInverse())

    return FieldContainer(points=tablePoints, box=wireframe, mesh=tableMesh, frame=t, dims=edgeLengths,
                          axes=axes), polyData


def segmentDrillAuto(point1, polyData=None):
    if polyData is None:
        inputObj = om.findObjectByName('pointcloud snapshot')
        polyData = inputObj.polyData

    expectedNormal = np.array([0.0, 0.0, 1.0])

    polyData, origin, normal = applyPlaneFit(polyData, expectedNormal=expectedNormal, perpendicularAxis=expectedNormal,
                                             searchOrigin=point1, searchRadius=0.4, angleEpsilon=0.2, returnOrigin=True)

    tablePoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(tablePoints, 'table plane points', parent=getDebugFolder(), visible=False)

    tablePoints = labelDistanceToPoint(tablePoints, point1)
    tablePointsClusters = extractClusters(tablePoints)
    tablePointsClusters.sort(key=lambda x: vtkNumpy.getNumpyFromVtk(x, 'distance_to_point').min())

    tablePoints = tablePointsClusters[0]
    updatePolyData(tablePoints, 'table points', parent=getDebugFolder(), visible=False)

    searchRegion = thresholdPoints(polyData, 'dist_to_plane', [0.03, 0.4])
    searchRegion = cropToSphere(searchRegion, point1, 0.30)
    drillPoints = extractLargestCluster(searchRegion, minClusterSize=1)

    # determine drill orientation (rotation about z axis)

    centroids = computeCentroids(drillPoints, axis=normal)

    centroidsPolyData = vtkNumpy.getVtkPolyDataFromNumpyPoints(centroids)
    d = DebugData()
    updatePolyData(centroidsPolyData, 'cluster centroids', parent=getDebugFolder(), visible=False)

    drillToTopPoint = np.array([-0.002904, -0.010029, 0.153182])

    zaxis = normal
    yaxis = centroids[0] - centroids[-1]
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)

    # note this hack to orient the drill correctly:
    t = getTransformFromAxes(yaxis, -xaxis, zaxis)
    t.PreMultiply()
    t.Translate(-drillToTopPoint)
    t.PostMultiply()
    t.Translate(centroids[-1])

    drillMesh = getDrillMesh()

    aff = showPolyData(drillMesh, 'drill', cls=FrameAffordanceItem, visible=True)
    aff.actor.SetUserTransform(t)
    showFrame(t, 'drill frame', parent=aff, visible=False, scale=0.2).addToView(app.getDRCView())

    params = getDrillAffordanceParams(origin, xaxis, yaxis, zaxis)
    aff.setAffordanceParams(params)
    aff.updateParamsFromActorTransform()
    aff.addToView(app.getDRCView())


def segmentDrillButton(point1):
    d = DebugData()
    d.addSphere([0, 0, 0], radius=0.005)
    obj = updatePolyData(d.getPolyData(), 'sensed drill button', color=[0, 0.5, 0.5], visible=True)

    # there is no orientation, but this allows the XYZ point to be queried
    pointerTipFrame = transformUtils.frameFromPositionAndRPY(point1, [0, 0, 0])
    obj.actor.SetUserTransform(pointerTipFrame)
    obj.addToView(app.getDRCView())

    frameObj = updateFrame(obj.actor.GetUserTransform(), 'sensed drill button frame', parent=obj, scale=0.2,
                           visible=False)
    frameObj.addToView(app.getDRCView())


def segmentPointerTip(point1):
    d = DebugData()
    d.addSphere([0, 0, 0], radius=0.005)
    obj = updatePolyData(d.getPolyData(), 'sensed pointer tip', color=[0.5, 0.5, 0.0], visible=True)

    # there is no orientation, but this allows the XYZ point to be queried
    pointerTipFrame = transformUtils.frameFromPositionAndRPY(point1, [0, 0, 0])
    obj.actor.SetUserTransform(pointerTipFrame)
    obj.addToView(app.getDRCView())

    frameObj = updateFrame(obj.actor.GetUserTransform(), 'sensed pointer tip frame', parent=obj, scale=0.2,
                           visible=False)
    frameObj.addToView(app.getDRCView())


def fitGroundObject(polyData=None, expectedDimensionsMin=[0.2, 0.02], expectedDimensionsMax=[1.3, 0.1]):
    removeGroundFunc = removeGroundSimple

    polyData = polyData or getCurrentRevolutionData()
    groundPoints, scenePoints = removeGroundFunc(polyData, groundThickness=0.02, sceneHeightFromGround=0.035)

    searchRegion = thresholdPoints(scenePoints, 'dist_to_plane', [0.05, 0.2])

    clusters = extractClusters(searchRegion, clusterTolerance=0.07, minClusterSize=4)

    candidates = []
    for clusterId, cluster in enumerate(clusters):

        origin, edges, _ = getOrientedBoundingBox(cluster)
        edgeLengths = [np.linalg.norm(edge) for edge in edges[:2]]

        found = (expectedDimensionsMin[0] <= edgeLengths[0] < expectedDimensionsMax[0]
                 and expectedDimensionsMin[1] <= edgeLengths[1] < expectedDimensionsMax[1])

        if not found:
            updatePolyData(cluster, 'candidate cluster %d' % clusterId, color=[1, 1, 0], parent=getDebugFolder(),
                           visible=False)
            continue

        updatePolyData(cluster, 'cluster %d' % clusterId, color=[0, 1, 0], parent=getDebugFolder(), visible=False)
        candidates.append(cluster)

    if not candidates:
        return None

    viewFrame = SegmentationContext.getGlobalInstance().getViewFrame()
    viewOrigin = np.array(viewFrame.GetPosition())

    dists = [np.linalg.norm(viewOrigin - computeCentroid(cluster)) for cluster in candidates]
    candidates = [candidates[i] for i in np.argsort(dists)]

    cluster = candidates[0]
    obj = makePolyDataFields(cluster)

    return vis.showClusterObjects([obj], parent='segmentation')[0]


def findHorizontalSurfaces(polyData, removeGroundFirst=False, normalEstimationSearchRadius=0.05,
                           clusterTolerance=0.025, minClusterSize=150, distanceToPlaneThreshold=0.0025,
                           normalsDotUpRange=[0.95, 1.0], showClusters=False):
    '''
    Find the horizontal surfaces, tuned to work with walking terrain
    '''

    searchZ = [0.0, 2.0]
    voxelGridLeafSize = 0.01
    verboseFlag = False

    if (removeGroundFirst):
        groundPoints, scenePoints = removeGround(polyData, groundThickness=0.02, sceneHeightFromGround=0.05)
        scenePoints = thresholdPoints(scenePoints, 'dist_to_plane', searchZ)
        updatePolyData(groundPoints, 'ground points', parent=getDebugFolder(), visible=verboseFlag)
    else:
        scenePoints = polyData

    if not scenePoints.GetNumberOfPoints():
        return

    f = vtk.vtkPCLNormalEstimation()
    f.SetSearchRadius(normalEstimationSearchRadius)
    f.SetInput(scenePoints)
    f.SetInput(1, applyVoxelGrid(scenePoints, voxelGridLeafSize))

    # Duration 0.2 sec for V1 log:
    f.Update()
    scenePoints = shallowCopy(f.GetOutput())

    normals = vtkNumpy.getNumpyFromVtk(scenePoints, 'normals')
    normalsDotUp = np.abs(np.dot(normals, [0, 0, 1]))

    vtkNumpy.addNumpyToVtk(scenePoints, normalsDotUp, 'normals_dot_up')
    surfaces = thresholdPoints(scenePoints, 'normals_dot_up', normalsDotUpRange)

    updatePolyData(scenePoints, 'scene points', parent=getDebugFolder(), colorByName='normals_dot_up',
                   visible=verboseFlag)
    updatePolyData(surfaces, 'surfaces points', parent=getDebugFolder(), colorByName='normals_dot_up',
                   visible=verboseFlag)

    clusters = extractClusters(surfaces, clusterTolerance=clusterTolerance, minClusterSize=minClusterSize)
    planeClusters = []
    clustersLarge = []

    om.removeFromObjectModel(om.findObjectByName('surface clusters'))
    folder = om.getOrCreateContainer('surface clusters', parentObj=getDebugFolder())

    for i, cluster in enumerate(clusters):

        updatePolyData(cluster, 'surface cluster %d' % i, parent=folder, color=getRandomColor(), visible=verboseFlag)
        planePoints, _ = applyPlaneFit(cluster, distanceToPlaneThreshold)
        planePoints = thresholdPoints(planePoints, 'dist_to_plane',
                                      [-distanceToPlaneThreshold, distanceToPlaneThreshold])

        if planePoints.GetNumberOfPoints() > minClusterSize:
            clustersLarge.append(cluster)
            obj = makePolyDataFields(planePoints)
            if obj is not None:
                planeClusters.append(obj)

    folder = om.getOrCreateContainer('surface objects', parentObj=getDebugFolder())
    if showClusters:
        vis.showClusterObjects(planeClusters, parent=folder)

    return clustersLarge


def fitVerticalPosts(polyData):
    groundPoints, scenePoints = removeGround(polyData)
    scenePoints = thresholdPoints(scenePoints, 'dist_to_plane', [0.1, 4.0])

    if not scenePoints.GetNumberOfPoints():
        return

    scenePoints = applyVoxelGrid(scenePoints, leafSize=0.03)
    clusters = extractClusters(scenePoints, clusterTolerance=0.15, minClusterSize=10)

    def isPostCluster(cluster, lineDirection):

        up = [0, 0, 1]
        minPostLength = 1.0
        maxRadius = 0.3
        angle = math.degrees(
            math.acos(np.dot(up, lineDirection) / (np.linalg.norm(up) * np.linalg.norm(lineDirection))))

        if angle > 15:
            return False

        origin, edges, _ = getOrientedBoundingBox(cluster)
        edgeLengths = [np.linalg.norm(edge) for edge in edges]

        if edgeLengths[0] < minPostLength:
            return False

        # extract top half
        zvalues = vtkNumpy.getNumpyFromVtk(cluster, 'Points')[:, 2].copy()
        vtkNumpy.addNumpyToVtk(cluster, zvalues, 'z')

        minZ = np.min(zvalues)
        maxZ = np.max(zvalues)

        cluster = thresholdPoints(cluster, 'z', [(minZ + maxZ) / 2.0, maxZ])
        origin, edges, _ = getOrientedBoundingBox(cluster)
        edgeLengths = [np.linalg.norm(edge) for edge in edges]

        if edgeLengths[1] > maxRadius or edgeLengths[2] > maxRadius:
            return False

        return True

    def makeCylinderAffordance(linePoints, lineDirection, lineOrigin, postId):

        pts = vtkNumpy.getNumpyFromVtk(linePoints, 'Points')
        dists = np.dot(pts - lineOrigin, lineDirection)
        p1 = lineOrigin + lineDirection * np.min(dists)
        p2 = lineOrigin + lineDirection * np.max(dists)

        origin = (p1 + p2) / 2.0
        lineLength = np.linalg.norm(p2 - p1)
        t = transformUtils.getTransformFromOriginAndNormal(origin, lineDirection)
        pose = transformUtils.poseFromTransform(t)

        desc = dict(classname='CylinderAffordanceItem', Name='post %d' % postId,
                    uuid=newUUID(), pose=pose, Radius=0.05, Length=float(lineLength), Color=[0.0, 1.0, 0.0])
        desc['Collision Enabled'] = True

        return affordanceManager.newAffordanceFromDescription(desc)

    rejectFolder = om.getOrCreateContainer('nonpost clusters', parentObj=getDebugFolder())
    keepFolder = om.getOrCreateContainer('post clusters', parentObj=getDebugFolder())

    for i, cluster in enumerate(clusters):

        linePoint, lineDirection, linePoints = applyLineFit(cluster, distanceThreshold=0.1)
        if isPostCluster(cluster, lineDirection):
            vis.showPolyData(cluster, 'cluster %d' % i, visible=False, color=getRandomColor(), alpha=0.5,
                             parent=keepFolder)
            makeCylinderAffordance(linePoints, lineDirection, linePoint, i)
        else:
            vis.showPolyData(cluster, 'cluster %d' % i, visible=False, color=getRandomColor(), alpha=0.5,
                             parent=rejectFolder)


def findAndFitDrillBarrel(polyData=None):
    ''' Find the horizontal surfaces
    on the horizontal surfaces, find all the drills
    '''

    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = polyData or inputObj.polyData

    groundPoints, scenePoints = removeGround(polyData, groundThickness=0.02, sceneHeightFromGround=0.50)

    scenePoints = thresholdPoints(scenePoints, 'dist_to_plane', [0.5, 1.7])

    if not scenePoints.GetNumberOfPoints():
        return

    normalEstimationSearchRadius = 0.10

    f = vtk.vtkPCLNormalEstimation()
    f.SetSearchRadius(normalEstimationSearchRadius)
    f.SetInput(scenePoints)
    f.Update()
    scenePoints = shallowCopy(f.GetOutput())

    normals = vtkNumpy.getNumpyFromVtk(scenePoints, 'normals')
    normalsDotUp = np.abs(np.dot(normals, [0, 0, 1]))

    vtkNumpy.addNumpyToVtk(scenePoints, normalsDotUp, 'normals_dot_up')

    surfaces = thresholdPoints(scenePoints, 'normals_dot_up', [0.95, 1.0])

    updatePolyData(groundPoints, 'ground points', parent=getDebugFolder(), visible=False)
    updatePolyData(scenePoints, 'scene points', parent=getDebugFolder(), colorByName='normals_dot_up', visible=False)
    updatePolyData(surfaces, 'surfaces', parent=getDebugFolder(), visible=False)

    clusters = extractClusters(surfaces, clusterTolerance=0.15, minClusterSize=50)

    fitResults = []

    viewFrame = SegmentationContext.getGlobalInstance().getViewFrame()
    forwardDirection = np.array([1.0, 0.0, 0.0])
    viewFrame.TransformVector(forwardDirection, forwardDirection)
    robotOrigin = viewFrame.GetPosition()
    robotForward = forwardDirection

    # print 'robot origin:', robotOrigin
    # print 'robot forward:', robotForward
    centroid = []

    for clusterId, cluster in enumerate(clusters):
        clusterObj = updatePolyData(cluster, 'surface cluster %d' % clusterId, color=[1, 1, 0], parent=getDebugFolder(),
                                    visible=False)

        origin, edges, _ = getOrientedBoundingBox(cluster)
        edgeLengths = [np.linalg.norm(edge) for edge in edges[:2]]

        skipCluster = False
        for edgeLength in edgeLengths:
            # print 'cluster %d edge length: %f' % (clusterId, edgeLength)
            if edgeLength < 0.35 or edgeLength > 0.75:
                skipCluster = True

        if skipCluster:
            continue

        clusterObj.setSolidColor([0, 0, 1])
        centroid = np.average(vtkNumpy.getNumpyFromVtk(cluster, 'Points'), axis=0)

        try:
            drillFrame = segmentDrillBarrelFrame(centroid, polyData=scenePoints, forwardDirection=robotForward)
            if drillFrame is not None:
                fitResults.append((clusterObj, drillFrame))
        except:
            print traceback.format_exc()
            print 'fit drill failed for cluster:', clusterId

    if not fitResults:
        return

    sortFittedDrills(fitResults, robotOrigin, robotForward)

    return centroid


def sortFittedDrills(fitResults, robotOrigin, robotForward):
    angleToFitResults = []

    for fitResult in fitResults:
        cluster, drillFrame = fitResult
        drillOrigin = np.array(drillFrame.GetPosition())
        angleToDrill = np.abs(computeSignedAngleBetweenVectors(robotForward, drillOrigin - robotOrigin, [0, 0, 1]))
        angleToFitResults.append((angleToDrill, cluster, drillFrame))
        # print 'angle to candidate drill:', angleToDrill

    angleToFitResults.sort(key=lambda x: x[0])

    # print 'using drill at angle:', angleToFitResults[0][0]

    drillMesh = getDrillBarrelMesh()

    for i, fitResult in enumerate(angleToFitResults):

        angleToDrill, cluster, drillFrame = fitResult

        if i == 0:

            drill = om.findObjectByName('drill')
            drill = updatePolyData(drillMesh, 'drill', color=[0, 1, 0], cls=FrameAffordanceItem, visible=True)
            drillFrame = updateFrame(drillFrame, 'drill frame', parent=drill, visible=False)
            drill.actor.SetUserTransform(drillFrame.transform)

            drill.setAffordanceParams(dict(otdf_type='dewalt_button', friendly_name='dewalt_button'))
            drill.updateParamsFromActorTransform()

            drill.setSolidColor([0, 1, 0])
            # cluster.setProperty('Visible', True)

        else:

            drill = showPolyData(drillMesh, 'drill candidate', color=[1, 0, 0], visible=False, parent=getDebugFolder())
            drill.actor.SetUserTransform(drillFrame)
            om.addToObjectModel(drill, parentObj=getDebugFolder())


def computeSignedAngleBetweenVectors(v1, v2, perpendicularVector):
    '''
    Computes the signed angle between two vectors in 3d, given a perpendicular vector
    to determine sign.  Result returned is radians.
    '''
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    perpendicularVector = np.array(perpendicularVector, dtype=float)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    perpendicularVector /= np.linalg.norm(perpendicularVector)
    return math.atan2(np.dot(perpendicularVector, np.cross(v1, v2)), np.dot(v1, v2))


def segmentDrillBarrelFrame(point1, polyData, forwardDirection):
    tableClusterSearchRadius = 0.4
    drillClusterSearchRadius = 0.5  # 0.3

    expectedNormal = np.array([0.0, 0.0, 1.0])

    if not polyData.GetNumberOfPoints():
        return

    polyData, plane_origin, plane_normal = applyPlaneFit(polyData, expectedNormal=expectedNormal,
                                                         perpendicularAxis=expectedNormal, searchOrigin=point1,
                                                         searchRadius=tableClusterSearchRadius, angleEpsilon=0.2,
                                                         returnOrigin=True)

    if not polyData.GetNumberOfPoints():
        return

    tablePoints = thresholdPoints(polyData, 'dist_to_plane', [-0.01, 0.01])
    updatePolyData(tablePoints, 'table plane points', parent=getDebugFolder(), visible=False)

    tablePoints = labelDistanceToPoint(tablePoints, point1)
    tablePointsClusters = extractClusters(tablePoints)
    tablePointsClusters.sort(key=lambda x: vtkNumpy.getNumpyFromVtk(x, 'distance_to_point').min())

    if not tablePointsClusters:
        return

    tablePoints = tablePointsClusters[0]
    updatePolyData(tablePoints, 'table points', parent=getDebugFolder(), visible=False)

    searchRegion = thresholdPoints(polyData, 'dist_to_plane', [0.02, 0.3])
    if not searchRegion.GetNumberOfPoints():
        return

    searchRegion = cropToSphere(searchRegion, point1, drillClusterSearchRadius)
    # drillPoints = extractLargestCluster(searchRegion, minClusterSize=1)

    t = fitDrillBarrel(searchRegion, forwardDirection, plane_origin, plane_normal)
    return t


def segmentDrillBarrel(point1):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    forwardDirection = -np.array(getCurrentView().camera().GetViewPlaneNormal())

    t = segmentDrillBarrel(point1, polyData, forwardDirection)
    assert t is not None

    drillMesh = getDrillBarrelMesh()

    aff = showPolyData(drillMesh, 'drill', visible=True)
    aff.addToView(app.getDRCView())

    aff.actor.SetUserTransform(t)
    drillFrame = showFrame(t, 'drill frame', parent=aff, visible=False)
    drillFrame.addToView(app.getDRCView())
    return aff, drillFrame


def segmentDrillAlignedWithTable(point, polyData=None):
    '''
    Yet Another Drill Fitting Algorithm [tm]
    This one fits the button drill assuming its on the table
    and aligned with the table frame (because the button drill orientation is difficult to find)
    Table must have long side facing robot
    '''
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = polyData or inputObj.polyData

    # segment the table and recover the precise up direction normal:
    polyDataOut, tablePoints, origin, normal = segmentTable(polyData, point)
    # print origin # this origin is bunk
    # tableCentroid = computeCentroid(tablePoints)

    # get the bounding box edges
    OBBorigin, edges, _ = getOrientedBoundingBox(tablePoints)
    # print "OBB out"
    # print OBBorigin
    # print edges
    edgeLengths = np.array([np.linalg.norm(edge) for edge in edges])
    axes = [edge / np.linalg.norm(edge) for edge in edges]
    # print edgeLengths
    # print axes

    # check which direction the robot is facing and flip x-axis of table if necessary
    viewDirection = SegmentationContext.getGlobalInstance().getViewDirection()
    # print "main axes", axes[1]
    # print "viewDirection", viewDirection
    # dp = np.dot(axes[1], viewDirection)
    # print dp

    if np.dot(axes[1], viewDirection) < 0:
        # print "flip the x-direction"
        axes[1] = -axes[1]

    # define the x-axis to be along the 2nd largest edge
    xaxis = axes[1]
    xaxis = np.array(xaxis)
    zaxis = np.array(normal)
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    tableOrientation = transformUtils.getTransformFromAxes(xaxis, yaxis, zaxis)

    # tableTransform = transformUtils.frameFromPositionAndRPY( tableCentroid , tableOrientation.GetOrientation() )
    # updateFrame(tableTransform, 'table frame [z up, x away face]', parent="segmentation", visible=True).addToView(app.getDRCView())

    data = segmentTableScene(polyData, point)
    # vis.showClusterObjects(data.clusters + [data.table], parent='segmentation')

    # crude use of the table frame to determine the frame of the drill on the table
    # t2 = transformUtils.frameFromPositionAndRPY([0,0,0], [180, 0 , 90] )
    # drillOrientationTransform = transformUtils.copyFrame( om.findObjectByName('object 1 frame').transform )
    # drillOrientationTransform.PreMultiply()
    # drillOrientationTransform.Concatenate(t2)
    # vis.updateFrame(t, 'drillOrientationTransform',visible=True)

    # table_xaxis, table_yaxis, table_zaxis = transformUtils.getAxesFromTransform( data.table.frame )
    # drillOrientation = transformUtils.orientationFromAxes( table_yaxis, table_xaxis,  -1*np.array( table_zaxis) )
    drillTransform = transformUtils.frameFromPositionAndRPY(data.clusters[0].frame.GetPosition(),
                                                            tableOrientation.GetOrientation())

    drillMesh = getDrillMesh()

    drill = om.findObjectByName('drill')
    om.removeFromObjectModel(drill)

    aff = showPolyData(drillMesh, 'drill', color=[0.0, 1.0, 0.0], cls=FrameAffordanceItem, visible=True)
    aff.actor.SetUserTransform(drillTransform)
    aff.addToView(app.getDRCView())

    frameObj = updateFrame(drillTransform, 'drill frame', parent=aff, scale=0.2, visible=False)
    frameObj.addToView(app.getDRCView())

    params = getDrillAffordanceParams(np.array(drillTransform.GetPosition()), [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                      drillType="dewalt_button")
    aff.setAffordanceParams(params)


def segmentDrillInHand(p1, p2):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    distanceToLineThreshold = 0.05

    polyData = labelDistanceToLine(polyData, p1, p2)
    polyData = thresholdPoints(polyData, 'distance_to_line', [0.0, distanceToLineThreshold])

    lineSegment = p2 - p1
    lineLength = np.linalg.norm(lineSegment)

    cropped, polyData = cropToPlane(polyData, p1, lineSegment / lineLength, [-0.03, lineLength + 0.03])

    updatePolyData(cropped, 'drill cluster', parent=getDebugFolder(), visible=False)

    drillPoints = cropped
    normal = lineSegment / lineLength

    centroids = computeCentroids(drillPoints, axis=normal)

    centroidsPolyData = vtkNumpy.getVtkPolyDataFromNumpyPoints(centroids)
    d = DebugData()
    updatePolyData(centroidsPolyData, 'cluster centroids', parent=getDebugFolder(), visible=False)

    drillToTopPoint = np.array([-0.002904, -0.010029, 0.153182])

    zaxis = normal
    yaxis = centroids[0] - centroids[-1]
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis = np.cross(zaxis, xaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PreMultiply()
    t.Translate(-drillToTopPoint)
    t.PostMultiply()
    t.Translate(p2)

    drillMesh = getDrillMesh()

    aff = showPolyData(drillMesh, 'drill', cls=FrameAffordanceItem, visible=True)
    aff.actor.SetUserTransform(t)
    showFrame(t, 'drill frame', parent=aff, visible=False).addToView(app.getDRCView())

    params = getDrillAffordanceParams(np.array(t.GetPosition()), xaxis, yaxis, zaxis)
    aff.setAffordanceParams(params)
    aff.updateParamsFromActorTransform()
    aff.addToView(app.getDRCView())


def addDrillAffordance():
    drillMesh = getDrillMesh()

    aff = showPolyData(drillMesh, 'drill', cls=FrameAffordanceItem, visible=True)
    t = vtk.vtkTransform()
    t.PostMultiply()
    aff.actor.SetUserTransform(t)
    showFrame(t, 'drill frame', parent=aff, visible=False).addToView(app.getDRCView())

    params = getDrillAffordanceParams(np.array(t.GetPosition()), [1, 0, 0], [0, 1, 0], [0, 0, 1])
    aff.setAffordanceParams(params)
    aff.updateParamsFromActorTransform()
    aff.addToView(app.getDRCView())
    return aff


def getLinkFrame(linkName):
    robotStateModel = om.findObjectByName('robot state model')
    assert robotStateModel
    t = vtk.vtkTransform()
    robotStateModel.model.getLinkToWorld(linkName, t)
    return t


def getDrillInHandOffset(zRotation=0.0, zTranslation=0.0, xTranslation=0.0, yTranslation=0.0, flip=False):
    drillOffset = vtk.vtkTransform()
    drillOffset.PostMultiply()
    if flip:
        drillOffset.RotateY(180)
    drillOffset.RotateZ(zRotation)
    drillOffset.RotateY(-90)
    # drillOffset.Translate(0, 0.09, zTranslation - 0.015)
    # drillOffset.Translate(zTranslation - 0.015, 0.035 + xTranslation, 0.0)
    drillOffset.Translate(zTranslation, xTranslation, 0.0 + yTranslation)
    return drillOffset


def moveDrillToHand(drillOffset, hand='right'):
    drill = om.findObjectByName('drill')
    if not drill:
        drill = addDrillAffordance()

    assert hand in ('right', 'left')
    drillTransform = drill.actor.GetUserTransform()
    rightBaseLink = getLinkFrame('%s_hand_face' % hand[0])
    drillTransform.PostMultiply()
    drillTransform.Identity()
    drillTransform.Concatenate(drillOffset)
    drillTransform.Concatenate(rightBaseLink)
    drill._renderAllViews()


class PointPicker(TimerCallback):
    def __init__(self, numberOfPoints=3):
        TimerCallback.__init__(self)
        self.targetFps = 30
        self.enabled = False
        self.numberOfPoints = numberOfPoints
        self.annotationObj = None
        self.drawLines = True
        self.clear()

    def clear(self):
        self.points = [None for i in xrange(self.numberOfPoints)]
        self.hoverPos = None
        self.annotationFunc = None
        self.lastMovePos = [0, 0]

    def onMouseMove(self, displayPoint, modifiers=None):
        self.lastMovePos = displayPoint

    def onMousePress(self, displayPoint, modifiers=None):

        # print 'mouse press:', modifiers
        # if not modifiers:
        #    return

        for i in xrange(self.numberOfPoints):
            if self.points[i] is None:
                self.points[i] = self.hoverPos
                break

        if self.points[-1] is not None:
            self.finish()

    def finish(self):

        self.enabled = False
        om.removeFromObjectModel(self.annotationObj)

        points = [p.copy() for p in self.points]
        if self.annotationFunc is not None:
            self.annotationFunc(*points)

        removeViewPicker(self)

    def handleRelease(self, displayPoint):
        pass

    def draw(self):

        d = DebugData()

        points = [p if p is not None else self.hoverPos for p in self.points]

        # draw points
        for p in points:
            if p is not None:
                d.addSphere(p, radius=0.01)

        if self.drawLines:
            # draw lines
            for a, b in zip(points, points[1:]):
                if b is not None:
                    d.addLine(a, b)

            # connect end points
            if points[-1] is not None:
                d.addLine(points[0], points[-1])

        self.annotationObj = updatePolyData(d.getPolyData(), 'annotation', parent=getDebugFolder())
        self.annotationObj.setProperty('Color', QtGui.QColor(0, 255, 0))
        self.annotationObj.actor.SetPickable(False)

    def tick(self):

        if not self.enabled:
            return

        if not om.findObjectByName('pointcloud snapshot'):
            self.annotationFunc = None
            self.finish()
            return

        pickedPointFields = pickPoint(self.lastMovePos, getSegmentationView(), obj='pointcloud snapshot')
        self.hoverPos = pickedPointFields.pickedPoint
        self.draw()


class LineDraw(TimerCallback):
    def __init__(self, view):
        TimerCallback.__init__(self)
        self.targetFps = 30
        self.enabled = False
        self.view = view
        self.renderer = view.renderer()
        self.line = vtk.vtkLeaderActor2D()
        self.line.SetArrowPlacementToNone()
        self.line.GetPositionCoordinate().SetCoordinateSystemToViewport()
        self.line.GetPosition2Coordinate().SetCoordinateSystemToViewport()
        self.line.GetProperty().SetLineWidth(4)
        self.line.SetPosition(0, 0)
        self.line.SetPosition2(0, 0)
        self.clear()

    def clear(self):
        self.p1 = None
        self.p2 = None
        self.annotationFunc = None
        self.lastMovePos = [0, 0]
        self.renderer.RemoveActor2D(self.line)

    def onMouseMove(self, displayPoint, modifiers=None):
        self.lastMovePos = displayPoint

    def onMousePress(self, displayPoint, modifiers=None):

        if self.p1 is None:
            self.p1 = list(self.lastMovePos)
            if self.p1 is not None:
                self.renderer.AddActor2D(self.line)
        else:
            self.p2 = self.lastMovePos
            self.finish()

    def finish(self):

        self.enabled = False
        self.renderer.RemoveActor2D(self.line)
        if self.annotationFunc is not None:
            self.annotationFunc(self.p1, self.p2)

    def handleRelease(self, displayPoint):
        pass

    def tick(self):

        if not self.enabled:
            return

        if self.p1:
            self.line.SetPosition(self.p1)
            self.line.SetPosition2(self.lastMovePos)
            self.view.render()


viewPickers = []


def addViewPicker(picker):
    global viewPickers
    viewPickers.append(picker)


def removeViewPicker(picker):
    global viewPickers
    viewPickers.remove(picker)


def distanceToLine(x0, x1, x2):
    numerator = np.sqrt(np.sum(np.cross((x0 - x1), (x0 - x2)) ** 2))
    denom = np.linalg.norm(x2 - x1)
    return numerator / denom


def labelDistanceToLine(polyData, linePoint1, linePoint2, resultArrayName='distance_to_line'):
    x0 = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    x1 = np.array(linePoint1)
    x2 = np.array(linePoint2)

    numerator = np.sqrt(np.sum(np.cross((x0 - x1), (x0 - x2)) ** 2, axis=1))
    denom = np.linalg.norm(x2 - x1)

    dists = numerator / denom

    polyData = shallowCopy(polyData)
    vtkNumpy.addNumpyToVtk(polyData, dists, resultArrayName)
    return polyData


def labelDistanceToPoint(polyData, point, resultArrayName='distance_to_point'):
    assert polyData.GetNumberOfPoints()
    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    points = points - point
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    polyData = shallowCopy(polyData)
    vtkNumpy.addNumpyToVtk(polyData, dists, resultArrayName)
    return polyData


def getPlaneEquationFromPolyData(polyData, expectedNormal):
    _, origin, normal = applyPlaneFit(polyData, expectedNormal=expectedNormal, returnOrigin=True)
    return origin, normal, np.hstack((normal, [np.dot(origin, normal)]))


def computeEdge(polyData, edgeAxis, perpAxis, binWidth=0.03):
    polyData = labelPointDistanceAlongAxis(polyData, edgeAxis, resultArrayName='dist_along_edge')
    polyData = labelPointDistanceAlongAxis(polyData, perpAxis, resultArrayName='dist_perp_to_edge')

    polyData, bins = binByScalar(polyData, 'dist_along_edge', binWidth)
    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    binLabels = vtkNumpy.getNumpyFromVtk(polyData, 'bin_labels')
    distToEdge = vtkNumpy.getNumpyFromVtk(polyData, 'dist_perp_to_edge')

    numberOfBins = len(bins) - 1
    edgePoints = []
    for i in xrange(numberOfBins):
        binPoints = points[binLabels == i]
        binDists = distToEdge[binLabels == i]
        if len(binDists):
            edgePoints.append(binPoints[binDists.argmax()])

    return np.array(edgePoints)


def computeCentroids(polyData, axis, binWidth=0.025):
    polyData = labelPointDistanceAlongAxis(polyData, axis, resultArrayName='dist_along_axis')

    polyData, bins = binByScalar(polyData, 'dist_along_axis', binWidth)
    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    binLabels = vtkNumpy.getNumpyFromVtk(polyData, 'bin_labels')

    numberOfBins = len(bins) - 1
    centroids = []
    for i in xrange(numberOfBins):
        binPoints = points[binLabels == i]

        if len(binPoints):
            centroids.append(np.average(binPoints, axis=0))

    return np.array(centroids)


def computePointCountsAlongAxis(polyData, axis, binWidth=0.025):
    polyData = labelPointDistanceAlongAxis(polyData, axis, resultArrayName='dist_along_axis')

    polyData, bins = binByScalar(polyData, 'dist_along_axis', binWidth)
    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    binLabels = vtkNumpy.getNumpyFromVtk(polyData, 'bin_labels')

    numberOfBins = len(bins) - 1
    binCount = []
    for i in xrange(numberOfBins):
        binPoints = points[binLabels == i]
        binCount.append(len(binPoints))

    return np.array(binCount)


def binByScalar(lidarData, scalarArrayName, binWidth, binLabelsArrayName='bin_labels'):
    '''
    Gets the array with name scalarArrayName from lidarData.
    Computes bins by dividing the scalar array into bins of size binWidth.
    Adds a new label array to the lidar points identifying which bin the point belongs to,
    where the first bin is labeled with 0.
    Returns the new, labeled lidar data and the bins.
    The bins are an array where each value represents a bin edge.
    '''

    scalars = vtkNumpy.getNumpyFromVtk(lidarData, scalarArrayName)
    bins = np.arange(scalars.min(), scalars.max() + binWidth, binWidth)
    binLabels = np.digitize(scalars, bins) - 1
    assert (len(binLabels) == len(scalars))
    newData = shallowCopy(lidarData)
    vtkNumpy.addNumpyToVtk(newData, binLabels, binLabelsArrayName)
    return newData, bins


def showObbs(polyData):
    labelsArrayName = 'cluster_labels'
    assert polyData.GetPointData().GetArray(labelsArrayName)

    f = vtk.vtkAnnotateOBBs()
    f.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, labelsArrayName)
    f.SetInput(polyData)
    f.Update()
    showPolyData(f.GetOutput(), 'bboxes')


def getOrientedBoundingBox(polyData):
    '''
    returns origin, edges, and outline wireframe
    '''
    nPoints = polyData.GetNumberOfPoints()
    assert nPoints
    polyData = shallowCopy(polyData)

    labelsArrayName = 'bbox_labels'
    labels = np.ones(nPoints)
    vtkNumpy.addNumpyToVtk(polyData, labels, labelsArrayName)

    f = vtk.vtkAnnotateOBBs()
    f.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, labelsArrayName)
    f.SetInput(polyData)
    f.Update()

    assert f.GetNumberOfBoundingBoxes() == 1

    origin = np.zeros(3)
    edges = [np.zeros(3) for i in xrange(3)]

    f.GetBoundingBoxOrigin(0, origin)
    for i in xrange(3):
        f.GetBoundingBoxEdge(0, i, edges[i])

    return origin, edges, shallowCopy(f.GetOutput())


def segmentBlockByAnnotation(blockDimensions, p1, p2, p3):
    segmentationObj = om.findObjectByName('pointcloud snapshot')
    segmentationObj.mapper.ScalarVisibilityOff()
    segmentationObj.setProperty('Point Size', 2)
    segmentationObj.setProperty('Alpha', 0.8)

    # constraint z to lie in plane
    # p1[2] = p2[2] = p3[2] = max(p1[2], p2[2], p3[2])

    zedge = p2 - p1
    zaxis = zedge / np.linalg.norm(zedge)

    # xwidth = distanceToLine(p3, p1, p2)

    # expected dimensions
    xwidth, ywidth = blockDimensions

    zwidth = np.linalg.norm(zedge)

    yaxis = np.cross(p2 - p1, p3 - p1)
    yaxis = yaxis / np.linalg.norm(yaxis)

    xaxis = np.cross(yaxis, zaxis)

    # reorient axes
    viewPlaneNormal = getSegmentationView().camera().GetViewPlaneNormal()
    if np.dot(yaxis, viewPlaneNormal) < 0:
        yaxis *= -1

    if np.dot(xaxis, p3 - p1) < 0:
        xaxis *= -1

    # make right handed
    zaxis = np.cross(xaxis, yaxis)

    origin = ((p1 + p2) / 2.0) + xaxis * xwidth / 2.0 + yaxis * ywidth / 2.0

    d = DebugData()
    d.addSphere(origin, radius=0.01)
    d.addLine(origin - xaxis * xwidth / 2.0, origin + xaxis * xwidth / 2.0)
    d.addLine(origin - yaxis * ywidth / 2.0, origin + yaxis * ywidth / 2.0)
    d.addLine(origin - zaxis * zwidth / 2.0, origin + zaxis * zwidth / 2.0)
    obj = updatePolyData(d.getPolyData(), 'block axes')
    obj.setProperty('Color', QtGui.QColor(255, 255, 0))
    obj.setProperty('Visible', False)
    om.findObjectByName('annotation').setProperty('Visible', False)

    cube = vtk.vtkCubeSource()
    cube.SetXLength(xwidth)
    cube.SetYLength(ywidth)
    cube.SetZLength(zwidth)
    cube.Update()
    cube = shallowCopy(cube.GetOutput())

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(origin)

    obj = updatePolyData(cube, 'block affordance', cls=BlockAffordanceItem, parent='affordances')
    obj.actor.SetUserTransform(t)

    obj.addToView(app.getDRCView())

    params = dict(origin=origin, xwidth=xwidth, ywidth=ywidth, zwidth=zwidth, xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)
    obj.setAffordanceParams(params)
    obj.updateParamsFromActorTransform()


####
# debrs task ground frame

def getBoardCorners(params):
    axes = [np.array(params[axis]) for axis in ['xaxis', 'yaxis', 'zaxis']]
    widths = [np.array(params[axis]) / 2.0 for axis in ['xwidth', 'ywidth', 'zwidth']]
    edges = [axes[i] * widths[i] for i in xrange(3)]
    origin = np.array(params['origin'])
    return [
        origin + edges[0] + edges[1] + edges[2],
        origin - edges[0] + edges[1] + edges[2],
        origin - edges[0] - edges[1] + edges[2],
        origin + edges[0] - edges[1] + edges[2],
        origin + edges[0] + edges[1] - edges[2],
        origin - edges[0] + edges[1] - edges[2],
        origin - edges[0] - edges[1] - edges[2],
        origin + edges[0] - edges[1] - edges[2],
    ]


def getPointDistances(target, points):
    return np.array([np.linalg.norm(target - p) for p in points])


def computeClosestCorner(aff, referenceFrame):
    corners = getBoardCorners(aff.params)
    dists = getPointDistances(np.array(referenceFrame.GetPosition()), corners)
    return corners[dists.argmin()]


def computeGroundFrame(aff, referenceFrame):
    refAxis = [0.0, -1.0, 0.0]
    referenceFrame.TransformVector(refAxis, refAxis)

    refAxis = np.array(refAxis)

    axes = [np.array(aff.params[axis]) for axis in ['xaxis', 'yaxis', 'zaxis']]
    axisProjections = np.array([np.abs(np.dot(axis, refAxis)) for axis in axes])
    boardAxis = axes[axisProjections.argmax()]
    if np.dot(boardAxis, refAxis) < 0:
        boardAxis = -boardAxis

    xaxis = boardAxis
    zaxis = np.array([0.0, 0.0, 1.0])
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, zaxis)
    closestCorner = computeClosestCorner(aff, referenceFrame)
    groundFrame = getTransformFromAxes(xaxis, yaxis, zaxis)
    groundFrame.PostMultiply()
    groundFrame.Translate(closestCorner[0], closestCorner[1], 0.0)
    return groundFrame


def computeCornerFrame(aff, referenceFrame):
    refAxis = [0.0, -1.0, 0.0]
    referenceFrame.TransformVector(refAxis, refAxis)

    refAxis = np.array(refAxis)

    axes = [np.array(aff.params[axis]) for axis in ['xaxis', 'yaxis', 'zaxis']]
    edgeLengths = [edgeLength for edgeLength in ['xwidth', 'ywidth', 'zwidth']]

    axisProjections = np.array([np.abs(np.dot(axis, refAxis)) for axis in axes])
    boardAxis = axes[axisProjections.argmax()]
    if np.dot(boardAxis, refAxis) < 0:
        boardAxis = -boardAxis

    longAxis = axes[np.argmax(edgeLengths)]

    xaxis = boardAxis
    yaxis = axes[2]
    zaxis = np.cross(xaxis, yaxis)

    closestCorner = computeClosestCorner(aff, referenceFrame)
    cornerFrame = getTransformFromAxes(xaxis, yaxis, zaxis)
    cornerFrame.PostMultiply()
    cornerFrame.Translate(closestCorner)
    return cornerFrame


def createBlockAffordance(origin, xaxis, yaxis, zaxis, xwidth, ywidth, zwidth, name, parent='affordances'):
    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(origin)

    obj = BoxAffordanceItem(name, view=app.getCurrentRenderView())
    obj.setProperty('Dimensions', [float(v) for v in [xwidth, ywidth, zwidth]])
    obj.actor.SetUserTransform(t)

    om.addToObjectModel(obj, parentObj=om.getOrCreateContainer(parent))
    frameObj = vis.showFrame(t, name + ' frame', scale=0.2, visible=False, parent=obj)

    obj.addToView(app.getDRCView())
    frameObj.addToView(app.getDRCView())

    affordanceManager.registerAffordance(obj)
    return obj


def segmentBlockByTopPlane(polyData, blockDimensions, expectedNormal, expectedXAxis, edgeSign=1,
                           name='block affordance'):
    polyData, planeOrigin, normal = applyPlaneFit(polyData, distanceThreshold=0.05, expectedNormal=expectedNormal,
                                                  returnOrigin=True)

    _, lineDirection, _ = applyLineFit(polyData)

    zaxis = lineDirection
    yaxis = normal
    xaxis = np.cross(yaxis, zaxis)

    if np.dot(xaxis, expectedXAxis) < 0:
        xaxis *= -1

    # make right handed
    zaxis = np.cross(xaxis, yaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis /= np.linalg.norm(zaxis)

    expectedXAxis = np.array(xaxis)

    edgePoints = computeEdge(polyData, zaxis, xaxis * edgeSign)
    edgePoints = vtkNumpy.getVtkPolyDataFromNumpyPoints(edgePoints)

    d = DebugData()
    obj = updatePolyData(edgePoints, 'edge points', parent=getDebugFolder(), visible=False)

    linePoint, lineDirection, _ = applyLineFit(edgePoints)
    zaxis = lineDirection
    xaxis = np.cross(yaxis, zaxis)

    if np.dot(xaxis, expectedXAxis) < 0:
        xaxis *= -1

    # make right handed
    zaxis = np.cross(xaxis, yaxis)
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis /= np.linalg.norm(zaxis)

    polyData = labelPointDistanceAlongAxis(polyData, xaxis, resultArrayName='dist_along_line')
    pts = vtkNumpy.getNumpyFromVtk(polyData, 'Points')

    dists = np.dot(pts - linePoint, zaxis)

    p1 = linePoint + zaxis * np.min(dists)
    p2 = linePoint + zaxis * np.max(dists)

    p1 = projectPointToPlane(p1, planeOrigin, normal)
    p2 = projectPointToPlane(p2, planeOrigin, normal)

    xwidth, ywidth = blockDimensions
    zwidth = np.linalg.norm(p2 - p1)

    origin = p1 - edgeSign * xaxis * xwidth / 2.0 - yaxis * ywidth / 2.0 + zaxis * zwidth / 2.0

    d = DebugData()

    # d.addSphere(linePoint, radius=0.02)
    # d.addLine(linePoint, linePoint + yaxis*ywidth)
    # d.addLine(linePoint, linePoint + xaxis*xwidth)
    # d.addLine(linePoint, linePoint + zaxis*zwidth)


    d.addSphere(p1, radius=0.01)
    d.addSphere(p2, radius=0.01)
    d.addLine(p1, p2)

    d.addSphere(origin, radius=0.01)
    # d.addLine(origin - xaxis*xwidth/2.0, origin + xaxis*xwidth/2.0)
    # d.addLine(origin - yaxis*ywidth/2.0, origin + yaxis*ywidth/2.0)
    # d.addLine(origin - zaxis*zwidth/2.0, origin + zaxis*zwidth/2.0)

    d.addLine(origin, origin + xaxis * xwidth / 2.0)
    d.addLine(origin, origin + yaxis * ywidth / 2.0)
    d.addLine(origin, origin + zaxis * zwidth / 2.0)

    # obj = updatePolyData(d.getPolyData(), 'block axes')
    # obj.setProperty('Color', QtGui.QColor(255, 255, 0))
    # obj.setProperty('Visible', False)

    obj = createBlockAffordance(origin, xaxis, yaxis, zaxis, xwidth, ywidth, zwidth, name)
    obj.setProperty('Color', [222 / 255.0, 184 / 255.0, 135 / 255.0])

    computeDebrisGraspSeed(obj)
    t = computeDebrisStanceFrame(obj)
    if t:
        showFrame(t, 'debris stance frame', parent=obj)

    return obj


def computeDebrisGraspSeed(aff):
    debrisReferenceFrame = om.findObjectByName('debris reference frame')
    if debrisReferenceFrame:
        debrisReferenceFrame = debrisReferenceFrame.transform
        affCornerFrame = computeCornerFrame(aff, debrisReferenceFrame)
        showFrame(affCornerFrame, 'board corner frame', parent=aff, visible=False)


def computeDebrisStanceFrame(aff):
    debrisReferenceFrame = om.findObjectByName('debris reference frame')
    debrisWallEdge = om.findObjectByName('debris plane edge')

    if debrisReferenceFrame and debrisWallEdge:

        debrisReferenceFrame = debrisReferenceFrame.transform

        affGroundFrame = computeGroundFrame(aff, debrisReferenceFrame)

        updateFrame(affGroundFrame, 'board ground frame', parent=getDebugFolder(), visible=False)

        affWallEdge = computeGroundFrame(aff, debrisReferenceFrame)

        framePos = np.array(affGroundFrame.GetPosition())
        p1, p2 = debrisWallEdge.points
        edgeAxis = p2 - p1
        edgeAxis /= np.linalg.norm(edgeAxis)
        projectedPos = p1 + edgeAxis * np.dot(framePos - p1, edgeAxis)

        affWallFrame = vtk.vtkTransform()
        affWallFrame.PostMultiply()

        useWallFrameForRotation = True

        if useWallFrameForRotation:
            affWallFrame.SetMatrix(debrisReferenceFrame.GetMatrix())
            affWallFrame.Translate(projectedPos - np.array(debrisReferenceFrame.GetPosition()))

            stanceWidth = 0.20
            stanceOffsetX = -0.35
            stanceOffsetY = 0.45
            stanceRotation = 0.0

        else:
            affWallFrame.SetMatrix(affGroundFrame.GetMatrix())
            affWallFrame.Translate(projectedPos - framePos)

            stanceWidth = 0.20
            stanceOffsetX = -0.35
            stanceOffsetY = -0.45
            stanceRotation = math.pi / 2.0

        stanceFrame, _, _ = getFootFramesFromReferenceFrame(affWallFrame, stanceWidth, math.degrees(stanceRotation),
                                                            [stanceOffsetX, stanceOffsetY, 0.0])

        return stanceFrame


def segmentBlockByPlanes(blockDimensions):
    planes = om.findObjectByName('selected planes').children()[:2]

    viewPlaneNormal = getSegmentationView().camera().GetViewPlaneNormal()
    origin1, normal1, plane1 = getPlaneEquationFromPolyData(planes[0].polyData, expectedNormal=viewPlaneNormal)
    origin2, normal2, plane2 = getPlaneEquationFromPolyData(planes[1].polyData, expectedNormal=viewPlaneNormal)

    xaxis = normal2
    yaxis = normal1
    zaxis = np.cross(xaxis, yaxis)
    xaxis = np.cross(yaxis, zaxis)

    pts1 = vtkNumpy.getNumpyFromVtk(planes[0].polyData, 'Points')
    pts2 = vtkNumpy.getNumpyFromVtk(planes[1].polyData, 'Points')

    linePoint = np.zeros(3)
    centroid2 = np.sum(pts2, axis=0) / len(pts2)
    vtk.vtkPlane.ProjectPoint(centroid2, origin1, normal1, linePoint)

    dists = np.dot(pts1 - linePoint, zaxis)

    p1 = linePoint + zaxis * np.min(dists)
    p2 = linePoint + zaxis * np.max(dists)

    xwidth, ywidth = blockDimensions
    zwidth = np.linalg.norm(p2 - p1)

    origin = p1 + xaxis * xwidth / 2.0 + yaxis * ywidth / 2.0 + zaxis * zwidth / 2.0

    d = DebugData()

    d.addSphere(linePoint, radius=0.02)
    d.addSphere(p1, radius=0.01)
    d.addSphere(p2, radius=0.01)
    d.addLine(p1, p2)

    d.addSphere(origin, radius=0.01)
    d.addLine(origin - xaxis * xwidth / 2.0, origin + xaxis * xwidth / 2.0)
    d.addLine(origin - yaxis * ywidth / 2.0, origin + yaxis * ywidth / 2.0)
    d.addLine(origin - zaxis * zwidth / 2.0, origin + zaxis * zwidth / 2.0)
    obj = updatePolyData(d.getPolyData(), 'block axes')
    obj.setProperty('Color', QtGui.QColor(255, 255, 0))
    obj.setProperty('Visible', False)

    cube = vtk.vtkCubeSource()
    cube.SetXLength(xwidth)
    cube.SetYLength(ywidth)
    cube.SetZLength(zwidth)
    cube.Update()
    cube = shallowCopy(cube.GetOutput())

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(origin)

    obj = updatePolyData(cube, 'block affordance', cls=BlockAffordanceItem, parent='affordances')
    obj.actor.SetUserTransform(t)
    obj.addToView(app.getDRCView())

    params = dict(origin=origin, xwidth=xwidth, ywidth=ywidth, zwidth=zwidth, xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)
    obj.setAffordanceParams(params)
    obj.updateParamsFromActorTransform()


def estimatePointerTip(robotModel, polyData):
    '''
    Given a robot model, uses forward kinematics to determine a pointer tip
    search region, then does a ransac line fit in the search region to find
    points on the pointer, and selects the maximum point along the line fit
    as the pointer tip.  Returns the pointer tip xyz on success and returns
    None on failure.
    '''
    palmFrame = robotModel.getLinkFrame('r_hand_force_torque')
    p1 = [0.0, 0.14, -0.06]
    p2 = [0.0, 0.24, -0.06]

    palmFrame.TransformPoint(p1, p1)
    palmFrame.TransformPoint(p2, p2)

    p1 = np.array(p1)
    p2 = np.array(p2)

    d = DebugData()
    d.addSphere(p1, radius=0.005)
    d.addSphere(p2, radius=0.005)
    d.addLine(p1, p2)
    vis.updatePolyData(d.getPolyData(), 'pointer line', color=[1, 0, 0], parent=getDebugFolder(), visible=False)

    polyData = cropToLineSegment(polyData, p1, p2)
    if not polyData.GetNumberOfPoints():
        # print 'pointer search region is empty'
        return None

    vis.updatePolyData(polyData, 'cropped to pointer line', parent=getDebugFolder(), visible=False)

    polyData = labelDistanceToLine(polyData, p1, p2)

    polyData = thresholdPoints(polyData, 'distance_to_line', [0.0, 0.07])

    if polyData.GetNumberOfPoints() < 2:
        # print 'pointer search region is empty'
        return None

    updatePolyData(polyData, 'distance to pointer line', colorByName='distance_to_line', parent=getDebugFolder(),
                   visible=False)

    ransacDistanceThreshold = 0.0075
    lineOrigin, lineDirection, polyData = applyLineFit(polyData, distanceThreshold=ransacDistanceThreshold)
    updatePolyData(polyData, 'line fit ransac', colorByName='ransac_labels', parent=getDebugFolder(), visible=False)

    lineDirection = np.array(lineDirection)
    lineDirection /= np.linalg.norm(lineDirection)

    if np.dot(lineDirection, (p2 - p1)) < 0:
        lineDirection *= -1

    polyData = thresholdPoints(polyData, 'ransac_labels', [1.0, 1.0])

    if polyData.GetNumberOfPoints() < 2:
        # print 'pointer ransac line fit failed to find inliers'
        return None

    obj = updatePolyData(polyData, 'line fit points', colorByName='dist_along_line', parent=getDebugFolder(),
                         visible=True)
    obj.setProperty('Point Size', 5)

    pts = vtkNumpy.getNumpyFromVtk(polyData, 'Points')

    dists = np.dot(pts - lineOrigin, lineDirection)

    p1 = lineOrigin + lineDirection * np.min(dists)
    p2 = lineOrigin + lineDirection * np.max(dists)

    d = DebugData()
    # d.addSphere(p1, radius=0.005)
    d.addSphere(p2, radius=0.005)
    d.addLine(p1, p2)
    vis.updatePolyData(d.getPolyData(), 'fit pointer line', color=[0, 1, 0], parent=getDebugFolder(), visible=True)

    return p2


def startBoundedPlaneSegmentation():
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.start()
    picker.annotationFunc = functools.partial(segmentBoundedPlaneByAnnotation)


def startValveSegmentationByWallPlane(expectedValveRadius):
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.start()
    picker.annotationFunc = functools.partial(segmentValveByWallPlane, expectedValveRadius)


def startValveSegmentationManual(expectedValveRadius):
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentValve, expectedValveRadius)


def startRefitWall():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.start()
    picker.annotationFunc = refitWall


def startWyeSegmentation():
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentWye)


def startDoorHandleSegmentation(otdfType):
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentDoorHandle, otdfType)


def startTrussSegmentation():
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = True
    picker.start()
    picker.annotationFunc = functools.partial(segmentTruss)


def startHoseNozzleSegmentation():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentHoseNozzle)


def storePoint(p):
    global _pickPoint
    _pickPoint = p


def getPickPoint():
    global _pickPoint
    return _pickPoint


def startPickPoint():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = storePoint


def startSelectToolTip():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = selectToolTip


def startDrillSegmentation():
    picker = PointPicker(numberOfPoints=3)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentDrill)


def startDrillAutoSegmentation():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentDrillAuto)


def startDrillButtonSegmentation():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentDrillButton)


def startPointerTipSegmentation():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentPointerTip)


def startDrillAutoSegmentationAlignedWithTable():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentDrillAlignedWithTable)


def startDrillBarrelSegmentation():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentDrillBarrel)


def startDrillWallSegmentation():
    picker = PointPicker(numberOfPoints=3)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = True
    picker.start()
    picker.annotationFunc = functools.partial(segmentDrillWall)


def startDrillWallSegmentationConstrained(rightAngleLocation):
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = False
    picker.start()
    picker.annotationFunc = functools.partial(segmentDrillWallConstrained, rightAngleLocation)


def startDrillInHandSegmentation():
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.drawLines = True
    picker.start()
    picker.annotationFunc = functools.partial(segmentDrillInHand)


def startSegmentDebrisWall():
    picker = PointPicker(numberOfPoints=1)
    addViewPicker(picker)
    picker.enabled = True
    picker.start()
    picker.annotationFunc = functools.partial(segmentDebrisWall)


def startSegmentDebrisWallManual():
    picker = PointPicker(numberOfPoints=2)
    addViewPicker(picker)
    picker.enabled = True
    picker.start()
    picker.annotationFunc = functools.partial(segmentDebrisWallManual)


def selectToolTip(point1):
    print point1


def segmentDebrisWallManual(point1, point2):
    p1, p2 = point1, point2

    d = DebugData()
    d.addSphere(p1, radius=0.01)
    d.addSphere(p2, radius=0.01)
    d.addLine(p1, p2)
    edgeObj = updatePolyData(d.getPolyData(), 'debris plane edge', visible=True)
    edgeObj.points = [p1, p2]

    xaxis = p2 - p1
    xaxis /= np.linalg.norm(xaxis)
    zaxis = np.array([0.0, 0.0, 1.0])
    yaxis = np.cross(zaxis, xaxis)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(p1)

    updateFrame(t, 'debris plane frame', parent=edgeObj, visible=False)

    refFrame = vtk.vtkTransform()
    refFrame.PostMultiply()
    refFrame.SetMatrix(t.GetMatrix())
    refFrame.Translate(-xaxis + yaxis + zaxis * 20.0)
    updateFrame(refFrame, 'debris reference frame', parent=edgeObj, visible=False)


def segmentDebrisWall(point1):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = shallowCopy(inputObj.polyData)

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())

    polyData, origin, normal = applyPlaneFit(polyData, distanceThreshold=0.02, expectedNormal=viewPlaneNormal,
                                             perpendicularAxis=viewPlaneNormal,
                                             searchOrigin=point1, searchRadius=0.25, angleEpsilon=0.7,
                                             returnOrigin=True)

    planePoints = thresholdPoints(polyData, 'dist_to_plane', [-0.02, 0.02])
    updatePolyData(planePoints, 'unbounded plane points', parent=getDebugFolder(), visible=False)

    planePoints = applyVoxelGrid(planePoints, leafSize=0.03)
    planePoints = labelOutliers(planePoints, searchRadius=0.06, neighborsInSearchRadius=10)

    updatePolyData(planePoints, 'voxel plane points', parent=getDebugFolder(), colorByName='is_outlier', visible=False)

    planePoints = thresholdPoints(planePoints, 'is_outlier', [0, 0])

    planePoints = labelDistanceToPoint(planePoints, point1)
    clusters = extractClusters(planePoints, clusterTolerance=0.10)
    clusters.sort(key=lambda x: vtkNumpy.getNumpyFromVtk(x, 'distance_to_point').min())

    planePoints = clusters[0]
    planeObj = updatePolyData(planePoints, 'debris plane points', parent=getDebugFolder(), visible=False)

    perpAxis = [0, 0, -1]
    perpAxis /= np.linalg.norm(perpAxis)
    edgeAxis = np.cross(normal, perpAxis)

    edgePoints = computeEdge(planePoints, edgeAxis, perpAxis)
    edgePoints = vtkNumpy.getVtkPolyDataFromNumpyPoints(edgePoints)
    updatePolyData(edgePoints, 'edge points', parent=getDebugFolder(), visible=False)

    linePoint, lineDirection, _ = applyLineFit(edgePoints)

    # binCounts = computePointCountsAlongAxis(planePoints, lineDirection)


    xaxis = lineDirection
    yaxis = normal

    zaxis = np.cross(xaxis, yaxis)

    if np.dot(zaxis, [0, 0, 1]) < 0:
        zaxis *= -1
        xaxis *= -1

    pts = vtkNumpy.getNumpyFromVtk(planePoints, 'Points')

    dists = np.dot(pts - linePoint, xaxis)

    p1 = linePoint + xaxis * np.min(dists)
    p2 = linePoint + xaxis * np.max(dists)

    p1 = projectPointToPlane(p1, origin, normal)
    p2 = projectPointToPlane(p2, origin, normal)

    d = DebugData()
    d.addSphere(p1, radius=0.01)
    d.addSphere(p2, radius=0.01)
    d.addLine(p1, p2)
    edgeObj = updatePolyData(d.getPolyData(), 'debris plane edge', parent=planeObj, visible=True)
    edgeObj.points = [p1, p2]

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(p1)

    updateFrame(t, 'debris plane frame', parent=planeObj, visible=False)

    refFrame = vtk.vtkTransform()
    refFrame.PostMultiply()
    refFrame.SetMatrix(t.GetMatrix())
    refFrame.Translate(-xaxis + yaxis + zaxis * 20.0)
    updateFrame(refFrame, 'debris reference frame', parent=planeObj, visible=False)


def segmentBoundedPlaneByAnnotation(point1, point2):
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = shallowCopy(inputObj.polyData)

    viewPlaneNormal = np.array(getSegmentationView().camera().GetViewPlaneNormal())

    polyData, origin, normal = applyPlaneFit(polyData, distanceThreshold=0.015, expectedNormal=viewPlaneNormal,
                                             perpendicularAxis=viewPlaneNormal,
                                             searchOrigin=point1, searchRadius=0.3, angleEpsilon=0.7, returnOrigin=True)

    planePoints = thresholdPoints(polyData, 'dist_to_plane', [-0.015, 0.015])
    updatePolyData(planePoints, 'unbounded plane points', parent=getDebugFolder(), visible=False)

    planePoints = applyVoxelGrid(planePoints, leafSize=0.03)
    planePoints = labelOutliers(planePoints, searchRadius=0.06, neighborsInSearchRadius=12)

    updatePolyData(planePoints, 'voxel plane points', parent=getDebugFolder(), colorByName='is_outlier', visible=False)

    planePoints = thresholdPoints(planePoints, 'is_outlier', [0, 0])

    planePoints = labelDistanceToPoint(planePoints, point1)
    clusters = extractClusters(planePoints, clusterTolerance=0.10)
    clusters.sort(key=lambda x: vtkNumpy.getNumpyFromVtk(x, 'distance_to_point').min())

    planePoints = clusters[0]
    updatePolyData(planePoints, 'plane points', parent=getDebugFolder(), visible=False)

    perpAxis = point2 - point1
    perpAxis /= np.linalg.norm(perpAxis)
    edgeAxis = np.cross(normal, perpAxis)

    edgePoints = computeEdge(planePoints, edgeAxis, perpAxis)
    edgePoints = vtkNumpy.getVtkPolyDataFromNumpyPoints(edgePoints)
    updatePolyData(edgePoints, 'edge points', parent=getDebugFolder(), visible=False)

    linePoint, lineDirection, _ = applyLineFit(edgePoints)

    zaxis = normal
    yaxis = lineDirection
    xaxis = np.cross(yaxis, zaxis)

    if np.dot(xaxis, perpAxis) < 0:
        xaxis *= -1

    # make right handed
    yaxis = np.cross(zaxis, xaxis)

    pts = vtkNumpy.getNumpyFromVtk(planePoints, 'Points')

    dists = np.dot(pts - linePoint, yaxis)

    p1 = linePoint + yaxis * np.min(dists)
    p2 = linePoint + yaxis * np.max(dists)

    p1 = projectPointToPlane(p1, origin, normal)
    p2 = projectPointToPlane(p2, origin, normal)

    d = DebugData()
    d.addSphere(p1, radius=0.01)
    d.addSphere(p2, radius=0.01)
    d.addLine(p1, p2)
    updatePolyData(d.getPolyData(), 'plane edge', parent=getDebugFolder(), visible=False)

    t = getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate((p1 + p2) / 2.0)

    updateFrame(t, 'plane edge frame', parent=getDebugFolder(), visible=False)


savedCameraParams = None


def perspective():
    global savedCameraParams
    if savedCameraParams is None:
        return

    aff = getDefaultAffordanceObject()
    if aff:
        aff.setProperty('Alpha', 1.0)

    obj = om.findObjectByName('pointcloud snapshot')
    if obj is not None:
        obj.actor.SetPickable(1)

    view = getSegmentationView()
    c = view.camera()
    c.ParallelProjectionOff()
    c.SetPosition(savedCameraParams['Position'])
    c.SetFocalPoint(savedCameraParams['FocalPoint'])
    c.SetViewUp(savedCameraParams['ViewUp'])
    view.setCameraManipulationStyle()
    view.render()


def saveCameraParams(overwrite=False):
    global savedCameraParams
    if overwrite or (savedCameraParams is None):
        view = getSegmentationView()
        c = view.camera()
        savedCameraParams = dict(Position=c.GetPosition(), FocalPoint=c.GetFocalPoint(), ViewUp=c.GetViewUp())


def getDefaultAffordanceObject():
    obj = om.getActiveObject()
    if isinstance(obj, AffordanceItem):
        return obj

    for obj in om.getObjects():
        if isinstance(obj, AffordanceItem):
            return obj


def orthoX():
    aff = getDefaultAffordanceObject()
    if not aff:
        return

    saveCameraParams()

    aff.updateParamsFromActorTransform()
    aff.setProperty('Alpha', 0.3)
    om.findObjectByName('pointcloud snapshot').actor.SetPickable(0)

    view = getSegmentationView()
    c = view.camera()
    c.ParallelProjectionOn()

    origin = aff.params['origin']
    viewDirection = aff.params['xaxis']
    viewUp = -aff.params['yaxis']
    viewDistance = aff.params['xwidth'] * 3
    scale = aff.params['zwidth']

    c.SetFocalPoint(origin)
    c.SetPosition(origin - viewDirection * viewDistance)
    c.SetViewUp(viewUp)
    c.SetParallelScale(scale)

    view.setActorManipulationStyle()
    view.render()


def orthoY():
    aff = getDefaultAffordanceObject()
    if not aff:
        return

    saveCameraParams()

    aff.updateParamsFromActorTransform()
    aff.setProperty('Alpha', 0.3)
    om.findObjectByName('pointcloud snapshot').actor.SetPickable(0)

    view = getSegmentationView()
    c = view.camera()
    c.ParallelProjectionOn()

    origin = aff.params['origin']
    viewDirection = aff.params['yaxis']
    viewUp = -aff.params['xaxis']
    viewDistance = aff.params['ywidth'] * 4
    scale = aff.params['zwidth']

    c.SetFocalPoint(origin)
    c.SetPosition(origin - viewDirection * viewDistance)
    c.SetViewUp(viewUp)
    c.SetParallelScale(scale)

    view.setActorManipulationStyle()
    view.render()


def orthoZ():
    aff = getDefaultAffordanceObject()
    if not aff:
        return

    saveCameraParams()

    aff.updateParamsFromActorTransform()
    aff.setProperty('Alpha', 0.3)
    om.findObjectByName('pointcloud snapshot').actor.SetPickable(0)

    view = getSegmentationView()
    c = view.camera()
    c.ParallelProjectionOn()

    origin = aff.params['origin']
    viewDirection = aff.params['zaxis']
    viewUp = -aff.params['yaxis']
    viewDistance = aff.params['zwidth']
    scale = aff.params['ywidth'] * 6

    c.SetFocalPoint(origin)
    c.SetPosition(origin - viewDirection * viewDistance)
    c.SetViewUp(viewUp)
    c.SetParallelScale(scale)

    view.setActorManipulationStyle()
    view.render()


def zoomToDisplayPoint(displayPoint, boundsRadius=0.5, view=None):
    pickedPointFields = pickPoint(displayPoint, getSegmentationView(), obj='pointcloud snapshot')
    pickedPoint = pickedPointFields.pickedPoint
    if pickedPoint is None:
        return

    view = view or app.getCurrentRenderView()

    worldPt1, worldPt2 = getRayFromDisplayPoint(getSegmentationView(), displayPoint)

    diagonal = np.array([boundsRadius, boundsRadius, boundsRadius])
    bounds = np.hstack([pickedPoint - diagonal, pickedPoint + diagonal])
    bounds = [bounds[0], bounds[3], bounds[1], bounds[4], bounds[2], bounds[5]]
    view.renderer().ResetCamera(bounds)
    view.camera().SetFocalPoint(pickedPoint)
    view.render()


def extractPointsAlongClickRay(position, ray, polyData=None, distanceToLineThreshold=0.025, nearestToCamera=False):
    # segmentationObj = om.findObjectByName('pointcloud snapshot')
    if polyData is None:
        polyData = getCurrentRevolutionData()

    if not polyData or not polyData.GetNumberOfPoints():
        return None

    polyData = labelDistanceToLine(polyData, position, position + ray)

    # extract points near line
    polyData = thresholdPoints(polyData, 'distance_to_line', [0.0, distanceToLineThreshold])
    if not polyData.GetNumberOfPoints():
        return None

    polyData = labelPointDistanceAlongAxis(polyData, ray, origin=position, resultArrayName='distance_along_line')
    polyData = thresholdPoints(polyData, 'distance_along_line', [0.20, 1e6])
    if not polyData.GetNumberOfPoints():
        return None

    updatePolyData(polyData, 'ray points', colorByName='distance_to_line', visible=False, parent=getDebugFolder())

    if nearestToCamera:
        dists = vtkNumpy.getNumpyFromVtk(polyData, 'distance_along_line')
    else:
        dists = vtkNumpy.getNumpyFromVtk(polyData, 'distance_to_line')

    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    intersectionPoint = points[dists.argmin()]

    d = DebugData()
    d.addSphere(intersectionPoint, radius=0.005)
    d.addLine(position, intersectionPoint)
    obj = updatePolyData(d.getPolyData(), 'intersecting ray', visible=False, color=[0, 1, 0], parent=getDebugFolder())
    obj.actor.GetProperty().SetLineWidth(2)

    d2 = DebugData()
    end_of_ray = position + 2 * ray
    d2.addLine(position, end_of_ray)
    obj2 = updatePolyData(d2.getPolyData(), 'camera ray', visible=False, color=[1, 0, 0], parent=getDebugFolder())
    obj2.actor.GetProperty().SetLineWidth(2)

    return intersectionPoint


def segmentDrillWallFromTag(position, ray):
    '''
    Fix the drill wall relative to a ray intersected with the wall
    Desc: given a position and a ray (typically derived from a camera pixel)
    Use that point to determine a position for the Drill Wall
    This function uses a hard coded offset between the position on the wall
    to produce the drill cutting origin
    '''

    # inputObj = om.findObjectByName('pointcloud snapshot')
    # polyData = shallowCopy(inputObj.polyData)
    polyData = getCurrentRevolutionData()

    if (polyData is None):  # no data yet
        print "no LIDAR data yet"
        return False

    point1 = extractPointsAlongClickRay(position, ray, polyData)

    # view direction is out:
    viewDirection = -1 * SegmentationContext.getGlobalInstance().getViewDirection()
    polyDataOut, origin, normal = applyPlaneFit(polyData, expectedNormal=viewDirection, searchOrigin=point1,
                                                searchRadius=0.3, angleEpsilon=0.3, returnOrigin=True)

    # project the lidar point onto the plane (older, variance is >1cm with robot 2m away)
    # intersection_point = projectPointToPlane(point1, origin, normal)
    # intersect the ray with the plane (variance was about 4mm with robot 2m away)
    intersection_point = intersectLineWithPlane(position, ray, origin, normal)

    # Define a frame:
    xaxis = -normal
    zaxis = [0, 0, 1]
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)
    t = transformUtils.getTransformFromAxes(xaxis, yaxis, zaxis)
    t.PostMultiply()
    t.Translate(intersection_point)

    t2 = transformUtils.copyFrame(t)
    t2.PreMultiply()
    t3 = transformUtils.frameFromPositionAndRPY([0, 0.6, -0.25], [0, 0, 0])
    t2.Concatenate(t3)

    rightAngleLocation = 'bottom left'
    createDrillWall(rightAngleLocation, t2)

    wall = om.findObjectByName('wall')
    vis.updateFrame(t, 'wall fit tag', parent=wall, visible=False, scale=0.2)

    d = DebugData()
    d.addSphere(intersection_point, radius=0.002)
    obj = updatePolyData(d.getPolyData(), 'intersection', parent=wall, visible=False, color=[0, 1, 0])  #
    obj.actor.GetProperty().SetLineWidth(1)
    return True


def segmentDrillWallFromWallCenter():
    '''
    Get the drill wall target as an offset from the center of
    the full wall
    '''

    # find the valve wall and its center
    inputObj = om.findObjectByName('pointcloud snapshot')
    polyData = inputObj.polyData

    # hardcoded position to target frame from center of wall
    # conincides with the distance from the april tag to this position
    wallFrame = transformUtils.copyFrame(findWallCenter(polyData))
    wallFrame.PreMultiply()
    t3 = transformUtils.frameFromPositionAndRPY([-0.07, -0.3276, 0], [180, -90, 0])
    wallFrame.Concatenate(t3)

    rightAngleLocation = 'bottom left'
    createDrillWall(rightAngleLocation, wallFrame)

    wall = om.findObjectByName('wall')
    vis.updateFrame(wallFrame, 'wall fit lidar', parent=wall, visible=False, scale=0.2)


def findFarRightCorner(polyData, linkFrame):
    '''
    Within a point cloud find the point to the far right from the link
    The input is the 4 corners of a minimum bounding box
    '''

    diagonalTransform = transformUtils.copyFrame(linkFrame)
    diagonalTransform.PreMultiply()
    diagonalTransform.Concatenate(transformUtils.frameFromPositionAndRPY([0, 0, 0], [0, 0, 45]))
    vis.updateFrame(diagonalTransform, 'diagonal frame', parent=getDebugFolder(), visible=False)

    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    viewOrigin = diagonalTransform.TransformPoint([0.0, 0.0, 0.0])
    viewX = diagonalTransform.TransformVector([1.0, 0.0, 0.0])
    viewY = diagonalTransform.TransformVector([0.0, 1.0, 0.0])
    viewZ = diagonalTransform.TransformVector([0.0, 0.0, 1.0])
    polyData = labelPointDistanceAlongAxis(polyData, viewY, origin=viewOrigin, resultArrayName='distance_along_foot_y')

    vis.updatePolyData(polyData, 'cornerPoints', parent='segmentation', visible=False)
    farRightIndex = vtkNumpy.getNumpyFromVtk(polyData, 'distance_along_foot_y').argmin()
    points = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    return points[farRightIndex, :]


def findMinimumBoundingRectangle(polyData, linkFrame):
    '''
    Find minimum bounding rectangle of a rectangular point cloud
    The input is assumed to be a rectangular point cloud e.g. the top of a block or table
    Returns transform of far right corner (pointing away from robot)
    '''

    # Originally From: https://github.com/dbworth/minimum-area-bounding-rectangle
    polyData = applyVoxelGrid(polyData, leafSize=0.02)

    def get2DAsPolyData(xy_points):
        '''
        Convert a 2D numpy array to a 3D polydata by appending z=0
        '''
        d = np.vstack((xy_points.T, np.zeros(xy_points.shape[0]))).T
        d2 = d.copy()
        return vtkNumpy.getVtkPolyDataFromNumpyPoints(d2)

    pts = vtkNumpy.getNumpyFromVtk(polyData, 'Points')
    xy_points = pts[:, [0, 1]]
    vis.updatePolyData(get2DAsPolyData(xy_points), 'xy_points', parent=getDebugFolder(), visible=False)
    hull_points = qhull_2d.qhull2D(xy_points)
    vis.updatePolyData(get2DAsPolyData(hull_points), 'hull_points', parent=getDebugFolder(), visible=False)
    # Reverse order of points, to match output from other qhull implementations
    hull_points = hull_points[::-1]
    # print 'Convex hull points: \n', hull_points, "\n"

    # Find minimum area bounding rectangle
    (rot_angle, rectArea, rectDepth, rectWidth, center_point, corner_points_ground) = min_bounding_rect.minBoundingRect(
        hull_points)
    vis.updatePolyData(get2DAsPolyData(corner_points_ground), 'corner_points_ground', parent=getDebugFolder(),
                       visible=False)

    polyDataCentroid = computeCentroid(polyData)
    cornerPoints = np.vstack((corner_points_ground.T, polyDataCentroid[2] * np.ones(corner_points_ground.shape[0]))).T
    cornerPolyData = vtkNumpy.getVtkPolyDataFromNumpyPoints(cornerPoints)

    # Create a frame at the far right point - which points away from the robot
    farRightCorner = findFarRightCorner(cornerPolyData, linkFrame)
    viewDirection = SegmentationContext.getGlobalInstance().getViewDirection()

    viewFrame = SegmentationContext.getGlobalInstance().getViewFrame()
    # vis.showFrame(viewFrame, "viewFrame")

    robotYaw = math.atan2(viewDirection[1], viewDirection[0]) * 180.0 / np.pi
    blockAngle = rot_angle * (180 / math.pi)
    # print "robotYaw   ", robotYaw
    # print "blockAngle ", blockAngle
    blockAngleAll = np.array([blockAngle, blockAngle + 90, blockAngle + 180, blockAngle + 270])

    values = blockAngleAll - robotYaw
    for i in range(0, 4):
        if (values[i] > 180):
            values[i] = values[i] - 360

    values = abs(values)
    min_idx = np.argmin(values)
    if ((min_idx == 1) or (min_idx == 3)):
        # print "flip rectDepth and rectWidth as angle is not away from robot"
        temp = rectWidth;
        rectWidth = rectDepth;
        rectDepth = temp

    # print "best angle", blockAngleAll[min_idx]
    rot_angle = blockAngleAll[min_idx] * math.pi / 180.0

    cornerTransform = transformUtils.frameFromPositionAndRPY(farRightCorner, [0, 0, np.rad2deg(rot_angle)])

    vis.showFrame(cornerTransform, "cornerTransform", parent=getDebugFolder(), visible=False)

    # print "Minimum area bounding box:"
    # print "Rotation angle:", rot_angle, "rad  (", rot_angle*(180/math.pi), "deg )"
    # print "rectDepth:", rectDepth, " rectWidth:", rectWidth, "  Area:", rectArea
    # print "Center point: \n", center_point # numpy array
    # print "Corner points: \n", cornerPoints, "\n"  # numpy array
    return cornerTransform, rectDepth, rectWidth, rectArea
