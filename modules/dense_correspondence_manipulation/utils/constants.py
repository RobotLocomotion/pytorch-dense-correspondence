from director import transformUtils

CROP_BOX_DATA = dict()
CROP_BOX_DATA['dimensions'] = [0.5, 0.7, 0.4]
CROP_BOX_DATA['transform'] = transformUtils.transformFromPose([0.66757267, 0, 0.2], [1., 0., 0., 0.])


cameraPoseTest = ([ 7.53674834e-01, -8.55423154e-19,  6.92103873e-01], [-0.34869616,  0.6090305 ,  0.62659914, -0.33891938])

CAMERA_TO_WORLD = transformUtils.transformFromPose(cameraPoseTest[0], cameraPoseTest[1])
DEPTH_IM_RESCALE = 4000.0