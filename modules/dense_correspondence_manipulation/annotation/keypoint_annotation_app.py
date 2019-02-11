#!/usr/bin/env directorPython

# system
import os

# director
from director import mainwindowapp

# pdf
from dense_correspondence_manipulation.annotation.keypoint_annotation import KeypointAnnotation
import dense_correspondence_manipulation.utils.utils as pdc_utils


KEYPOINT_ANNOTATION_CONFIG_FILE = os.path.join(pdc_utils.getDenseCorrespondenceSourceDir(), "config/annotation/shoe_keypoint_annotation.yaml")
KEYPOINT_ANNOTATION_CONFIG = pdc_utils.getDictFromYamlFilename(KEYPOINT_ANNOTATION_CONFIG_FILE)
if __name__ == "__main__":

    app = mainwindowapp.construct()
    app.gridObj.setProperty('Visible', True)
    app.viewOptions.setProperty('Orientation widget', True)
    app.viewOptions.setProperty('View angle', 30)
    app.sceneBrowserDock.setVisible(True)
    app.propertiesDock.setVisible(False)
    app.mainWindow.setWindowTitle('Keypoint Annotation')
    app.mainWindow.show()
    app.mainWindow.resize(920, 600)
    app.mainWindow.move(0, 0)

    view = app.view

    globalsDict = globals()
    globalsDict['app'] = app
    globalsDict['view'] = view

    keypoint_annotation = KeypointAnnotation(view, config=KEYPOINT_ANNOTATION_CONFIG)
    globalsDict['keypoint_annotation'] = keypoint_annotation
    globalsDict['k'] = keypoint_annotation

    app.app.start(restoreWindow=True)





