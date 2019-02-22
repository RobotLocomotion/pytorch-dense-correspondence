#!/usr/bin/env directorPython

# system
import os

# director
from director import mainwindowapp

# pdf
from dense_correspondence_manipulation.annotation.keypoint_annotation import KeypointAnnotation, KeypointAnnotationVisualizer
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


    # keypoint_annotation.setup() # standard

    mesh_filepath = os.path.join(pdc_utils.get_data_dir(), "templates/mugs/mug.stl")
    keypoint_annotation.setup_for_synthetic_mesh_annotation(mesh_filepath)

    globalsDict['keypoint_annotation'] = keypoint_annotation
    globalsDict['k'] = keypoint_annotation

    # filename = "scene_2018-05-15-00-40-41_date_2019-02-14-02-16-38.yaml"
    filename = "scene_2018-05-09-17-36-30_date_2019-02-14-01-20-15.yaml"

    annotation_data_file = os.path.join(pdc_utils.get_data_dir(), "annotations/keypoints/mug_3_keypoint", filename)
    annotation_data = pdc_utils.getDictFromYamlFilename(annotation_data_file)[0]
    kv = KeypointAnnotationVisualizer(view)
    kv.visualize_annotation(annotation_data)


    app.app.start(restoreWindow=True)





