#!/usr/bin/env directorPython

# system
import os
import argparse

# director
import director.vtkAll as vtk
import director.vtkNumpy as vnp
import director.objectmodel as om
import director.visualization as vis

# pdc
from dense_correspondence_manipulation.mesh_processing.mesh_render import MeshRender




"""
Launches a mesh rendering director app.
This should be launched from the <path_to_log_folder>/processed location
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="(optional) dataset folder to load")
    # parser.add_argument('--current_dir', action='store_true', default=False, help="uses the current director as the data_foler")

    args = parser.parse_args()
    if args.data_dir:
        data_folder = args.data_dir
    else:
        print "running with data_dir set to current working directory . . . "
        data_folder = os.getcwd()


    obj_dict = MeshRender.from_data_folder(data_folder)
    app = obj_dict['app']

    globalsDict = globals()
    globalsDict.update(obj_dict)

    app.app.start(restoreWindow=True)