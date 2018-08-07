#!/usr/bin/env directorPython

import os
import argparse

# pdc
import dense_correspondence_manipulation.change_detection.mesh_processing as mesh_processing
import director.vtkAll as vtk
import director.vtkNumpy as vnp
import director.objectmodel as om
import director.visualization as vis

from dense_correspondence_manipulation.mesh_processing.mesh_render import DescriptorMeshColor


"""
Launches a mesh processing director app.
This should be launched from the <path_to_log_folder>/processed location
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="(optional) dataset folder to load")

    parser.add_argument("--colorize", action='store_true', default=False, help="(optional) colorize mesh with descriptors")


    args = parser.parse_args()
    if args.data_dir:
        data_folder = args.data_dir
    else:
        print "running with data_dir set to current working directory . . . "
        data_folder = os.getcwd()


    globalsDict = mesh_processing.main(globals(), data_folder)
    app = globalsDict['app']
    reconstruction = globalsDict['reconstruction']

    if args.colorize:
        dmc = DescriptorMeshColor(reconstruction.vis_obj, data_folder)
        globalsDict['dmc'] = dmc
        dmc.color_mesh_using_descriptors()

    app.app.start(restoreWindow=True)