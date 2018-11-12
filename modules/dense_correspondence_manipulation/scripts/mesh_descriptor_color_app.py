#!/usr/bin/env directorPython

# system
import os
import argparse
import numpy as np

# director
import director.vtkAll as vtk
import director.vtkNumpy as vnp
import director.objectmodel as om
import director.visualization as vis

# pdc
from dense_correspondence_manipulation.mesh_processing.mesh_render import MeshRender
from dense_correspondence_manipulation.mesh_processing.mesh_render import MeshColorizer
from dense_correspondence_manipulation.mesh_processing.mesh_render import DescriptorMeshColor




"""
Launches a mesh rendering director app.
This should be launched from the <path_to_log_folder>/processed location
"""
if __name__ == "__main__":

    print "\n--------------------------------\n"
    print "Have you disabled anti-aliasing in nvidia-settings? If not this won't work correctly"
    print "\n--------------------------------\n"

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
    mesh_render = obj_dict['mesh_render']
    descriptor_mesh_color = DescriptorMeshColor(mesh_render.poly_data_item, mesh_render._data_folder)
    descriptor_mesh_color.color_mesh_using_descriptors()
    obj_dict['dmc'] = descriptor_mesh_color
    app = obj_dict['app']


    globalsDict = globals()
    globalsDict.update(obj_dict)

    app.app.start(restoreWindow=True)