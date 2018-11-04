#!/usr/bin/env directorPython

import os
import argparse

# pdc
import dense_correspondence_manipulation.change_detection.mesh_processing as mesh_processing
import dense_correspondence_manipulation.pose_estimation.utils as pose_utils
import director.vtkAll as vtk
import director.vtkNumpy as vnp
import director.objectmodel as om
import director.visualization as vis

from dense_correspondence_manipulation.mesh_processing.mesh_render import DescriptorMeshColor
from dense_correspondence_manipulation.pose_estimation.descriptor_pose_estimation import DescriptorPoseEstimator
from dense_correspondence.dataset.scene_structure import SceneStructure


"""
Launches a mesh processing director app.
This should be launched from the <path_to_log_folder>/processed location
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="(optional) dataset folder to load")

    parser.add_argument("--colorize", action='store_true', default=False, help="(optional) colorize mesh with descriptors")

    parser.add_argument("--network_name", type=str, help="(optional) which network to use when colorizing the mesh, ensure that the descriptor dimension is 3")


    args = parser.parse_args()
    if args.data_dir:
        data_folder = args.data_dir
    else:
        print "running with data_dir set to current working directory . . . "
        data_folder = os.getcwd()


    globalsDict = mesh_processing.main(globals(), data_folder)
    app = globalsDict['app']
    reconstruction = globalsDict['reconstruction']

    scene_structure = SceneStructure(data_folder)

    debug = True
    if debug:
        poly_data = reconstruction.poly_data
        globalsDict['p'] = reconstruction.poly_data
        globalsDict['t'] = poly_data.GetCell(0)

        poly_data_copy = vtk.vtkPolyData()
        poly_data_copy.CopyStructure(poly_data)


        dpe = DescriptorPoseEstimator(scene_structure.mesh_descriptor_statistics_filename(args.network_name))
        dpe.poly_data = poly_data_copy
        dpe.view = globalsDict['view']
        globalsDict['dpe'] = dpe
        dpe.initialize_debug()



    if args.colorize:
        if not args.network_name:
            raise ValueError("you must specify the `network_name` arg if you use"
                             "the `colorize` flag")
        network_name = args.network_name

        dmc = DescriptorMeshColor(reconstruction.vis_obj)
        globalsDict['dmc'] = dmc
        descriptor_stats_file = scene_structure.mesh_descriptor_statistics_filename(network_name)
        dmc.color_mesh_using_descriptors(descriptor_stats_file)

    app.app.start(restoreWindow=True)