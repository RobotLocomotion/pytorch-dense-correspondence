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
from dense_correspodence_manipulation.poser.poser_client import PoserClient
from dense_correspondence.dataset.scene_structure import SceneStructure


"""
Launches a poser client app
"""
if __name__ == "__main__":

    globalsDict = mesh_processing.main(globals(), data_folder)
    app = globalsDict['app']
    poser = PoserClient()

    app.app.start(restoreWindow=True)