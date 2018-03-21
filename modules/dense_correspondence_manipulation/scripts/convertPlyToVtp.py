'''
Usage:

directorPython convertPlyToVtp.py <path/to/data.ply>

This script will read the ply file and save a vtp file.
The output vtp file will be given the same name as the
input file with the ply extension replaced with vtp.

Note, the ply file needs to be in ascii format and not
binary.  The vtk ply reader seems to crash on binary
ply files.  You can use meshlab to open a binary ply
file and re-save it as an ascii ply file.
'''

import os
from director import ioUtils
from director import vtkNumpy as vnp


if __name__ == '__main__':

    filename = sys.argv[1]
    outputFilename = os.path.splitext(filename)[0] + '.vtp'

    polyData = ioUtils.readPolyData(filename)

    # TODO:
    # This should just be fixed in ioUtils.readPolyData, but for now
    # there is a workaround for an issue with the ply reader.
    # The output of the ply reader has points but not vertex cells,
    # so create a new polydata with vertex cells and copy the cells over.
    points = vnp.getNumpyFromVtk(polyData, 'Points')
    newPolyData = vnp.numpyToPolyData(points, createVertexCells=True)
    polyData.SetVerts(newPolyData.GetVerts())

    print 'writing:', outputFilename
    ioUtils.writePolyData(polyData, outputFilename)
