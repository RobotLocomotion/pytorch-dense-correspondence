#!/usr/bin/python

import numpy as np
from skimage import measure
import time
from plyfile import PlyData, PlyElement

def main():
    filename = "/home/manuelli/code/data_volume/fusion/tsdf-2mm.bin"
    fin = open(filename, "rb")

    import array
    tsdfHeader = array.array("f")  # L is the typecode for uint32
    tsdfHeader.fromfile(fin, 8)
    print tsdfHeader
    print type(tsdfHeader)

    voxelGridDim = tsdfHeader[0:3]
    voxelGridOrigin = tsdfHeader[3:6]
    voxelSize = tsdfHeader[6]
    truncMargin = tsdfHeader[7]


    headerSize = 8
    tsdf = np.fromfile(filename, np.float32)
    tsdf = tsdf[headerSize:]
    print "tsdf.shape:", tsdf.shape


    voxelGridDim = np.asarray(voxelGridDim, dtype=np.int)
    print "voxelGridDim: ", voxelGridDim
    print "voxeGridOrigin: ", voxelGridOrigin

    tsdf = np.reshape(tsdf, voxelGridDim)
    print "tsdf.shape:", tsdf.shape

    verts, faces, normals, values = measure.marching_cubes_lewiner(tsdf, spacing=[voxelSize]*3)

    print "type(verts): ", type(verts)
    print "verts.shape: ", verts.shape

    print "np.max(verts[:,0]): ", np.max(verts[:,0])
    print "np.min(verts[:,0]): ", np.min(verts[:, 0])


    print "verts[0,:] = ", verts[0,:]

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points = verts
    # mesh_points[:,0] = voxelGridOrigin[0] + verts[:,2]
    # mesh_points[:,1] = voxelGridOrigin[1] + verts[:,1]
    # mesh_points[:,2] = voxelGridOrigin[2] + verts[:,0]



    # try writing to the ply file
    print "converting numpy arrays to format for ply file"
    ply_conversion_start_time = time.time()

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[('x', 'f4'), ('y', 'f4'),
                                                ('z', 'f4')])
    faces_tuple = np.zeros((num_faces,), dtype=[('vertex_indices', 'i4', (3,))])

    for i in xrange(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    for i in xrange(0, num_faces):
        faces_tuple[i] = faces[i, :].tolist()



    # save it out
    # try to save it
    el_verts = PlyElement.describe(verts_tuple, 'vertex')
    el_faces = PlyElement.describe(faces_tuple, 'face')

    ply_data = PlyData([el_verts, el_faces])
    ply = ply_data.write('test_reconstruction.ply')

    print "converting to ply format and writing to file took", time.time() - ply_conversion_start_time




if __name__ == "__main__":
    main()