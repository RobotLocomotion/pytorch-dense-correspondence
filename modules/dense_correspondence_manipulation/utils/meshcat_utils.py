import numpy as np

# meshcat
import meshcat
import meshcat.geometry as g
from meshcat.geometry import Geometry
import meshcat.transformations as tf


# pdc
import dense_correspondence_manipulation.utils.utils as pdc_utils



def visualize_pointcloud(vis, # meshcat.Visualize
                         name, # name of the object in Tree
                         depth, # depth image (in meters) #[H, W]
                         K, # camera matrix
                         rgb=None, # optional RGB image,
                         T_world_camera=None, # optional T_world_camera
                         ):
    # visualize in 3D
    out = pdc_utils.project_image_to_pointcloud(depth_image_meters=depth, K=K, rgb=rgb)

    geom = g.Points(
        g.PointsGeometry(out['pts'].transpose(), color=out['color'].transpose() / 255.0),
        g.PointsMaterial(size=0.001)
    )
    vis[name].set_object(geom)

    # set the transform if it is not None
    if T_world_camera is not None:
        vis[name].set_transform(T_world_camera)


def visualize_points(vis,
                     name,
                     pts,  # [N,3]
                     color=None,  # [3,] array of float in [0,1]
                     size=0.001,  # size of the points
                     T=None,  # T_world_pts transform
                     ):
    if color is not None:
        N, _ = pts.shape
        color = 1.0 * np.ones([N, 3]) * np.array(color)
        color = color.transpose()

    geom = g.Points(
        g.PointsGeometry(pts.transpose(), color=color),
        g.PointsMaterial(size=size)
    )

    vis[name].set_object(geom)

    if T is not None:
        vis[name].set_transform(T)

