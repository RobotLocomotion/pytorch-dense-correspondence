# Coordinate Conventions in pytorch-dense-correspondence

Keeping coordinate systems consistent is not a spontaneous event.

There are a few reasons this is not always straightforward:

- Typical computer vision "x,y,z" convention in 3D space is "right-down-forward"
- Associated with this right-down-forward convention, it's easiest to think about pixel positions 
as being associated with this right, down convention.  This gives us the "(u, v)" convention for defining
pixel coordinates.
- On the other hand, images that are arrays/tensors in numpy or torch are accessed as row, column, in that order.
That isn't consisent with the (u, v) convention
- Also on the other hand, typical robot convention is to think of "x" as being forward, away from the robot, and "y" to the left.
This gives "forward-left-up" for "x,y,z"

## Conventions

In this repo we would like to follow this simple convention for cartesian 3D frames:

- <b>All cartesian frames are specified in right-down-forward coordinates</b>.

But we currently have a more hybrid / mixed approach with accessing pixel data:

- <b>When doing 3D vision, we use the OpenCV "(u,v)" convention for pixel coordinates.  This aligns with our
cartesian coordinate system as follows in this diagram:

<p align="center">
  <img src="./OpenCVcoordinates.png" width="450"/>
</p>

- <b>All data stored as numpy arrays or torch tensors should be stored in row, column format</b>.  This means
that in order to access the correct pixel in this data, if using (u,v) coordinates, you should access the data
as:

```
pixel_at_u_v = data_tensor[v,u]
```

