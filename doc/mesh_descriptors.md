# Mesh Descriptors

This file outlines how we annotate the mesh with the learned descriptors.

## Annotating Mesh

There are a few steps required to annotate the mesh with the learned descriptors. 

1. For each image in the dataset know the mapping from mesh cells to image pixels. Do this using the classes in `mesh_render.py`
. Specifically launch `mesH_render_app.py` and use the command `mesh_render.render_images()`. This creates files in
`processed/rendered_images/000000_mesh_cells.png`.

2. Compute the descriptor images, and match them up with the cell_ids computed in the previous step. Run
`compute_mesh_descriptors.py`. This does two things.
    - For each image in the scene compute the mapping between cell_ids and their corresponding descriptors. The results are 
  stored in `processed/mesh_descriptors/000000_mesh_descriptors.npz`.
    - Compile all this information into the average descriptor for each cell. Save the results in 
    `processed/mesh_descriptors/mesh_descriptor_stats.npz`
    

## Visualizing the mesh
We can visualize the mesh using `mesh_processing_app.py` by passing the `--colorize` flag. This will normalize the
descriptors into the range `[0,1]` and display them.
    
