# Mesh Descriptors

This file outlines how we annotate the mesh with the learned descriptors.

## Annotating Mesh

There are a few steps required to annotate the mesh with the learned descriptors.

1. Make sure you **disable anti-aliasing before proceeding**. 
    - This can be done by running `nvidia-settings` inside the docker image.
    - Then `X Screen 0 --> Antialiasing Settings`
    - Select `Override Application Settings` in the spinbox.

1. For each image in the dataset know the mapping from mesh cells to image pixels. Do this using the classes in `mesh_render.py`
. Specifically launch `mesh_render_app.py` and use the command `mesh_render.render_images()`. This creates files in
`processed/rendered_images/000000_mesh_cells.png`.

2. Compute the descriptor images, and match them up with the cell_ids computed in the previous step. The main
functionality for this is contained in `mesh_descriptors.py`. A convenience script allows you to run this.
    - Edit the script `compute_mesh_descriptors.py` and set appropriate variables for network, scene, config
    file etc.
    - For each image in the scene compute the mapping between cell_ids and their corresponding descriptors. The results are 
  stored in `processed/mesh_descriptors/<network_name>/000000_mesh_descriptors.npz`. Note that these are specific to what network you are computing descriptors for.
    - Compile all this information into the average descriptor for each cell. Save the results in 
    `processed/mesh_descriptors/<network_name>/mesh_descriptor_stats.npz`
    

## Visualizing the mesh
We can visualize the mesh using `mesh_procesing_app`. From `<path_to_log_folder>/processed` run

```
mesh_processing_app.py --colorize --network_name <network_name>
```
This will normalize the descriptors into the range `[0,1]` and display them.

## Compute Descriptor Images
You can extract the descriptor images for a given scene by using the script `compute_descriptor_images.py`. See
the python file for additional documentation.
    
