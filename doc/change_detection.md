# Change Detection Pipeline
We have code that takes a scene reconstructed with ElasticFusion and produces masks for the object sitting on the table.

## Generating Masks
Use the script `run_change_detection.py`. Make sure to source the correct environment.

```
use_pytorch_dense_correspondence 
use_director
```

```
directorPython run_change_detection.py --data_dir <path_to_elastic_fusion_reconstruction_folder>
```
