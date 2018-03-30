# Change Detection Pipeline
We have code that takes a scene reconstructed with ElasticFusion and produces masks for the object sitting on the table.

## Generating Masks
Use the script `run_change_detection.py`. Make sure to source the correct environment.

```
use_pytorch_dense_correspondence 
use_director
```

```
run_change_detection_pipeline.py --data_dir <path_to_elastic_fusion_reconstruction_folder>
```

Another way to do the same is to navigate to the appropriate data folder and use the `--current_dir` flag.
```
run_change_detection_pipeline.py --current_dir
```

```
cd <path_to_elastic_fusion_reconstruction_folder>
run_change_detection_pipeline.py --current_dir
```

## Automatically run change detection on all subfolders ("batch" change detection)

```
cd data_volume
use_pytorch_dense_correspondence 
use_director
batch_run_change_detection_pipeline.py
```
