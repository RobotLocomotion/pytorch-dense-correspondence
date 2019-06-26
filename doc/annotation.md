# Annotation


## Keypoint Annotation Tool

The main file is `keypoint_annotation_tool.py`. To launch the tool make a config like `config/annotation/shoe_keypoint_annotation.yaml` and modify the variable `KEYPOINT_ANNOTATION_CONFIG_FILE` in `kyepoint_annotation_app.py` to point to it.

```
use_pytorch_dense_correspondence
use_director
cd ~/code/modules/dense_correspondence_manipulation/annotation
./keypoint_annotation_app.py
```

Holding shift will show a red dot where the keypoint would be. Click to drop a keypoint there. Then double click that red sphere to move it around. You can either drag the frame or use the arrow keys (up/down = x, left/right = y, shift + up/down = z). 

Once you are done annotating a scene use `F8` to spawn the python terminal. Then use `k.save_annotations()` to save the annotations for that scene. Use `k.load_next_scene()` to load a new scene. If you mess up and want to start a scene over use `k.reload_scene()`.

### Output Format
It will save annotations in yaml files that look like 

```
- annotation_type: shoe_standard
  date_time: 02-11-2019_19-15-29
  keypoints:
    heel_top:
      position:
      - 0.6089581538173765
      - -0.022375569439354592
      - 0.08521565039787238
    toe:
      position:
      - 0.6695766143832599
      - 0.22942525767106536
      - 0.021247244843585644
  object_type: shoe
  scene_name: 2018-05-14-22-10-53
```
