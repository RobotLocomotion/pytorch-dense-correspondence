# Installing maskrcnn-benchmark
When you enter the container you need to build the maskrcnn-benchmark code. There is a bash function `build_maskrcnn_benchmark` that does this. Equivalently you can do

```
cd ~/code/external/,askrcnn-benchmark
python setup.py build develop
```

## Training a Model
See the notebook in `dense_correspondence/experiments/maskrcnn_shoes/maskrcnn_shoes.ipynb`
