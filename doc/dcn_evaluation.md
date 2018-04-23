# Dense Correspondence Network Evaluation

## Qualitative Evaluation
 See the notebook `evaluation_plots_example.ipynb` for an example of how to use the `DenseCorrespondenceEvaluation` tool.
 The tool allows for loading in a network and dataset and running images through the DCN (dense correspondence network). The function 
 `evaluate_network_qualitative` produces plots of the dense descriptors and is a good starting point for exploring the 
 functionality of `DenseCorrespondenceEvaluation`.


## Quantitative Evaluation
There are two steps in the quantitative analysis. 

1. Compute statistics for a given (network, dataset) pair. Save the results as a csv file.
2. Analyze the compiled statistics to make plots and extract useful information.

### 1. DenseCorrespondenceEvaluation

##### Within Scene

See the notebook `evaluation_quantitative.ipynb` for example usage. The main class is `DenseCorrespondenceEvaluation`. In summary the steps are

1. Sample image pairs
2. For each image pair compute N matches (typically N = 100)
3. For each match find the best match in descriptor space and record statistics about this match.
4. Compile the results into a Pandas.DataFrame and save the results to a `data.csv` file.

##### Across Scene

In order to evaluate our performance across scenes we need pixelwise matches that go across scenes. This is not something that our data collection pipeline can do automatically, so there is some human labeling involved here. Use the [cross scene match annotation tool](cross_scene_annotation_tool.md) to generate some labeled data. Then see the notebook `evaluation_quantitative_cross_scene.ipynb` for how to use the cross scene evaluation functionality. The main function in `DenseCorrespondenceEvaluation` is `evaluate_single_network_cross_scene`. The main steps in that function call are exactly the same as the **within scene** evaluation above, except that the matches come from manual annotations rather than our camera pose tracking.

### 2. DenseCorrespondenceEvaluationPlotter


Processes the `data.csv` to make plots and a `stats.yaml` with important summary statistics. See `evaluation_quantitative_plots.ipynb` for example usage. The main plots that we use are all plots of cdf's (cumulative distribution functions) of distributions of interest. The statistics plotted are

1. Pixel match error
2. 3D match error
3. Descriptor match error (between ground truth matches)
4. Fraction false positives
5. Average l2 pixel distance for false postiives
