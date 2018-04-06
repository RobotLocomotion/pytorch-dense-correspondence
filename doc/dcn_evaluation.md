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
See the notebook `evaluation_quantitative.ipynb` for example usage. The main is `DenseCorrespondenceEvaluation`. In summary the steps are

1. Sample image pairs
2. For each image pair compute N matches (typically N = 100)
3. For each match find the best match in descriptor space and record statistics about this match.
4. Compile the results into a Pandas.DataFrame and save the results to a `data.csv` file.


### 2. DenseCorrespondenceEvaluationPlotter


Processes the `data.csv` to make plots and a `stats.yaml` with important summary statistics. See `evaluation_quantitative_plots.ipynb` for example usage.
