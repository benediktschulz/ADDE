# Aggregating Distribution forecasts from Deep Ensembles (ADDE)

This repository provides code and descriptions accompanying the paper

> Schulz, B. and Lerch, S. (2024). 
> Aggregating distribution forecasts from deep ensembles.
> Preprint available at TODO.

In particular, code for the implementation of the networks, aggregation methods, ensembling strategies, case study and evaluation in addition to input and evaluation data of the case study is available.

## Summary

We address the question of how to aggregate distribution forecasts based on ensembles of neural networks, referred to as `deep ensembles’. Using theoretical arguments and a comprehensive analysis on twelve benchmark data sets, we systematically compare probability- and quantile-based aggregation methods for three neural network-based approaches with different forecast distribution types as output. Our results show that combining forecast distributions from deep ensembles can substantially improve the predictive performance. We propose a general quantile aggregation framework for deep ensembles that allows for corrections of systematic deficiencies and performs well in a variety of settings, often superior compared to a linear combination of the forecast densities.

## Previous Version of Manuscript

The manuscript this repository accompanies is the revised version of a previous manuscript. The first version can be found at https://arxiv.org/abs/2204.02291 and is accompanied by the repository https://github.com/benediktschulz/agg_distr_deep_ens. While the first version of the manuscript covers only randomly initialized deep ensembles and one real-world data set, the revised version analyzes the aggregation methods for seven ensembling strategies and nine additional benchmark data sets. Further, the code of the previous version relied on R, whereas the revised version uses Python (but still uses an R-interface via r2py).

## Code

The environment can be generated via the `environment.yml`-file. The source code can be found in the `src`-directory. The files in the following table are differentiated in the categories data generation, forecast generation, evaluation and hyperparameter tuning.

| File | Description |
| ---- | ---- | 
| **_General_** | | 
| `BaseModel` | General neural network functions. |
| `BQNModels` | Neural network functions for BQN. |
| `DRNModels` | Neural network functions for DRN. |
| `HENModels` | Neural network functions for HEN. |
| `fn_basic` |  Auxiliary functions. |
| `fn_eval` |  Evaluation functions. |
| **_Data Generation_** | | 
| `run_data` | Run code for data generation. |
| `process_uci_dataset` | Preprocessing of real-world data sets. |
| `ss_0_data` | Generation of simulated data. |
| **_Forecast Generation_** | |
| `run_calc` | Run code for forecast generation. |
| `config`| Configuration file. Note that you have to adapt the paths for use. |
| `ss_1_ensemble` | Deep ensemble generation. |
| `ss_2_aggregation` | Deep ensemble aggregation. |
| **_Forecast Evaluation_** | |
| `run_eval` | Run code for forecast evaluation. |
| `config_eval` | Configuration file. Note that you have to adapt the paths for use. |
| `ss_3_scores` | Evaluation of deep ensemble and aggregated forecasts. |
| `ss_4_diversity` | Evaluation of deep ensemble and aggregated forecasts in terms of ensemble diversity. |
| `figures` | Generation of evaluation figures (not in manuscript). |
| `paper_figures_data` | Generation of data required for evaluation figures. |
| `paper_figures_data.pkl` | Data required for evaluation figures. |
| `paper_figures_in_depth_analysis` | Generation of evaluation figures in Section 4.1 of the manuscript. |
| `paper_figures_benchmark_study` | Generation of evaluation figures in Section 4.2 of the manuscript. |
| `r_figures` | Generation of evaluation figures in Sections 2 and 3 of the manuscript (R-file). |
|  **_Hyperparameter Tuning_** | |
| `run_hpar_calc` | Run code for hyperparameter tuning runs. |
| `run_hpar_eval` | Run code for evaluation of hyperparameter tuning runs. |
| `hpar_ls` | Configuration file for hyperparameter choices. |
| `config_hpar` | Configuration file for runs. Note that you have to adapt the paths for use. |
| `config_hpar_eval` | Configuration file for evaluation. Note that you have to adapt the paths for use. | 
| `hpar_network` | Calculation of hyperparameter tuning runs. |
| `hpar_eval` | Evaluation of hyperparameter tuning runs. |
| `hpar_summary` | Summary of hyperparameter tuning runs. |
| `hpar_tables` | Generated tables for hyperparameter tuning. |

Next to setting up the environment, one has to adapt the paths in the configuration files and use the correct structure for the data directory.

## Data

The entire data comprises around 1.2 TB and is therefore too large to be stored in this repository. However, this repository still contains subsets of the entire data from the study in the `data`-directory, which is structured as required by the code. Note that Scenario 2 from the manuscript is referred to as `scen_4` in the code and data for historical reasons.

**Included:** 
- The input data files the deep ensemble forecasts are generated with
- Configuration files of the chosen hyperparameter values
- Evaluation data

**Not included:**
- Deep ensemble forecasts
- Aggregated forecasts
- Hyperparameter tuning runs

Note that the input data had to be split for the Wind data set. Hence, the two data sets need to be appended (in the right order) to reproduce the original file.

## Evaluation Plots

The repository further includes various evaluation plots for the analysis of the different settings in the `plot`-directory. For each ensembling strategy, figures analogous to those used in the in-depth analysis in Section 4.1 are available for each data set (in the corresponding subdirectory). Further, the figures from Section 4 are given in the subdirectories `in_depth` and `benchmark_data`. 

## Note on the Generation of Forecasts for the Wind Data

The NN methods for the Wind data set in Section 4.2 are not identical to that in Schulz and Lerch (2022; Monthly Weather Review) and the previous manuscript, as we modified the variants such that they are applied analogously to the other data sets. The modifications are listed in the following:

- Here, we do not employ an embedding of the station IDs to reduce the complexity of the implementation.
- We excluded the station-wise bias and ensemble coverage as predictor variables, as they are based on the choice of the training period. For each partition, the variables would take different values for the same sample. Hence, we did not include them.
- In contrast to the benchmark data sets, the partitions of the wind gusts data are not randomly sampled but instead based on the calendar year of the samples. For each of the 5 partitions, we use one full calendar year as the test period. The remaining data is used for training, the last 20% of the (chronologically ordered) set are used for validation. This contrasts to the original paper, where the validation period was also chosen based on a calendar year.
- While BQN used the individual member forecasts from the numerical weather prediction model as predictors, DRN and HEN use the mean and standard deviation. Here, all three variants are based on the mean and standard deviation.
- While the HEN variant in the original paper uses a binning scheme tailored to the data set, we here employ the binning scheme described in Appendix S1, i.e., analogous to the other data sets.
- Analogous to the other data sets, we apply DRN with a normal distribution.
