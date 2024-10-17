## Hyperparameter tuning: Scores of different hyperparameter combinations

import os
import pickle
import json
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time_ns

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def main():
    ### Get Config ###
    with open("src/config_hpar_eval.json", "rb") as f:
        CONFIG = json.load(f)

    ### Settings ###
    # Method
    ens_method = CONFIG["ENS_METHOD"]

    # Path of deep ensemble forecasts
    data_dir = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
    )

    # Path of results
    plot_dir = os.path.join(
        CONFIG["PATHS"]["GIT_DIR"],
        CONFIG["PATHS"]["PLOT_DIR"],
        ens_method,
        "hpar",
    )

    ### Initialize ###
    # Models considered
    dataset_ls = CONFIG["DATASET"]

    # Network types
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]

    # To evaluate
    sr_eval = ["crps", "logs", "lgt", "cov", "mae", "me", "rmse"]

    # Hyperparameters
    hpar_type0 = ["n_batch", "actv", "lr_adam", "layers"]

    # Additional hyperparameters dependent on method
    if ens_method == "mc_dropout":
        hpar_type0 = hpar_type0 + ["p_dropout"]
    if ens_method == "bayesian":
        hpar_type0 = hpar_type0 + ["prior"]

    # For-Loop over scenarios and simulations
    for dataset in dataset_ls:
        # For-Loop over network types
        for temp_nn in nn_vec:
            # Network-dependent hyperparameters
            if temp_nn == "drn":
                    hpar_type = hpar_type0
            elif temp_nn == "bqn":
                    hpar_type = hpar_type0 + ["p_degree"]
            elif temp_nn == "hen":
                    hpar_type = hpar_type0 + ["N_bins"]
            
            # Take time
            start_time = time_ns()

            # Get hyperparameter data
            filename = os.path.join(
                    data_dir,
                    dataset,
                    ens_method,
                    f"eval_{temp_nn}_hpar_{dataset}_{ens_method}.pkl",
                )
            
            # Load data
            with open(filename, "rb") as f:
                df = pickle.load( f )
            
            # Read out number of simulations
            n_sim = df["n_sim"].max() + 1
            
            # Save hyperparameter counter as integer
            df["i_hpar"] = df["i_hpar"].astype(int)

            # Define custom aggregation function
            def mean_with_nan(series):
                return series.mean(skipna=False)
            
            # Choose aggregator for different columns
            agg_functions = {}
            for col in sr_eval:
                agg_functions[col] = mean_with_nan
            for col in hpar_type:
                agg_functions[col] = "first"

            # Aggregate simulations
            df_mean = df.groupby("i_hpar").agg(agg_functions)
            
            # Sort aggregated runs by CRPS
            df_sorted = df_mean.sort_values(by='crps')

            #### Print out as table ####
            # Name of HTML
            filepdf = os.path.join(
                plot_dir,
                f"hpar_{dataset}_{temp_nn}_table.html",
            )

            # Save as HTML
            df_sorted.to_html(filepdf, index=True)

            #### Make boxplots ####
            # Start plotting
            fig, axes = plt.subplots(nrows=len(hpar_type), ncols=3, figsize=(15,20))
            axes = axes.flatten()

            # Index
            i_axes = 0
            
            # For-Loop over hyperparameters
            for temp_hpar in hpar_type:
                # For-Loop over measures of interest
                for temp_sr in ["crps", "lgt", "me"]:
                    # Set position
                    ax = axes[i_axes]

                    # Make a boxplot over different values of hyperparameter
                    sns.boxplot(ax=ax, x=temp_hpar, y=temp_sr, data=df_mean)
                    ax.set_xlabel(temp_hpar)
                    ax.set_ylabel(temp_sr)
                    
                    # Add a dashed horizontal line for a optimal value
                    if temp_sr == "me":
                        ax.axhline(0, color='black', linestyle='--')

                    # Increase index
                    i_axes += 1
                    
            # Name of PDF
            filepdf = os.path.join(
                plot_dir,
                f"hpar_{dataset}_{temp_nn}_boxlplot_outliers.pdf",
            )

            # Save as file
            plt.savefig(filepdf)
            plt.close()

            #### Make boxplot without outliers ####
            # Start plotting
            fig, axes = plt.subplots(nrows=len(hpar_type), ncols=3, figsize=(15,20))
            axes = axes.flatten()

            # Index
            i_axes = 0
            
            # For-Loop over hyperparameters
            for temp_hpar in hpar_type:
                # For-Loop over measures of interest
                for temp_sr in ["crps", "lgt", "me"]:
                    # Set position
                    ax = axes[i_axes]

                    # Make a boxplot over different values of hyperparameter
                    sns.boxplot(ax=ax, x=temp_hpar, y=temp_sr, data=df_mean, showfliers = False)
                    ax.set_xlabel(temp_hpar)
                    ax.set_ylabel(temp_sr)
                    
                    # Add a dashed horizontal line for a optimal value
                    if temp_sr == "me":
                        ax.axhline(0, color='black', linestyle='--')

                    # Increase index
                    i_axes += 1
                    
            # Name of PDF
            filepdf = os.path.join(
                plot_dir,
                f"hpar_{dataset}_{temp_nn}_boxlplot_nooutliers.pdf",
            )

            # Save as file
            plt.savefig(filepdf)
            plt.close()

            #### PIT histograms of best performing models ####
            # Number of models to plot
            n_best = 10

            # Number of bins in PIT histogram
            n_bins = 10

            # Get number of rows and columns
            n_rows = int(n_best/2)
            n_cols = int(min(4, 2*n_sim))

            # Start plotting (Manually adjust dependent on n_best)
            fig, axes = plt.subplots(nrows=n_rows, 
                                     ncols=n_cols, 
                                     figsize=(4*n_cols,4*n_rows))
            axes = axes.flatten()
            
            # Axis index
            i_axes = 0

            # For-Loop over best models
            for i_best in range(n_best):
                # Get hyperparameter index
                i_hpar = int(df_sorted.index[i_best])

                # For-Loop over simulated runs
                for i_sim in range(n_sim):
                    # Get name of file
                    filepit = os.path.join(
                            data_dir,
                            dataset,
                            ens_method,
                            CONFIG["PATHS"]["HPAR_F"],
                            f"{temp_nn}_hpar_{i_hpar}_sim_{i_sim}.pkl",
                        )
                    
                    # Load data
                    with open(filepit, "rb") as f:
                        pred_nn,_,_,_ = pickle.load( f )
                    
                    # Make a histogram
                    ax = axes[i_axes]
                    ax.hist(pred_nn["scores"]["pit"], bins = n_bins, density = True, color = "lightgrey", alpha = 0.7, edgecolor = "black")
                    ax.set_ylim(0, 2)
                    
                    # Add a dashed horizontal line for a flat histogram
                    ax.axhline(1, color='black', linestyle='--')

                    # Add labels and title
                    ax.set_title(f"Hpar: {i_hpar}, Sim: {i_sim}")

                    # Increase index
                    i_axes += 1

            # Name of PDF
            filepdf = os.path.join(
                plot_dir,
                f"hpar_{dataset}_{temp_nn}_pit_hists_best_{n_best}.pdf",
            )

            # Save as file
            plt.savefig(filepdf)
            plt.close()

            # Take time
            end_time = time_ns()

            log_message = (
                f"{dataset.upper()}, {ens_method}, {temp_nn.upper()}: Finished evaluation."
                f"- {(end_time - start_time) / 1e+9:.2f}s"
            )
            logging.info(log_message)


if __name__ == "__main__":
    # np.seterr(all="raise")
    main()
