## Hyperparameter tuning: Scores of different hyperparameter combinations


import json
import logging
import os
import pickle
import itertools
from time import time_ns

import numpy as np
import pandas as pd

from fn_eval import fn_cover

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
    data_ens_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
        "dataset",
        ens_method,
        CONFIG["PATHS"]["HPAR_F"],
    )

    # Path of results
    data_out_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
        "dataset",
        ens_method,
    )

    ### Initialize ###
    # Models considered
    dataset_ls = CONFIG["DATASET"]

    # Number of simulations
    n_sim = CONFIG["N_SIM"]
    n_sim_protein = CONFIG["N_SIM_PROTEIN"]

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

    # For-Loop over network types
    for temp_nn in nn_vec:
        # Network-dependent hyperparameters
        if temp_nn == "drn":
                hpar_type = hpar_type0
        elif temp_nn == "bqn":
                hpar_type = hpar_type0 + ["p_degree"]
        elif temp_nn == "hen":
                hpar_type = hpar_type0 + ["N_bins"]
        
        # Vector of column names
        col_vec_pp = (
            [
                "model",
                "n_sim",
            ]
            + sr_eval
            + hpar_type
        )
        
        # Load hyperparameters
        with open("src/hpar_ls.json", "rb") as f:
            CONFIG_HPAR = json.load(f)

        # Omit non-relevant hyperparameter
        if ens_method != "bayesian":
            del CONFIG_HPAR["prior"]
        if ens_method != "mc_dropout":
            del CONFIG_HPAR["p_dropout"]
        if temp_nn != "bqn":
            del CONFIG_HPAR["p_degree"]
        if temp_nn != "hen":
            del CONFIG_HPAR["N_bins"]
    
        # Generate a list of possible combinations
        n_hpar = len([dict(zip(CONFIG_HPAR.keys(), values)) for values in itertools.product(*CONFIG_HPAR.values())])
        
        # For-Loop over scenarios and simulations
        for dataset in dataset_ls:
            ### Create data frame ###
            # Take time
            start_time = time_ns()

            # Number of simulations for protein
            if dataset in ["gusts", "protein"]:
                temp_n_sim = n_sim_protein
            else:
                temp_n_sim = n_sim
            
            # Create data frame
            df_scores = pd.DataFrame(columns=col_vec_pp)
            df_runtime = pd.DataFrame(
                columns=[
                    "i_sim",
                    "dataset",
                    "runtime_train",
                    "runtime_pred",
                ]
            )
            
            # For-Loop over simulated runs
            for i_sim in range(temp_n_sim):
                ### Read out scores dependent on hyperparameter run ###
                # For-Loop over repetitions
                temp_dict = {}
                for i_hpar in range(n_hpar):
                    # Write in data frame
                    new_row = {
                        "model": dataset,
                        "n_sim": i_sim,
                        "i_hpar": i_hpar,
                        "nn": temp_nn,
                    }

                    # Load hyperparameter run
                    filename = os.path.join(
                        f"{temp_nn}_hpar_{i_hpar}_sim_{i_sim}.pkl",  # noqa: E501
                    )
                    temp_data_ens_path = data_ens_path.replace(
                        "dataset", dataset
                    )
                    with open(
                        os.path.join(temp_data_ens_path, filename), "rb"
                    ) as f:
                        pred_nn, _, _, hpars = pickle.load(
                            f
                        )  # pred_nn, y_valid, y_test, hpars

                    # Cut test data from scores
                    pred_nn["scores"] = pred_nn["scores"].iloc[0:pred_nn["n_valid"]]

                    # For-Loop over evaluation measures
                    for temp_sr in sr_eval:
                        # Depending on measure
                        if temp_sr == "mae":
                            new_row[temp_sr] = np.mean(
                                np.abs(pred_nn["scores"]["e_md"])
                            )
                        elif temp_sr == "me":
                            new_row[temp_sr] = np.mean(
                                pred_nn["scores"]["e_md"]
                            )
                        elif temp_sr == "rmse":
                            new_row[temp_sr] = np.sqrt(
                                np.mean(pred_nn["scores"]["e_me"] ** 2)
                            )
                        elif temp_sr == "cov":
                            new_row[temp_sr] = np.mean(
                                fn_cover(pred_nn["scores"]["pit"])
                            )
                        else:
                            new_row[temp_sr] = np.mean(
                                pred_nn["scores"][temp_sr]
                            )

                    # Add hyperparameters to list
                    for temp_hpar in hpars:
                        if temp_hpar == "layers":
                            new_row[temp_hpar] = str(hpars[temp_hpar])
                        else:
                            new_row[temp_hpar] = hpars[temp_hpar]
                    
                    # Append to data frame
                    df_scores = pd.concat(
                        [
                            df_scores,
                            pd.DataFrame(new_row, index=[0]),
                        ],
                        ignore_index=True,
                    )

                    ### Save runtime ###
                    runtime_train = pred_nn["runtime_est"]
                    runtime_pred = pred_nn["runtime_pred"]

                    new_row = {
                        "i_sim": i_sim,
                        "i_hpar": i_hpar,
                        "dataset": dataset,
                        "runtime_train": runtime_train,
                        "runtime_pred": runtime_pred,
                    }

                    # Append to data frame
                    df_runtime = pd.concat(
                        [
                            df_runtime,
                            pd.DataFrame(new_row, index=[0]),
                        ],
                        ignore_index=True,
                    )

            # Take time
            end_time = time_ns()

            ### Save ###
            filename = f"eval_{temp_nn}_hpar_{dataset}_{ens_method}.pkl"
            temp_data_out_path = data_out_path.replace("dataset", dataset)
            with open(os.path.join(temp_data_out_path, filename), "wb") as f:
                pickle.dump(df_scores, f)
            # Save runtime dataframe
            filename = f"runtime_{temp_nn}_hpar_{dataset}_{ens_method}.pkl"
            temp_data_out_path = data_out_path.replace("dataset", dataset)
            with open(os.path.join(temp_data_out_path, filename), "wb") as f:
                pickle.dump(df_runtime, f)

            log_message = (
                f"{dataset.upper()}, {ens_method}, {temp_nn.upper()}: Finished scoring. "
                f"- {(end_time - start_time) / 1e+9:.2f}s"
            )
            logging.info(log_message)


if __name__ == "__main__":
    # np.seterr(all="raise")
    main()
