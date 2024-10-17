## Simulation study: Script 4
# Diversity of deep ensembles

import json
import logging
import os
import pickle
from time import time_ns

import numpy as np
import pandas as pd

from fn_eval import fn_cover

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def main():
    ### Get Config ###
    with open("src/config_eval.json", "rb") as f:
        CONFIG = json.load(f)

    ### Settings ###
    ens_method = CONFIG["ENS_METHOD"]
    
    # Path of deep ensemble forecasts
    data_ens_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
        "dataset",
        ens_method,
        CONFIG["PATHS"]["ENSEMBLE_F"],
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
    n_sim = CONFIG["PARAMS"]["N_SIM"]

    # (Maximum) number of network ensembles
    n_rep = CONFIG["PARAMS"]["N_ENS"]

    # Ensemble sizes to be combined
    if n_rep > 1:
        step_size = 2
        n_ens_vec = np.arange(
            start=step_size, stop=n_rep + step_size, step=step_size
        )
    else:
        n_ens_vec = [0]

    # Network types
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]

    # To evaluate
    sr_eval = ["loc", "crps", "lgt", "z_loc", "z_crps", "z_lgt"]

    # Vector of column names
    col_vec_pp = (
        [
            "model",
            "nn",
            "n_sim",
            "n_ens",
        ]
        + sr_eval
    )

    # For-Loop over scenarios and simulations
    for dataset in dataset_ls:
        # Take time
        start_time = time_ns()

        ### Create data frame ###
        df_diversity = pd.DataFrame(columns=col_vec_pp)
        
        # Check number of partitions
        if dataset in ["gusts", "protein", "year"]:
            temp_n_sim = 5
        else:
            temp_n_sim = n_sim

        # For-Loop over network types
        for temp_nn in nn_vec:
            # For-Loop over partitions
            for i_sim in range(temp_n_sim):
                # For-Loop over number of aggregated members
                for i_ens in n_ens_vec:
                    # Write in data frame
                    new_row = {
                        "model": dataset,
                        "nn": temp_nn,
                        "n_sim": i_sim,
                        "n_ens": i_ens,
                    }

                    # Generate lists for mean, crps and lgt values
                    loc_ls = []
                    crps_ls = []
                    lgt_ls = []

                    # For-Loop over repetitions
                    for i_rep in range(i_ens):
                        # Load ensemble member
                        filename = os.path.join(
                            f"{temp_nn}_sim_{i_sim}_ens_{i_rep}.pkl",  # noqa: E501
                        )
                        temp_data_ens_path = data_ens_path.replace(
                            "dataset", dataset
                        )
                        with open(
                            os.path.join(temp_data_ens_path, filename), "rb"
                        ) as f:
                            pred_nn, _, y_test = pickle.load(
                                f
                            )  # pred_nn, y_valid, y_test

                        # Cut validation data from scores
                        pred_nn["scores"] = pred_nn["scores"].drop(
                            range(pred_nn["n_valid"])
                        )

                        # Convert length values in special case of HEN
                        if temp_nn == "hen":
                            pred_nn["scores"]["lgt"] = pred_nn["scores"]["lgt"].apply(lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x)

                        # Get values
                        loc_ls.append(pred_nn["scores"]["e_me"] + y_test)
                        crps_ls.append(pred_nn["scores"]["crps"])
                        lgt_ls.append(pred_nn["scores"]["lgt"])

                    # Transform to matrices
                    loc_mtx = np.array(loc_ls).T
                    crps_mtx = np.array(crps_ls).T
                    lgt_mtx = np.array(lgt_ls).T

                    # Calculate standard deviations
                    loc_std = loc_mtx.std(ddof=1)
                    crps_std = loc_mtx.std(ddof=1)
                    lgt_std = loc_mtx.std(ddof=1)
                    
                    # Special case: No differences
                    if loc_std == 0:
                        new_row["loc"] = 0
                        new_row["crps"] = 0
                        new_row["lgt"] = 0
                        new_row["z_loc"] = 0
                        new_row["z_crps"] = 0
                        new_row["z_lgt"] = 0
                        log_message = f"{temp_nn}, sim {i_sim}, ens {i_ens}"
                        logging.info(log_message)
                    else:
                        # Calculate standardized values
                        z_loc_mtx = (loc_mtx - loc_mtx.mean())/loc_std
                        z_crps_mtx = (crps_mtx - crps_mtx.mean())/crps_std
                        z_lgt_mtx = (lgt_mtx - lgt_mtx.mean())/lgt_std

                        # Calculate mean of standard deviations
                        new_row["loc"] = np.mean(loc_mtx.std(axis=1))
                        new_row["crps"] = np.mean(crps_mtx.std(axis=1))
                        new_row["lgt"] = np.mean(lgt_mtx.std(axis=1))
                        new_row["z_loc"] = np.mean(z_loc_mtx.std(axis=1))
                        new_row["z_crps"] = np.mean(z_crps_mtx.std(axis=1))
                        new_row["z_lgt"] = np.mean(z_lgt_mtx.std(axis=1))

                    # Append to data frame
                    df_diversity = pd.concat(
                        [
                            df_diversity,
                            pd.DataFrame(new_row, index=[0]),
                        ],
                        ignore_index=True,
                    )
        
        # Take time
        end_time = time_ns()

        ### Save ###
        filename = f"diversity_{dataset}_{ens_method}.pkl"
        temp_data_out_path = data_out_path.replace("dataset", dataset)
        with open(os.path.join(temp_data_out_path, filename), "wb") as f:
            pickle.dump(df_diversity, f)

        log_message = (
            f"{dataset.upper()}, {ens_method}: Finished diversity of {filename} "
            f"- {(end_time - start_time) / 1e+9:.2f}s"
        )
        logging.info(log_message)


if __name__ == "__main__":
    # np.seterr(all="raise")
    main()
