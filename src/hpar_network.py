## Hyperparameter tuning: Runs for different hyperparameter combinations


import itertools
import json
import logging
import os
import pickle
import time
from typing import Any, Tuple, Type

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from joblib import Parallel, delayed
from nptyping import Float, NDArray
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr
from sklearn.utils import resample

import BQNModels  # noqa: F401
import DRNModels  # noqa: F401
import HENModels  # noqa: F401
from BaseModel import BaseModel
from fn_basic import fn_upit

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

METHOD_CLASS_CONFIG = {
    "mc_dropout": "Dropout",
    "variational_dropout": "VariationalDropout",
    "concrete_dropout": "ConcreteDropout",
    "bayesian": "Bayesian",
    "rand_init": "RandInit",
    "bagging": "RandInit",
    "batchensemble": "BatchEnsemble",
}
METHOD_NUM_MODELS = {
    "single_model": [
        "mc_dropout",
        "variational_dropout",
        "bayesian",
        "concrete_dropout",
    ],
    "multi_model": [
        "rand_init",
        "bagging",
    ],
}


def run_ensemble_single_model(
    dataset: str,
    i_sim: int,
    n_ens: int,
    i_hpar: int,
    hpars: list,
    nn_vec: list[str],
    data_in_path: str,
    data_out_path: str,
    loss: list[Any],
    ens_method: str = "mc_dropout",
    **kwargs,
) -> None:
    """Use one model to predict n_ens times

    Saves the following information to a pickle file:
    [pred_nn, y_valid, y_test]

    Parameters
    ----------
    dataset : str
        Name of dataset
    i_sim : int
        Simulation run
    n_ens : int
        Ensemble size
    i_hpar : int
        Index of chosen hyperparameters
    nn_vec : list[str]
        Contains NN types
    data_in_path : str
        Location of generated simulation data (see ss_0_data.py)
    data_out_path : str
        Location to save results
    ens_method : str
        Specifies the initialization method to use
    """
    ### Initialization ###
    # Initialize rpy elements for all scoring functions
    rpy_elements = {
        "base": importr("base"),
        "scoring_rules": importr("scoringRules"),
        "crch": importr("crch"),
        "np_cv_rules": default_converter + numpy2ri.converter,
    }

    n_mean_prediction = kwargs.get("n_mean_prediction")

    ### Get and split data ###
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = train_valid_test_split(
        data_in_path=data_in_path, dataset=dataset, i_sim=i_sim
    )

    ### Loop over network variants ###
    # For-Loop over network variants
    for temp_nn in nn_vec:
        # Read out class
        model_class = get_model_class(temp_nn, ens_method)

        # Set seed (same for each network variant)
        np.random.seed(123 + 100 * i_sim)

        ### Run model ###
        # Create model
        model = model_class(
            n_ens=n_ens,
            dataset=dataset,
            ens_method=ens_method,
            rpy_elements=rpy_elements,
            hpars = hpars,
            loss=loss,
            n_mean_prediction=n_mean_prediction,
        )

        # Build model
        model.fit(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )
        log_message = (
            f"{ens_method.upper()}, {dataset.upper()}, {temp_nn.upper()}: "
            "Finished training of "
            f"{temp_nn}_hpar_{i_hpar}_sim_{i_sim}.pkl "
            f"- {(model.runtime_est)/1e+9:.2f}s"
        )
        logging.info(log_message)

        # Take time
        start_time = time.time_ns()

        # Make prediction
        model.predict(X_test=np.r_[X_valid, X_test])

        # Get results
        pred_nn = model.get_results(y_test=np.hstack((y_valid, y_test)))

        # Take time
        end_time = time.time_ns()

        # Save ensemble member
        filename = os.path.join(
            f"{temp_nn}_hpar_{i_hpar}_sim_{i_sim}.pkl",  # noqa: E501
        )
        temp_data_out_path = os.path.join(data_out_path, filename)
        
        # Check for NaNs in predictions
        if np.any(np.isnan(pred_nn["f"])):
            log_message = f"NaNs predicted in {temp_data_out_path}"
            logging.error(log_message)

        with open(temp_data_out_path, "wb") as f:
            pickle.dump([pred_nn, y_valid, y_test, hpars], f)

        log_message = (
            f"{ens_method.upper()}, {dataset.upper()}, {temp_nn.upper()}: "
            f"Finished prediction of {filename} - "
            f"{(end_time - start_time)/1e+9:.2f}s"
        )
        logging.info(log_message)

        del model
        K.clear_session()


def run_ensemble_multi_model(
    dataset: str,
    i_sim: int,
    n_ens: int,
    i_hpar: int,
    hpars: list,
    nn_vec: list[str],
    data_in_path: str,
    data_out_path: str,
    loss: list[Any],
    ens_method: str = "rand_init",
    **kwargs,
) -> None:
    """Run and train a model type n_ens times

    Saves the following information to a pickle file:
    [pred_nn, y_valid, y_test]

    Parameters
    ----------
    dataset : str
        Name of dataset
    i_sim : int
        Simulation run
    n_ens : int
        Ensemble size
    i_hpar : int
        Index of chosen hyperparameters
    hpars : list
        List of hyperparameter to adapt
    nn_vec : list[str]
        Contains NN types
    data_in_path : str
        Location of generated simulation data (see ss_0_data.py)
    data_out_path : str
        Location to save results
    ens_method : str
        Specifies the initialization method to use
    """
    
    ### Initialization ###
    # Initialize rpy elements for all scoring functions
    rpy_elements = {
        "base": importr("base"),
        "scoring_rules": importr("scoringRules"),
        "crch": importr("crch"),
        "np_cv_rules": default_converter + numpy2ri.converter,
    }

    ### Get and split data ###
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = train_valid_test_split(
        data_in_path=data_in_path, dataset=dataset, i_sim=i_sim
    )

    ### Loop over network variants ###
    # For-Loop over network variants
    for temp_nn in nn_vec:
        # Read out class
        model_class = get_model_class(temp_nn, ens_method)

        # Set seed (same for each network variant)
        np.random.seed(123 + 100 * i_sim)

        # For-Loop over ensemble member
        for i_ens in range(n_ens):
            # Take time
            start_time = time.time_ns()

            # Bagging
            # Draw sample with replacement of same size (5_000)
            X_train_nn: NDArray
            y_train_nn: NDArray
            X_train_nn, y_train_nn = X_train, y_train

            ### Run model ###
            # Create model
            model = model_class(
                n_ens=n_ens,
                dataset=dataset,
                ens_method=ens_method,
                rpy_elements=rpy_elements,
                hpars=hpars,
                loss=loss,
            )

            # Build model
            model.fit(
                X_train=X_train_nn,
                y_train=y_train_nn,
                X_valid=X_valid,
                y_valid=y_valid,
            )

            log_message = (
                f"{ens_method.upper()}, {dataset.upper()}, {temp_nn.upper()}: "
                f"Finished training of {temp_nn}_hpar_{i_hpar}_sim_{i_sim}.pkl -"
                f" {(model.runtime_est)/1e+9:.2f}s"
            )
            logging.info(log_message)

            # Make prediction
            model.predict(X_test=np.r_[X_valid, X_test])

            # Get results
            pred_nn = model.get_results(y_test=np.hstack((y_valid, y_test)))

            # Take time
            end_time = time.time_ns()

            # Save ensemble member
            filename = os.path.join(
                f"{temp_nn}_hpar_{i_hpar}_sim_{i_sim}.pkl",  # noqa: E501
            )
            temp_data_out_path = os.path.join(data_out_path, filename)

            with open(temp_data_out_path, "wb") as f:
                pickle.dump([pred_nn, y_valid, y_test, hpars], f)

            log_message = (
                f"{ens_method.upper()}, {dataset.upper()}, {temp_nn.upper()}: "
                f"Finished prediction of {filename} - "
                f"{(end_time - start_time)/1e+9:.2f}s"
            )
            logging.info(log_message)

            del model
            K.clear_session()


def get_model_class(temp_nn: str, ens_method: str) -> Type[BaseModel]:
    """Get model class based on string

    Parameters
    ----------
    temp_nn : str
        DRN / BQN / HEN

    Returns
    -------
    BaseModel
        Returns class that inherits from abstract class BaseModel
    """
    temp_nn_upper = temp_nn.upper()
    module = globals()[f"{temp_nn_upper}Models"]
    method = METHOD_CLASS_CONFIG[ens_method]
    model_class = getattr(module, f"{temp_nn_upper}{method}Model")

    return model_class


def train_valid_test_split(
    data_in_path: str, dataset: str, i_sim: int
) -> Tuple[
    NDArray[Any, Float],
    NDArray[Any, Float],
    NDArray[Any, Float],
    NDArray[Any, Float],
    NDArray[Any, Float],
    NDArray[Any, Float],
]:
    """Performs data split in train, validation and test set

    Parameters
    ----------
    data_in_path : str
        Location of simulated data
    dataset : str
        Name of dataset
    i_sim : int
        Run number

    Returns
    -------
    tuple
        Contains (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    ### Simulated dataset ###
    if dataset.startswith("scen"):
        # Load corresponding data
        temp_data_in_path = os.path.join(data_in_path, f"sim_{i_sim}.pkl")

        with open(temp_data_in_path, "rb") as f:
            (
                X_train,  # n_train = 6_000
                y_train,
                X_test,  # n_test = 10_000
                y_test,
                _,
                _,
            ) = pickle.load(f)

        # Indices of validation set
        if dataset.endswith("6"):
            i_valid = np.arange(start=2_500, stop=3_000, step=1)
        else:
            i_valid = np.arange(
                start=5_000, stop=6_000, step=1
            )  # n_valid = 1000 -> n_train = 5000

        # Split X_train/y_train in train and validation set
        X_valid = X_train[i_valid]  # length 1_000
        y_valid: NDArray[Any, Float] = y_train[i_valid]
        X_train = np.delete(arr=X_train, obj=i_valid, axis=0)  # length 5_000
        y_train = np.delete(arr=y_train, obj=i_valid, axis=0)

    ### UCI Dataset ###
    # Code used from https://github.com/yaringal/DropoutUncertaintyExps
    else:
        data = np.loadtxt(os.path.join(data_in_path, "data.txt"))
        index_features = np.loadtxt(
            os.path.join(data_in_path, "index_features.txt")
        )
        index_target = np.loadtxt(
            os.path.join(data_in_path, "index_target.txt")
        )

        # Separate features and target
        X = data[:, [int(i) for i in index_features.tolist()]]
        y = data[:, int(index_target.tolist())]

        # Get train and test split (i_sim)
        index_train = np.loadtxt(
            os.path.join(data_in_path, f"index_train_{i_sim}.txt")
        )
        index_test = np.loadtxt(
            os.path.join(data_in_path, f"index_test_{i_sim}.txt")
        )

        X_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]]

        X_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]]

        # Add validation split
        num_training_examples = int(0.8 * X_train.shape[0])
        X_valid = X_train[num_training_examples:, :]
        y_valid = y_train[num_training_examples:]
        X_train = X_train[0:num_training_examples, :]
        y_train = y_train[0:num_training_examples]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def main():
    ### Get Config ###
    with open("src/config_hpar.json", "rb") as f:
        CONFIG = json.load(f)

    ### Initialize ###
    # Check whether files exists?
    file_check = True
    
    # Networks
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]
    
    # Datasets considered
    dataset_ls = CONFIG["DATASET"]

    # Number of simulated runs
    n_sim = CONFIG["N_SIM"]
    n_sim_protein = CONFIG["N_SIM_PROTEIN"]

    # Size of network ensembles
    n_ens = 1

    # Number of predictions to average per sample for single models
    n_mean_prediction = CONFIG["PARAMS"]["N_MEAN_PREDICTION"]

    # Loss function "norm", "0tnorm", "tnorm"
    loss = CONFIG["PARAMS"]["LOSS"]

    # Get method for generating ensemble members
    ens_method = CONFIG["ENS_METHOD"]

    # Get number of cores for parallelization
    num_cores = CONFIG["NUM_CORES"]

    # Train a single model for each ensemble member
    if ens_method in METHOD_NUM_MODELS["single_model"]:
        run_ensemble = run_ensemble_single_model
    # Or use the same model to predict each ensemble member
    elif ens_method in METHOD_NUM_MODELS["multi_model"]:
        run_ensemble = run_ensemble_multi_model

    # For-Loop over network variants
    for temp_nn in nn_vec:
        logging.log(msg=f"## {temp_nn.upper()} ##", level=25)

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
        hpar_ls = [dict(zip(CONFIG_HPAR.keys(), values)) for values in itertools.product(*CONFIG_HPAR.values())]

        # Generate grid with necessary information for each run
        run_grid = pd.DataFrame(
            columns=["dataset", "i_sim", "i_hpar", "data_in_path", "data_out_path"]
        )
        for dataset in dataset_ls:
            if dataset in ["gusts", "protein"]:
                temp_n_sim = n_sim_protein
            else:
                temp_n_sim = n_sim
            
            data_in_path = os.path.join(
                CONFIG["PATHS"]["DATA_DIR"],
                CONFIG["PATHS"]["INPUT_DIR"],
                dataset,
            )
            data_out_path = os.path.join(
                CONFIG["PATHS"]["DATA_DIR"],
                CONFIG["PATHS"]["RESULTS_DIR"],
                dataset,
                CONFIG["ENS_METHOD"],
                CONFIG["PATHS"]["HPAR_F"],
            )
            for i_sim in range(temp_n_sim):
                for i_hpar in range(len(hpar_ls)):
                    # Check if this file already exists
                    if file_check:
                        # Name of file
                        filename = os.path.join(
                            data_out_path,
                            f"{temp_nn}_hpar_{i_hpar}_sim_{i_sim}.pkl",  # noqa: E501
                        )

                        # Continue with next case if file already exists
                        if os.path.exists(filename):
                            continue
                    
                    new_row = {
                        "dataset": dataset,
                        "i_sim": i_sim,
                        "i_hpar": i_hpar,
                        "data_in_path": data_in_path,
                        "data_out_path": data_out_path,
                    }
                    run_grid = pd.concat(
                        [run_grid, pd.DataFrame(new_row, index=[0])],
                        ignore_index=True,
                    )

        # Check for model and agg directories and create if necessary
        check_directories(run_grid=run_grid)

        # Run sequential or run parallel
        run_parallel = True
        # run_parallel = False

        if run_parallel:
            ### Run parallel ###
            Parallel(n_jobs=num_cores, backend="multiprocessing")(
                delayed(run_ensemble)(
                    dataset=row["dataset"],
                    i_sim=row["i_sim"],
                    hpars=hpar_ls[row["i_hpar"]],
                    i_hpar=row["i_hpar"],
                    n_ens=n_ens,
                    nn_vec=[temp_nn],
                    data_in_path=row["data_in_path"],
                    data_out_path=row["data_out_path"],
                    ens_method=ens_method,
                    loss=loss,
                    n_mean_prediction=n_mean_prediction,
                )
                for _, row in run_grid.iterrows()
            )
        else:
            ### Run sequential ###
            for _, row in run_grid.iterrows():
                run_ensemble(
                    dataset=row["dataset"],
                    i_sim=row["i_sim"],
                    hpars=hpar_ls[row["i_hpar"]],
                    i_hpar=row["i_hpar"],
                    n_ens=n_ens,
                    nn_vec=[temp_nn],
                    data_in_path=row["data_in_path"],
                    data_out_path=row["data_out_path"],
                    ens_method=ens_method,
                    loss=loss,
                    n_mean_prediction=n_mean_prediction,
                )


def check_directories(run_grid) -> None:
    for i_row in range(len(run_grid)):
        temp_path = run_grid["data_out_path"][i_row]
        if not os.path.isdir(temp_path):
            os.makedirs(temp_path)


if __name__ == "__main__":
    main()
