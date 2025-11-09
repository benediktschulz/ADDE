## Simulation study: Analyse computational costs of aggregation methods

import json
import logging
import os
import pickle
import sys
from functools import partial
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.optimize import minimize

from fn_eval import quant_hd

from ss_2_aggregation import (
    fn_apply_bqn,
    fn_vi_bqn,
    fn_vi_drn,
    fn_vi_hen,
)


### Set log Level ###
logging.basicConfig(
    format="%(asctime)s - %(message)s", 
    level=logging.INFO,
    filename='logging_time_keeping.log',
)

# Define function for time-keepimg
def update_time_dict(
    elapsed_time: float,
    time_dict: dict[str, float],
    time_keys: list[str],
) -> dict[str, float]:
    """
    Update time keeping dictionary of selected aggregation methods.

    Parameters
    ----------
    elapsed_time : float
        Elapsed time to add to dictionary 
    time_dict : dict[str, float]
        Current status of elapsed time per key
    time_keys : list[str]
        Key of time value to update

    Returns
    -------
    None
        Updated time keeping dictionary
    """
    for key in time_keys:
        time_dict[key] += elapsed_time

    return time_dict


### Parallel-Function ###
def fn_mc(
    temp_nn: str,
    n_ens: int,
    i_sim: int,
    **kwargs,
) -> None:
    """Function for parallel computing

    Parameters
    ----------
    temp_nn : str
        Network type
    n_ens : integer
        Size of network ensemble
    i_sim : integer
        Number of simulation

    Returns
    -------
    None
        Aggregation results are saved to data/agg
    """
    ### Initialization ###
    # Get aggregation methods
    agg_meths = kwargs["agg_meths_ls"][temp_nn]

    ## Get subset of keys for time keeping
    # VI methods (key for prediction only)
    vi_meths = [
        temp_agg for temp_agg in agg_meths 
        if temp_agg in ["vi", "vi-a", "vi-w", "vi-aw"]
    ]

    # VI methods with estimated parameters (key for prediction only)
    vi_coeff_meths = [
        temp_agg for temp_agg in agg_meths 
        if temp_agg in ["vi-a", "vi-w", "vi-aw"]
    ]

    # Keys for estimation of parameters (without prediction)
    vi_coeff_est = [
        f"{temp_agg}-est" for temp_agg in vi_coeff_meths
    ]
    
    # Get loss distribution
    distr, lower, upper = kwargs["loss"]

    # Create a dict for time keeping
    agg_times = {agg_meth: 0.0 for agg_meth in agg_meths + vi_coeff_est}

    # Create list for ensemble member forecasts on test and validation set
    f_ls = {}
    f_valid_ls = {}

    ### Get data ###
    # For-Loop over ensemble member
    y_test = []
    for i_ens in range(n_ens):
        # Load ensemble member
        filename = f"{temp_nn}_sim_{i_sim}_ens_{i_ens}.pkl"
        temp_data_in_path = os.path.join(kwargs["data_in_path"], filename)
        with open(temp_data_in_path, "rb") as f:
            pred_nn, y_valid, y_test = pickle.load(f)

        # Get indices of validation and test set
        i_valid = list(range(len(y_valid)))
        i_test = [x + len(y_valid) for x in range(len(y_test))]

        # Save forecasts
        if temp_nn == "drn":
            # Validation set
            f_valid_ls[f"f{i_ens}"] = pred_nn["f"][i_valid, :]

            # Test set
            f_ls[f"f{i_ens}"] = pred_nn["f"][i_test, :]
        elif temp_nn == "bqn":
            # Validation set
            f_valid_ls[f"f{i_ens}"] = pred_nn["f"][i_valid, :]
            f_valid_ls[f"alpha{i_ens}"] = pred_nn["alpha"][i_valid, :]

            # Test set
            f_ls[f"f{i_ens}"] = pred_nn["f"][i_test, :]
            f_ls[f"alpha{i_ens}"] = pred_nn["alpha"][i_test, :]
        elif temp_nn == "hen":
            # Function for normalizing probabilities (based on arrays)
            def norm_probs_np(p, p_min = 1e-10):
                p[p < p_min] = 0
                p = p / np.sum(p, axis=1, keepdims=True)

                return p
            
            # Adjust HEN forecasts
            pred_nn["f"] = norm_probs_np(pred_nn["f"])
            
            # Validation set
            f_valid_ls[f"f{i_ens}"] = pred_nn["f"][i_valid, :]
            f_valid_ls[f"bin_edges{i_ens}"] = pred_nn["bin_edges"]

            # Test set
            f_ls[f"f{i_ens}"] = pred_nn["f"][i_test, :]
            f_ls[f"bin_edges{i_ens}"] = pred_nn["bin_edges"]

    # Average forecasts depending on method
    f_sum = np.empty(shape=f_ls["f0"].shape)
    f_valid_sum = np.empty(shape=f_valid_ls["f0"].shape)
    alpha_sum = np.empty(shape=f_ls["f0"].shape)
    
    if temp_nn == "drn":
        # Start time measurement
        start = perf_counter()

        # Average parameters
        # Sum mean and sd for each obs over ensembles
        f_sum = np.asarray(
            sum([f_ls[key] for key in f_ls.keys() if "f" in key])
        )

        # End time measurement
        end = perf_counter()

        # Update times
        agg_times = update_time_dict(
            elapsed_time=(end - start),
            time_dict=agg_times,
            time_keys=vi_meths,
        )

        # Start time measurement
        start = perf_counter()

        # Calculate sum of validation set forecasts
        f_valid_sum = np.asarray(
            sum([f_valid_ls[key] for key in f_valid_ls.keys() if "f" in key])
        )

        # End time measurement
        end = perf_counter()

        # Update times
        agg_times = update_time_dict(
            elapsed_time=(end - start),
            time_dict=agg_times,
            time_keys=vi_coeff_est,
        )
    elif temp_nn == "bqn":
        # Start time measurement
        start = perf_counter()

        # Average forecasts
        f_sum = np.asarray(
            sum([f_ls[key] for key in f_ls.keys() if "f" in key])
        )

        # Average coefficients
        alpha_sum = np.asarray(
            sum([f_ls[key] for key in f_ls.keys() if "alpha" in key])
        )

        # End time measurement
        end = perf_counter()

        # Update times
        agg_times = update_time_dict(
            elapsed_time=(end - start),
            time_dict=agg_times,
            time_keys=vi_meths,
        )

        # Start time measurement
        start = perf_counter()

        # Calculate sum of validation set forecasts
        f_valid_sum = np.asarray(
            sum(
                [
                    f_valid_ls[key]
                    for key in f_valid_ls.keys()
                    if "alpha" in key
                ]
            )
        )

        # End time measurement
        end = perf_counter()

        # Update times
        agg_times = update_time_dict(
            elapsed_time=(end - start),
            time_dict=agg_times,
            time_keys=vi_coeff_est,
        )
    elif temp_nn == "hen":
        # Function to calculate cumulative probabilities
        def calculate_cumulative_probs(temp_f_ls):
            def fn_rep(i_rep):
                return [np.minimum(1, np.cumsum(x)) for x in temp_f_ls[f"f{i_rep}"]]
            
            temp_p_cum = np.concatenate(np.array([fn_rep(i_rep) for i_rep in range(n_ens)]), axis=1)

            return temp_p_cum

        # Function to sort rows
        def sort_rows(temp_p_cum):
            return [np.unique(np.r_[0, np.sort(row)]) for row in temp_p_cum]

        # Function to calculate and normalize bin probabilities
        def calculate_bin_probs(temp_p_cum, p_min = 1e-10):
            # Calculate from cumulative probabilities
            p = [np.diff(row) for row in temp_p_cum]

            # Normalize
            p = [row*(row >= p_min) for row in p]
            p = [row/sum(row) for row in p]

            return p

        # Function to generate bin edges
        def generate_bin_edges(temp_f_ls, temp_p_cum):
            def fn_rep(i_rep, i):
                return quant_hd(tau=temp_p_cum[i], 
                                probs=temp_f_ls[f"f{i_rep}"][i,:], 
                                bin_edges=temp_f_ls[f"bin_edges{i_rep}"])
            
            return [np.sum(np.array([
                fn_rep(i_rep=i_rep, i=i) for i_rep in range(n_ens)
                ]), axis=0) for i in range(len(temp_p_cum)) ]
            
        # Function to merge bins with duplicated edges
        def merge_bins(temp_f_ls, temp_edges):
            # Find cases with duplicated bin edges
            log_db = [np.any(np.diff(temp_edges[i]) == 0) for i in range(len(temp_edges))]

            # Merge bins if duplicates are included
            if np.any(log_db):
                # Go over each row
                for i_row in np.where(log_db)[0]:
                    # Get duplicated bin edges (w.r.t. upper bin)
                    log_edges = np.r_[(np.diff(temp_edges[i_row]) == 0), False]
                    i_bins = np.where(log_edges[:-1])[0]

                    # For-Loop (requuired due to adjacent bins)
                    for i_bin in i_bins:
                        # Merging indices (adjust for last bin)
                        if i_bin == (len(temp_f_ls[i_row]) - 1):
                            i_merge = i_bin - 1
                        else:
                            i_merge = i_bin + 1
                            
                        # Merge bin probabilities in upper/lower bin (w.r.t. i_bin)
                        temp_f_ls[i_row][i_merge] = temp_f_ls[i_row][i_bin] + temp_f_ls[i_row][i_merge]
                    
                    # Drop probs of duplicates that have not been updated
                    temp_f_ls[i_row] = temp_f_ls[i_row][~log_edges[:-1]]

                    # Drop duplicated edges
                    temp_edges[i_row] = temp_edges[i_row][~log_edges]

            return temp_f_ls, temp_edges
        
        # Start time measurement
        start = perf_counter()

        # Calculate cumulative probabilities
        p_cum = calculate_cumulative_probs(f_ls)

        # Sort by row and round
        p_cum_sort = sort_rows(p_cum)

        # Calculate bin probabilities
        f_bins = calculate_bin_probs(p_cum_sort)

        # Update cumulative probabilities
        p_cum_sort = [np.r_[0, np.minimum(1, np.cumsum(x))] for x in f_bins]

        # Generate bin edges
        bin_edges_sum = generate_bin_edges(f_ls, p_cum_sort)

        # Throw out multiple bin edges
        f_bins, bin_edges_sum = merge_bins(f_bins, bin_edges_sum)

        # End time measurement
        end = perf_counter()

        # Update times
        agg_times = update_time_dict(
            elapsed_time=(end - start),
            time_dict=agg_times,
            time_keys=vi_meths,
        )

        # Start time measurement
        start = perf_counter()

        # Calculate cumulative probabilities
        p_cum_valid = calculate_cumulative_probs(f_valid_ls)

        # Sort by row and round
        p_cum_sort_valid = sort_rows(p_cum_valid)

        # Calculate bin probabilities
        f_valid = calculate_bin_probs(p_cum_sort_valid)

        # Update cumulative probabilities
        p_cum_sort_valid = [np.r_[0, np.minimum(1, np.cumsum(x))] for x in f_valid]

        # Generate bin edges
        bin_edges_valid_sum = generate_bin_edges(f_valid_ls, p_cum_sort_valid)

        # Throw out multiple bin edges
        f_valid, bin_edges_valid_sum = merge_bins(f_valid, bin_edges_valid_sum)

        # End time measurement
        end = perf_counter()

        # Update times
        agg_times = update_time_dict(
            elapsed_time=(end - start),
            time_dict=agg_times,
            time_keys=vi_coeff_est,
        )

    ### Aggregation ###
    # For-Loop over aggregation methods
    for temp_agg in agg_meths:
        # Create list
        pred_agg = {}

        # Different cases
        if (temp_nn == "drn") & (temp_agg == "lp"):
            # Function for mixture ensemble
            def fn_apply_drn(i, **kwargs):
                # Sample individual distribution
                i_rep = np.random.choice(
                    a=range(n_ens), size=kwargs["n_lp_samples"], replace=True
                )  # with replacement

                # Get distributional parameters
                temp_f = np.asarray(
                    [kwargs["f_ls"][f"f{j}"][i, :] for j in i_rep]
                )

                # Draw from individual distribution
                if distr == "0tnorm":  # 0 truncated normal distribution
                    a = (0 - temp_f[:, 0]) / temp_f[:, 1]
                    b = np.full(
                        shape=temp_f[:, 0].shape, fill_value=float("inf")
                    )
                    res = ss.truncnorm.rvs(
                        size=kwargs["n_lp_samples"],
                        loc=temp_f[:, 0],
                        scale=temp_f[:, 1],
                        a=a,
                        b=b,
                    )
                elif distr == "tnorm":  # Truncated normal distribution
                    a = (lower - temp_f[:, 0]) / temp_f[:, 1]
                    b = (upper - temp_f[:, 0]) / temp_f[:, 1]
                    res = ss.truncnorm.rvs(
                        size=kwargs["n_lp_samples"],
                        loc=temp_f[:, 0],
                        scale=temp_f[:, 1],
                        a=a,
                        b=b,
                    )
                else:  # Normal distribution
                    res = ss.norm.rvs(
                        size=kwargs["n_lp_samples"],
                        loc=temp_f[:, 0],
                        scale=temp_f[:, 1],
                    )
                # Output
                return np.asarray(res)

            # Start time measurement
            start = perf_counter()

            # Simulate ensemble for mixture
            pred_agg["f"] = np.asarray(
                [
                    fn_apply_drn(i=i, **dict(kwargs, f_ls=f_ls))
                    for i in range(len(y_test))
                ]
            )

            # If nans get drawn: replace by mean
            if np.any(np.isnan(pred_agg["f"])):
                ind = np.where(np.isnan(pred_agg["f"]))
                for row, col in zip(ind[0], ind[1]):
                    replace_value = np.random.choice(pred_agg["f"][row, :])
                    while np.isnan(replace_value):
                        replace_value = np.random.choice(pred_agg["f"][row, :])
                    pred_agg["f"][row, col] = replace_value
                    log_message = f"NaN replaced in ({row}, {col})"
                    logging.warning(log_message)

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "drn") & (temp_agg == "vi"):
            # Start time measurement
            start = perf_counter()

            # Average parameters
            pred_agg["f"] = f_sum / n_ens

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "drn") & (temp_agg == "vi-w"):
            # Wrapper function
            def fn_optim_drn_vi_w(x):
                return fn_vi_drn(
                    a=0,
                    w=x,
                    f_sum=f_valid_sum,
                    y=y_valid,
                    loss=[distr, lower, upper],
                )

            # Start time measurement
            start = perf_counter()

            # Optimize
            est = minimize(
                fun=fn_optim_drn_vi_w, x0=1 / n_ens, method="Nelder-Mead"
            )
            
            # End time measurement
            end = perf_counter()

            # Update times
            agg_times = update_time_dict(
                elapsed_time=(end - start),
                time_dict=agg_times,
                time_keys=[f"{temp_agg}-est"],
            )

            # Read out weight
            pred_agg["w"] = est.x

            # Start time measurement
            start = perf_counter()

            # Calculate optimally weighted VI
            pred_agg["f"] = pred_agg["w"] * f_sum

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "drn") & (temp_agg == "vi-a"):
            # Wrapper function
            def fn_optim_drn_vi_a(x):
                return fn_vi_drn(
                    a=x,
                    w=1 / n_ens,
                    f_sum=f_valid_sum,
                    y=y_valid,
                    loss=[distr, lower, upper],
                )

            # Start time measurement
            start = perf_counter()

            # Optimize
            est = minimize(
                fun=fn_optim_drn_vi_a,
                x0=0,
                method="Nelder-Mead",
            )

            # End time measurement
            end = perf_counter()

            # Update times
            agg_times = update_time_dict(
                elapsed_time=(end - start),
                time_dict=agg_times,
                time_keys=[f"{temp_agg}-est"],
            )

            # Read out intercept
            pred_agg["a"] = est.x
            
            # Start time measurement
            start = perf_counter()

            # Calculate equally weighted VI
            pred_agg["f"] = f_sum / n_ens

            # Add intercept term (only to location)
            pred_agg["f"][:, 0] = pred_agg["a"] + pred_agg["f"][:, 0]

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "drn") & (temp_agg == "vi-aw"):
            # Wrapper function
            def fn_optim_drn_vi_aw(x):
                return fn_vi_drn(
                    a=x[0],
                    w=x[1],
                    f_sum=f_valid_sum,
                    y=y_valid,
                    loss=[distr, lower, upper],
                )

            # Start time measurement
            start = perf_counter()

            # Optimize
            est = minimize(
                fun=fn_optim_drn_vi_aw, x0=[0, 1 / n_ens], method="Nelder-Mead"
            )

            # End time measurement
            end = perf_counter()

            # Update times
            agg_times = update_time_dict(
                elapsed_time=(end - start),
                time_dict=agg_times,
                time_keys=[f"{temp_agg}-est"],
            )

            # Read out intercept and weight
            pred_agg["a"] = est.x[0]
            pred_agg["w"] = est.x[1]
            
            # Start time measurement
            start = perf_counter()

            # Calculate optimally weighted VI
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Add intercept term (only to location)
            pred_agg["f"][:, 0] = pred_agg["a"] + pred_agg["f"][:, 0]

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "bqn") & (temp_agg == "lp"):
            # Start time measurement
            start = perf_counter()

            # Simulate ensemble for mixture
            pred_agg["f"] = np.asarray(
                list(
                    map(
                        partial(
                            fn_apply_bqn,
                            **dict(kwargs, n_ens=n_ens, f_ls=f_ls),
                        ),
                        range(len(y_test)),
                    )
                )
            )

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "bqn") & (temp_agg == "vi"):
            # Start time measurement
            start = perf_counter()

            # Average parameters
            pred_agg["alpha"] = alpha_sum / n_ens
            pred_agg["f"] = f_sum / n_ens

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "bqn") & (temp_agg == "vi-w"):
            # Wrapper function
            def fn_optim_bqn_vi_w(x):
                return fn_vi_bqn(
                    a=0,
                    w=x,
                    f_sum=f_valid_sum,
                    y=y_valid,
                    q_levels=kwargs["q_levels"],
                )

            # Start time measurement
            start = perf_counter()

            # Optimize
            est = minimize(
                fun=fn_optim_bqn_vi_w, x0=1 / n_ens, method="Nelder-Mead"
            )

            # End time measurement
            end = perf_counter()

            # Update times
            agg_times = update_time_dict(
                elapsed_time=(end - start),
                time_dict=agg_times,
                time_keys=[f"{temp_agg}-est"],
            )

            # Read out weight
            pred_agg["w"] = est.x
            
            # Start time measurement
            start = perf_counter()

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["w"] * alpha_sum
            pred_agg["f"] = pred_agg["w"] * f_sum

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "bqn") & (temp_agg == "vi-a"):
            # Wrapper function
            def fn_optim_bqn_vi_a(x):
                return fn_vi_bqn(
                    a=x,
                    w=1 / n_ens,
                    f_sum=f_valid_sum,
                    y=y_valid,
                    q_levels=kwargs["q_levels"],
                )

            # Start time measurement
            start = perf_counter()

            # Optimize
            est = minimize(fun=fn_optim_bqn_vi_a, x0=0, method="Nelder-Mead")

            # End time measurement
            end = perf_counter()

            # Update times
            agg_times = update_time_dict(
                elapsed_time=(end - start),
                time_dict=agg_times,
                time_keys=[f"{temp_agg}-est"],
            )

            # Read out intercept
            pred_agg["a"] = est.x
            
            # Start time measurement
            start = perf_counter()

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["a"] + alpha_sum / n_ens
            pred_agg["f"] = pred_agg["a"] + f_sum / n_ens

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "bqn") & (temp_agg == "vi-aw"):
            # Wrapper function
            def fn_optim_bqn_vi_aw(x):
                return fn_vi_bqn(
                    a=x[0],
                    w=x[1],
                    f_sum=f_valid_sum,
                    y=y_valid,
                    q_levels=kwargs["q_levels"],
                )

            # Start time measurement
            start = perf_counter()

            # Optimize
            est = minimize(
                fun=fn_optim_bqn_vi_aw, x0=[0, 1 / n_ens], method="Nelder-Mead"
            )

            # End time measurement
            end = perf_counter()

            # Update times
            agg_times = update_time_dict(
                elapsed_time=(end - start),
                time_dict=agg_times,
                time_keys=[f"{temp_agg}-est"],
            )

            # Read out intercept and weight
            pred_agg["a"] = est.x[0]
            pred_agg["w"] = est.x[1]
            
            # Start time measurement
            start = perf_counter()

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["a"] + pred_agg["w"] * alpha_sum
            pred_agg["f"] = pred_agg["a"] + pred_agg["w"] * f_sum

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "hen") & (temp_agg == "lp"):
            # Convert the values from f_ls to a NumPy array 
            f_values = np.array([f_ls[f"f{i}"] for i in range(n_ens)])

            # Start time measurement
            start = perf_counter()

            # Calculate the mean along the third dimension (axis=0)
            pred_agg["f"] = np.mean(f_values, axis=0)

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "hen") & (temp_agg == "vi"):
            # Start time measurement
            start = perf_counter()

            # Get edges and probabilities
            pred_agg["f"] = f_bins
            pred_agg["bin_edges_f"] = [x / n_ens for x in bin_edges_sum]

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "hen") & (temp_agg == "vi-w"):
            # Wrapper function
            def fn_optim_hen_vi_w(x):
                return fn_vi_hen(
                    a=0,
                    w=x,
                    f=f_valid,
                    bin_edges_sum=bin_edges_valid_sum,
                    y=y_valid,
                )

            # Start time measurement
            start = perf_counter()

            # Optimize
            est = minimize(
                fun=fn_optim_hen_vi_w, x0=1/n_ens, method="Nelder-Mead"
            )

            # End time measurement
            end = perf_counter()

            # Update times
            agg_times = update_time_dict(
                elapsed_time=(end - start),
                time_dict=agg_times,
                time_keys=[f"{temp_agg}-est"],
            )

            # Read out weight
            pred_agg["w"] = est.x
            
            # Start time measurement
            start = perf_counter()

            # Get edges and probabilities
            pred_agg["f"] = f_bins
            pred_agg["bin_edges_f"] = [pred_agg["w"] * x for x in bin_edges_sum]

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "hen") & (temp_agg == "vi-a"):
            # Wrapper function
            def fn_optim_hen_vi_a(x):
                return fn_vi_hen(
                    a=x,
                    w=1/n_ens,
                    f=f_valid,
                    bin_edges_sum=bin_edges_valid_sum,
                    y=y_valid,
                )

            # Start time measurement
            start = perf_counter()

            # Optimize
            est = minimize(fun=fn_optim_hen_vi_a, x0=0, method="Nelder-Mead")

            # End time measurement
            end = perf_counter()

            # Update times
            agg_times = update_time_dict(
                elapsed_time=(end - start),
                time_dict=agg_times,
                time_keys=[f"{temp_agg}-est"],
            )

            # Read out intercept
            pred_agg["a"] = est.x
            
            # Start time measurement
            start = perf_counter()

            # Get edges and probabilities
            pred_agg["f"] = f_bins
            pred_agg["bin_edges_f"] = [pred_agg["a"] + x / n_ens for x in bin_edges_sum]

            # End time measurement
            end = perf_counter()
        elif (temp_nn == "hen") & (temp_agg == "vi-aw"):
            # Wrapper function
            def fn_optim_hen_vi_aw(x):
                return fn_vi_hen(
                    a=x[0],
                    w=x[1],
                    f=f_valid,
                    bin_edges_sum=bin_edges_valid_sum,
                    y=y_valid,
                )

            # Start time measurement
            start = perf_counter()

            # Optimize
            est = minimize(
                fun=fn_optim_hen_vi_aw, x0=[0, 1 / n_ens], method="Nelder-Mead"
            )

            # End time measurement
            end = perf_counter()

            # Update times
            agg_times = update_time_dict(
                elapsed_time=(end - start),
                time_dict=agg_times,
                time_keys=[f"{temp_agg}-est"],
            )

            # Read out intercept and weight
            pred_agg["a"] = est.x[0]
            pred_agg["w"] = est.x[1]
            
            # Start time measurement
            start = perf_counter()

            # Get edges and probabilities
            pred_agg["f"] = f_bins
            pred_agg["bin_edges_f"] = [pred_agg["a"] + pred_agg["w"] * x for x in bin_edges_sum]

            # End time measurement
            end = perf_counter()

        # Update times
        agg_times = update_time_dict(
            elapsed_time=(end - start),
            time_dict=agg_times,
            time_keys=[temp_agg],
        )
        
        # Delete and clean
        del pred_agg

    # Name of file
    filename = (
        f"time_keeping_{temp_nn}_sim_{i_sim}_ens_{n_ens}.pkl"  # noqa: E501
    )
    temp_data_out_path = os.path.join(kwargs["data_out_path"], filename)

    # Save aggregated forecasts and scores
    with open(temp_data_out_path, "wb") as f:
        pickle.dump(agg_times, f)

    log_message = f"File saved: {temp_data_out_path}"
    logging.info(log_message)

    # Delete
    if temp_nn == "drn":
        del f_sum, f_valid_sum
    elif temp_nn == "bqn":
        del f_sum, alpha_sum, f_valid_sum
    elif temp_nn == "hen":
        del p_cum, p_cum_valid, p_cum_sort, p_cum_sort_valid, f_bins, f_valid, bin_edges_valid_sum, bin_edges_sum


def main():
    ### Get Config ###
    with open("src/config.json", "rb") as f:
        CONFIG = json.load(f)

    ### Initialize ###
    # Output
    logging.info(msg="## Time Keeping ##")

    # Network variantes
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]

    # Aggregation methods
    agg_meths_ls = CONFIG["PARAMS"]["AGG_METHS_LS"]

    # Models considered
    dataset_ens_method_pairs = [
        ("boston", "bagging"),
        ("kin8nm", "bayesian"),
    ]

    # Size of network ensembles
    n_ens = CONFIG["PARAMS"]["N_ENS"]

    # Size of network ensembles
    n_sim = CONFIG["PARAMS"]["N_SIM"]

    # Loss function "norm", "0tnorm", "tnorm"
    loss = CONFIG["PARAMS"]["LOSS"]

    # Ensemble sizes to be combined
    step_size = 2
    n_ens_vec = np.arange(
        start=step_size,
        stop=n_ens + step_size,
        step=step_size,
    )

    # Size of LP mixture samples
    n_lp_samples = 1_000  # 1000 solves problem of relatively low LP CRPSS

    # Size of BQN quantile samples
    n_q_samples = 99

    # Quantile levels for evaluation
    q_levels = np.arange(
        start=1 / (n_q_samples + 1), stop=1, step=1 / (n_q_samples + 1)
    )

    # Grid for runs
    run_grid = pd.DataFrame(columns=["dataset", "temp_nn", "n_ens", "i_sim"])
    
    for (dataset, ens_method) in dataset_ens_method_pairs:
        data_in_path = os.path.join(
            CONFIG["PATHS"]["DATA_DIR"],
            CONFIG["PATHS"]["RESULTS_DIR"],
            dataset,
            ens_method,
            CONFIG["PATHS"]["ENSEMBLE_F"],
        )
        data_out_path = os.path.join(
            CONFIG["PATHS"]["DATA_DIR"],
            CONFIG["PATHS"]["RESULTS_DIR"],
            dataset,
            ens_method,
            "time_keeping",
        )

        # Get number of repetitions
        if dataset in ["gusts", "protein", "year"]:
            temp_n_sim = 5
        else:
            temp_n_sim = n_sim

        for temp_nn in nn_vec:
            for i_ens in n_ens_vec[::-1]:
                for i_sim in range(temp_n_sim):
                    new_row = {
                        "dataset": dataset,
                        "ens_method": ens_method,
                        "temp_nn": temp_nn,
                        "n_ens": i_ens,
                        "i_sim": i_sim,
                        "data_in_path": data_in_path,
                        "data_out_path": data_out_path,
                    }

                    run_grid = pd.concat(
                        [run_grid, pd.DataFrame(new_row, index=[0])],
                        ignore_index=True,
                    )

    log_message = f"Number of iterations needed: {run_grid.shape[0]}"
    logging.info(log_message)

    # Take time
    total_start_time = perf_counter()
    
    for i_row, row in run_grid.iterrows():
        log_message = f"Number of iteration: {i_row}/{run_grid.shape[0]}"
        logging.info(log_message)

        fn_mc(
            temp_nn=row["temp_nn"],
            n_ens=row["n_ens"],
            i_sim=row["i_sim"],
            q_levels=q_levels,
            nn_vec=nn_vec,
            agg_meths_ls=agg_meths_ls,
            data_in_path=row["data_in_path"],
            data_out_path=row["data_out_path"],
            n_lp_samples=n_lp_samples,
            n_q_samples=n_q_samples,
            loss=loss,
        )

    # Take time
    total_end_time = perf_counter()

    # Print processing time
    log_message = (
        "Finished processing of all threads within "
        f"{(total_end_time - total_start_time)/60:.2f}min"
    )
    logging.info(log_message)


if __name__ == "__main__":
    main()
