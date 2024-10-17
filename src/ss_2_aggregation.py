## Simulation study: Script 2
# Aggregation of deep ensembles

# import concurrent.futures
import json
import logging
import os
import pickle
from functools import partial
from time import time_ns

import numpy as np
import pandas as pd
import scipy.stats as ss
from joblib import Parallel, delayed
from rpy2.robjects import default_converter, numpy2ri, vectors
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy.optimize import minimize

from fn_basic import fn_upit
from fn_eval import bern_quants, crps_hd, quant_hd, fn_scores_distr, fn_scores_ens, fn_scores_hd

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


### Functions for weight estimation (and uPIT) ###
def fn_vi_drn(a: float, w: float, f_sum, y: np.ndarray, **kwargs) -> float:
    """CRPS of VI forecast in case of DRN

    Parameters
    ----------
    a : scalar
        Intercept
    w : positive scalar
        Equal weight
    f_sum : n_train x 2 matrix
        Sum of single DRN forecasts
    y : n_train vector
        Observations

    Returns
    -------
    float
        CRPS of VI forecast
    """
    ### Initiation ###
    # Penalty for non-positive weights
    if w <= 0:
        return 1e6

    ### Calculation ###
    # Calculate weighted average
    f = w * f_sum

    # Add intercept term (only to location)
    f[:, 0] = a + f[:, 0]

    distr, lower, upper = kwargs["loss"]
    # Calculate CRPS
    scoring_rules = importr("scoringRules")
    np_cv_rules = default_converter + numpy2ri.converter
    y_vector = vectors.FloatVector(y)
    with localconverter(np_cv_rules) as cv:  # noqa: F841
        if distr == "0tnorm":  # 0-truncated normal distribution
            res = np.mean(
                scoring_rules.crps_tnorm(
                    y=y_vector, location=f[:, 0], scale=f[:, 1], lower=0
                )
            )
        elif distr == "tnorm":  # Truncated normal distribution
            res = np.mean(
                scoring_rules.crps_tnorm(
                    y=y_vector,
                    location=f[:, 0],
                    scale=f[:, 1],
                    lower=lower,
                    upper=upper,
                )
            )
        else:  # Normal distribution
            res = np.mean(
                scoring_rules.crps_norm(y=y_vector, mean=f[:, 0], sd=f[:, 1])
            )

    return res

def fn_vi_bqn(a, w, f_sum, y, **kwargs):
    """CRPS of VI forecast in case of BQN

    Parameters
    ----------
    a : scalar
        Intercept
    w : positive scalar
        Weight
    f_sum : n_train x 2 matrix
        Sum of single BQN coefficients
    y : n_train vector
        Observations
    q_levels : vector
        Quantile levels for evaluation

    Returns
    -------
    scalar
        CRPS of VI forecast
    """
    ### Initiation ###
    # penalty for non-positive weights
    if w <= 0:
        return 1e6

    ### Calculation ###
    # Calculate weighted average
    alpha = a + w * f_sum

    # Calculate quantiles (adds a to every entry)
    q = bern_quants(alpha=alpha, q_levels=kwargs["q_levels"])

    # Calculate CRPS
    scoring_rules = importr("scoringRules")
    np_cv_rules = default_converter + numpy2ri.converter
    y_vector = vectors.FloatVector(y)
    with localconverter(np_cv_rules) as cv:  # noqa: F841
        res = np.mean(scoring_rules.crps_sample(y=y_vector, dat=q))

    return res

def fn_vi_hen(a, w, f, bin_edges_sum, y, **kwargs):
    """CRPS of VI forecast in case of BQN

    Parameters
    ----------
    a : scalar
        Intercept
    w : positive scalar
        Weight
    f : list of n_train x n_bins matrices
        Bin probabilities
    bin_edges_sum : (n_bins+1) vector
        Accumulated bin edges
    y : n_train vector
        Observations

    Returns
    -------
    scalar
        CRPS of VI forecast
    """
    ### Initiation ###
    # penalty for non-positive weights
    if w <= 0:
        return 1e6

    ### Calculation ###
    # Calculate weighted average
    bin_edges_f = [(a + w * x) for x in bin_edges_sum]

    # Calculate CRPS
    res = np.mean(crps_hd(y=y, f=f, bin_edges=bin_edges_f))

    return res

def fn_apply_bqn(i, n_ens, **kwargs):
    # Sample individual distribution
    i_rep = np.random.choice(
        a=range(n_ens), size=kwargs["n_lp_samples"], replace=True
    )  # with replacement

    # Draw from individual distributions
    alpha_vec = [
        bern_quants(
            alpha=kwargs["f_ls"][f"alpha{j}"][i, :],
            q_levels=np.random.uniform(size=1),
        )
        for j in i_rep
    ]

    return np.reshape(np.asarray(alpha_vec), newshape=(kwargs["n_lp_samples"]))


### Parallel-Function ###
def fn_mc(
    temp_nn: str,
    dataset: str,
    ens_method: str,
    n_ens: int,
    i_sim: int,
    **kwargs,
) -> None:
    """Function for parallel computing

    Parameters
    ----------
    temp_nn : str
        Network type
    dataset : str
        Name of dataset
    ens_method : str
        Name of ensemble method
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
    # Initialize rpy elements for all scoring functions
    rpy_elements = {
        "base": importr("base"),
        "scoring_rules": importr("scoringRules"),
        "crch": importr("crch"),
        "np_cv_rules": default_converter + numpy2ri.converter,
    }

    # Get aggregation methods
    agg_meths = kwargs["agg_meths_ls"][temp_nn]

    # Get loss distribution
    distr, lower, upper = kwargs["loss"]

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
        # Average parameters
        # Sum mean and sd for each obs over ensembles
        f_sum = np.asarray(
            sum([f_ls[key] for key in f_ls.keys() if "f" in key])
        )

        # Calculate sum of validation set forecasts
        f_valid_sum = np.asarray(
            sum([f_valid_ls[key] for key in f_valid_ls.keys() if "f" in key])
        )
    elif temp_nn == "bqn":
        # Average forecasts
        f_sum = np.asarray(
            sum([f_ls[key] for key in f_ls.keys() if "f" in key])
        )

        # Average coefficients
        alpha_sum = np.asarray(
            sum([f_ls[key] for key in f_ls.keys() if "alpha" in key])
        )

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
        
        # Calculate cumulative probabilities
        p_cum_valid = calculate_cumulative_probs(f_valid_ls)
        p_cum = calculate_cumulative_probs(f_ls)

        # Sort by row and round
        p_cum_sort_valid = sort_rows(p_cum_valid)
        p_cum_sort = sort_rows(p_cum)

        # Calculate bin probabilities
        f_valid = calculate_bin_probs(p_cum_sort_valid)
        f_bins = calculate_bin_probs(p_cum_sort)

        # Update cumulative probabilities
        p_cum_sort_valid = [np.r_[0, np.minimum(1, np.cumsum(x))] for x in f_valid]
        p_cum_sort = [np.r_[0, np.minimum(1, np.cumsum(x))] for x in f_bins]

        # Generate bin edges
        bin_edges_valid_sum = generate_bin_edges(f_valid_ls, p_cum_sort_valid)
        bin_edges_sum = generate_bin_edges(f_ls, p_cum_sort)

        # Throw out multiple bin edges
        f_valid, bin_edges_valid_sum = merge_bins(f_valid, bin_edges_valid_sum)
        f_bins, bin_edges_sum = merge_bins(f_bins, bin_edges_sum)

    ### Aggregation ###
    # For-Loop over aggregation methods
    for temp_agg in agg_meths:
        # Create list
        pred_agg = {}

        # Take time
        start_time = time_ns()

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

            # Calculate evaluation measure of simulated ensemble (mixture)
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"], y=y_test, rpy_elements=rpy_elements
            )
        elif (temp_nn == "drn") & (temp_agg == "vi"):
            # Average parameters
            pred_agg["f"] = f_sum / n_ens

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr=distr,
                lower=lower,
                upper=upper,
                rpy_elements=rpy_elements,
            )
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

            # Optimize
            est = minimize(
                fun=fn_optim_drn_vi_w, x0=1 / n_ens, method="Nelder-Mead"
            )

            # Read out weight
            pred_agg["w"] = est.x

            # Calculate optimally weighted VI
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr=distr,
                lower=lower,
                upper=upper,
                rpy_elements=rpy_elements,
            )
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

            # Optimize
            est = minimize(
                fun=fn_optim_drn_vi_a,
                x0=0,
                method="Nelder-Mead",
            )

            # Read out intercept
            pred_agg["a"] = est.x

            # Calculate equally weighted VI
            pred_agg["f"] = f_sum / n_ens

            # Add intercept term (only to location)
            pred_agg["f"][:, 0] = pred_agg["a"] + pred_agg["f"][:, 0]

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr=distr,
                lower=lower,
                upper=upper,
                rpy_elements=rpy_elements,
            )
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

            # Optimize
            est = minimize(
                fun=fn_optim_drn_vi_aw, x0=[0, 1 / n_ens], method="Nelder-Mead"
            )

            # Read out intercept and weight
            pred_agg["a"] = est.x[0]
            pred_agg["w"] = est.x[1]

            # Calculate optimally weighted VI
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Add intercept term (only to location)
            pred_agg["f"][:, 0] = pred_agg["a"] + pred_agg["f"][:, 0]

            # Scores
            pred_agg["scores"] = fn_scores_distr(
                f=pred_agg["f"],
                y=y_test,
                distr=distr,
                lower=lower,
                upper=upper,
                rpy_elements=rpy_elements,
            )
        elif (temp_nn == "bqn") & (temp_agg == "lp"):
            # Function for mixture ensemble
            # Function on main level to allow for parallelization

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

            # Calculate evaluation measure of simulated ensemble (mixture)
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                rpy_elements=rpy_elements,
            )
        elif (temp_nn == "bqn") & (temp_agg == "vi"):
            # Average parameters
            pred_agg["alpha"] = alpha_sum / n_ens
            pred_agg["f"] = f_sum / n_ens

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
                rpy_elements=rpy_elements,
            )

            # Calculate bias of mean forecast (formula given)
            pred_agg["scores"]["e_me"] = ( np.mean(pred_agg["alpha"], axis=1) - y_test )
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

            # Optimize
            est = minimize(
                fun=fn_optim_bqn_vi_w, x0=1 / n_ens, method="Nelder-Mead"
            )

            # Read out weight
            pred_agg["w"] = est.x

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["w"] * alpha_sum
            pred_agg["f"] = pred_agg["w"] * f_sum

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
                rpy_elements=rpy_elements,
            )

            # Calculate bias of mean forecast (formula given)
            pred_agg["scores"]["e_me"] = ( np.mean(pred_agg["alpha"], axis=1) - y_test )
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

            # Optimize
            est = minimize(fun=fn_optim_bqn_vi_a, x0=0, method="Nelder-Mead")

            # Read out intercept
            pred_agg["a"] = est.x

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["a"] + alpha_sum / n_ens
            pred_agg["f"] = pred_agg["a"] + f_sum / n_ens

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
                rpy_elements=rpy_elements,
            )

            # Calculate bias of mean forecast (formula given)
            pred_agg["scores"]["e_me"] = ( np.mean(pred_agg["alpha"], axis=1) - y_test  )
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

            # Optimize
            est = minimize(
                fun=fn_optim_bqn_vi_aw, x0=[0, 1 / n_ens], method="Nelder-Mead"
            )

            # Read out intercept and weight
            pred_agg["a"] = est.x[0]
            pred_agg["w"] = est.x[1]

            # Optimally weighted parameters
            pred_agg["alpha"] = pred_agg["a"] + pred_agg["w"] * alpha_sum
            pred_agg["f"] = pred_agg["a"] + pred_agg["w"] * f_sum

            # Scores
            pred_agg["scores"] = fn_scores_ens(
                ens=pred_agg["f"],
                y=y_test,
                skip_evals=["e_me"],
                rpy_elements=rpy_elements,
            )

            # Calculate bias of mean forecast (formula given)
            pred_agg["scores"]["e_me"] = ( np.mean(pred_agg["alpha"], axis=1) - y_test )
        elif (temp_nn == "hen") & (temp_agg == "lp"):
            ## Chat GPT code, check with data format
            # Convert the values from f_ls to a NumPy array 
            f_values = np.array([f_ls[f"f{i}"] for i in range(n_ens)])

            # Calculate the mean along the third dimension (axis=0)
            pred_agg["f"] = np.mean(f_values, axis=0)

            # Calculate evaluation measure of simulated ensemble (mixture)
            pred_agg["scores"] = fn_scores_hd(
                f=pred_agg["f"],
                y=y_test,
                bin_edges=f_ls["bin_edges0"],
            )
        elif (temp_nn == "hen") & (temp_agg == "vi"):
            # Get edges and probabilities
            pred_agg["f"] = f_bins
            pred_agg["bin_edges_f"] = [x / n_ens for x in bin_edges_sum]

            # Calculate evaluation measure of simulated ensemble (mixture)
            pred_agg["scores"] = fn_scores_hd(
                f=pred_agg["f"],
                y=y_test,
                bin_edges=pred_agg["bin_edges_f"],
            )
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

            # Optimize
            est = minimize(
                fun=fn_optim_hen_vi_w, x0=1/n_ens, method="Nelder-Mead"
            )

            # Read out weight
            pred_agg["w"] = est.x

            # Get edges and probabilities
            pred_agg["f"] = f_bins
            pred_agg["bin_edges_f"] = [pred_agg["w"] * x for x in bin_edges_sum]

            # Calculate evaluation measure
            pred_agg["scores"] = fn_scores_hd(
                f=pred_agg["f"],
                y=y_test,
                bin_edges=pred_agg["bin_edges_f"],
            )
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

            # Optimize
            est = minimize(fun=fn_optim_hen_vi_a, x0=0, method="Nelder-Mead")

            # Read out intercept
            pred_agg["a"] = est.x

            # Get edges and probabilities
            pred_agg["f"] = f_bins
            pred_agg["bin_edges_f"] = [pred_agg["a"] + x / n_ens for x in bin_edges_sum]

            # Calculate evaluation measure of simulated ensemble (mixture)
            pred_agg["scores"] = fn_scores_hd(
                f=pred_agg["f"],
                y=y_test,
                bin_edges=pred_agg["bin_edges_f"],
            )
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

            # Optimize
            est = minimize(
                fun=fn_optim_hen_vi_aw, x0=[0, 1 / n_ens], method="Nelder-Mead"
            )

            # Read out intercept and weight
            pred_agg["a"] = est.x[0]
            pred_agg["w"] = est.x[1]

            # Get edges and probabilities
            pred_agg["f"] = f_bins
            pred_agg["bin_edges_f"] = [pred_agg["a"] + pred_agg["w"] * x for x in bin_edges_sum]

            # Calculate evaluation measure of simulated ensemble (mixture)
            pred_agg["scores"] = fn_scores_hd(
                f=pred_agg["f"],
                y=y_test,
                bin_edges=pred_agg["bin_edges_f"],
            )

        # Take time
        end_time = time_ns()
                    
        # Check
        if np.isnan(pred_agg["scores"]["pit"]).any():
            log_message = f"NaN in PIT detected in {filename}"
            logging.info(log_message)

        # Name of file
        filename = (
            f"{temp_nn}_sim_{i_sim}_{temp_agg}_ens_{n_ens}.pkl"  # noqa: E501
        )
        temp_data_out_path = os.path.join(kwargs["data_out_path"], filename)
        # Save aggregated forecasts and scores
        with open(temp_data_out_path, "wb") as f:
            pickle.dump(pred_agg, f)

        log_message = (
            f"{ens_method.upper()}, {dataset.upper()}, {temp_nn.upper()}: "
            f"Finished aggregation of {filename} - "
            f"{(end_time - start_time)/1e+9:.2f}s"
        )
        logging.info(log_message)
        # Delete and clean
        del pred_agg

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
    logging.info(msg="## Aggregation ##")

    # Cores to use
    num_cores = CONFIG["NUM_CORES"]

    # Network variantes
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]

    # Aggregation methods
    agg_meths_ls = CONFIG["PARAMS"]["AGG_METHS_LS"]

    # Models considered
    dataset_ls = CONFIG["DATASET"]

    # Ensemble method
    ens_method = CONFIG["ENS_METHOD"]

    # Number of simulations
    n_sim = CONFIG["PARAMS"]["N_SIM"]

    # Size of network ensembles
    n_ens = CONFIG["PARAMS"]["N_ENS"]

    # Loss function "norm", "0tnorm", "tnorm"
    loss = CONFIG["PARAMS"]["LOSS"]

    # Ensemble sizes to be combined
    step_size = 2
    n_ens_vec = np.arange(
        start=step_size, stop=n_ens + step_size, step=step_size
    )

    # Size of LP mixture samples
    n_lp_samples = 1_000  # 1000 solves problem of relatively low LP CRPSS

    # Size of BQN quantile samples
    n_q_samples = 99

    # Quantile levels for evaluation
    q_levels = np.arange(
        start=1 / (n_q_samples + 1), stop=1, step=1 / (n_q_samples + 1)
    )

    ### Initialize parallel computing ###
    # Grid for parallel computing
    run_grid = pd.DataFrame(columns=["dataset", "temp_nn", "n_ens", "i_sim"])
    for dataset in dataset_ls:
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
            CONFIG["PATHS"]["AGG_F"],
        )
        if dataset.startswith("scen"):
            temp_n_sim = n_sim
        elif dataset in ["gusts", "protein", "year"]:
            temp_n_sim = 5
        else:
            temp_n_sim = n_sim
        for temp_nn in nn_vec:
            for i_ens in n_ens_vec[::-1]:
                for i_sim in range(temp_n_sim):
                    # Initial value
                    file_check = True

                    # For-Loop over aggregation methods
                    for temp_agg in agg_meths_ls[temp_nn]:
                        # Name of file
                        filename = os.path.join(
                            data_out_path,
                            f"{temp_nn}_sim_{i_sim}_{temp_agg}_ens_{i_ens}.pkl",  # noqa: E501
                        )

                        # Change value if file does not exist
                        if not os.path.exists(filename):
                            file_check = False

                    # Continue with next case if files already exist
                    if file_check:
                        continue

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

    # Check if any runs need to be conducted
    if len(run_grid) == 0:
        logging.info(msg="Ensembles already aggregated")
        return

    ### Parallel-Loop ###
    # Maximum number of cores
    num_cores = min(num_cores, run_grid.shape[0])

    # Take time
    total_start_time = time_ns()

    # Run sequential or run parallel
    run_parallel = True
    # run_parallel = False

    log_message = f"Number of iterations needed: {run_grid.shape[0]}"
    logging.info(log_message)
    if run_parallel:
        ### Run parallel ###
        Parallel(n_jobs=num_cores, backend="multiprocessing")(
            delayed(fn_mc)(
                temp_nn=row["temp_nn"],
                dataset=row["dataset"],
                ens_method=row["ens_method"],
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
            for _, row in run_grid.iterrows()
        )
    else:
        ### Run sequential ###
        for _, row in run_grid.iterrows():
            fn_mc(
                temp_nn=row["temp_nn"],
                dataset=row["dataset"],
                ens_method=row["ens_method"],
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
    total_end_time = time_ns()

    # Print processing time
    log_message = (
        "Finished processing of all threads within "
        f"{(total_end_time - total_start_time) / 1e+9:.2f}s"
    )
    logging.info(log_message)


if __name__ == "__main__":
    main()
