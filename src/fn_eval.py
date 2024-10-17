## Function file
# Evaluation of probabilistic forecasts
import logging

import numpy as np
import pandas as pd
import scipy.stats as ss
from rpy2.robjects import default_converter, numpy2ri, vectors
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from fn_basic import fn_upit

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


### Coverage ###
def fn_cover(x, alpha=0.1):
    """Calculate coverage of a probabilistic forecast

    Parameters
    ----------
    x : n vector
        PIT values / ranks
    alpha : probability
        Significance level, by default 0.1

    Returns
    -------
    n vector
        Coverage in percentage
    """
    ### Coverage calculation ###
    # Mean coverage
    res = np.mean((alpha / 2 <= x) & (x <= (1 - alpha / 2)))

    # Output as percentage
    return 100 * res


### BQN: Bernstein Quantile function ###
def bern_quants(alpha, q_levels):
    """Function that calculates quantiles for given coefficients

    1. Called from BQN LP: alpha is vector of length "dimension", q_level
       is scalar
    2. Called after BQN NN: Converting n alpha predictions into fixed set of
       quantiles
    3. Called from BQN VI-a/w: Optimizing a and/or w based on validation set
       finally minimizing sample CRPS

    Parameters
    ----------
    alpha : n x (p_degree + 1) matrix
        Coefficients of Bernstein basis
    q_levels : n_q vector
        Quantile levels

    Returns
    -------
    n x n_q matrix
        Quantile forecasts for given coefficients
    """
    ### Initiation ###
    if len(alpha.shape) == 1:
        p_degree = alpha.shape[0] - 1
    else:
        p_degree = alpha.shape[1] - 1

    ### Calculation ###
    # Calculate quantiles (sum of coefficients times basis polynomials)
    if len(q_levels) == 1:
        # 1. Called from LP sampling (one set of bern_coeffs and one quantile)
        fac = ss.binom.pmf(list(range(p_degree + 1)), n=p_degree, p=q_levels)
        return np.dot(alpha, fac)
    else:
        # 3. Called from Neural Network based on predicted bern_coeffs and
        # given quantiles
        fac = [
            ss.binom.pmf(list(range(p_degree + 1)), n=p_degree, p=current_q)
            for current_q in q_levels
        ]
        fac = np.asarray(fac)
        return np.dot(alpha, fac.T)


### ENS: Evaluation of ensemble ###
def fn_scores_ens(ens, y, alpha=0.1, skip_evals=None, scores_ens=True, rpy_elements=None):
    """
    Scores a forecast based forecasted and 'true' samples.

    Parameters
    ----------
    ens : n x n_ens matrix
        Ensemble data for prediction
    y : n vector
        Observations for prediction
    alpha : probability
        Significance level for PI, by default 0.1
    skip_evals : string vector
        Skip the following evaluation measures, Default: None -> Calculate
        all
    scores_ens : bool
        Should scores of ensemble forecasts, interval lengths, ranks be
        calculated?, Default: True -> Calculate

    Returns
    -------
    scores_ens : n x 6 DataFrame
        DataFrame containing:
            pit : n vector
                uPIT values of observations in ensemble forecasts
            crps : n vector
                CRPS of ensemble forecasts
            logs : n vector
                Log-Score of ensemble forecasts
            lgt : n vector
                Length of PI
            e_md : n vector
                Bias of median forecast
            e_me : n vector
                Bias of mean forecast

    Raises
    ------
    Exception
        If ensemble should not be scored
    """
    ### Initiation ###
    # Load R packages
    if rpy_elements is None:
        base = importr("base")
        scoring_rules = importr("scoringRules")
        np_cv_rules = default_converter + numpy2ri.converter
    else:
        base = rpy_elements["base"]
        scoring_rules = rpy_elements["scoring_rules"]
        np_cv_rules = rpy_elements["np_cv_rules"]
    # Convert y to rpy2.robjects.vectors.FloatVector
    y_vector = vectors.FloatVector(y)

    # Calculate only if scores_ens is True
    if not scores_ens:
        raise Exception("Ensemble should not be scored.")

    # Get number of ensembles
    n = ens.shape[0]

    # Get ensemble size
    n_ens = ens.shape[1]

    # Make data frame
    scores_ens = pd.DataFrame(
        columns=["pit", "crps", "logs", "lgt", "e_me", "e_md"], index=range(n)
    )

    ### Calculation ###
    # Calculate observation ranks
    if "pit" in scores_ens.columns:
        with localconverter(np_cv_rules) as cv:
            # Calculate ranks
            scores_ens["pit"] = np.apply_along_axis(
                func1d=lambda x: base.rank(x, ties="random")[0],
                axis=1,
                arr=np.c_[y, ens],
            )
            #Calculate uPIT values from ranks
            scores_ens["pit"] = fn_upit(
                ranks = scores_ens["pit"],
                max_rank = n_ens + 1,
            )

    # Calculate CRPS or raw ensemble
    if "crps" in scores_ens.columns:
        with localconverter(np_cv_rules) as cv:
            scores_ens["crps"] = scoring_rules.crps_sample(y=y_vector, dat=ens)

    # Calculate Log-Score of raw ensemble
    if "logs" in scores_ens.columns:
        with localconverter(np_cv_rules) as cv:  # noqa: F841
            scores_ens["logs"] = scoring_rules.logs_sample(y=y_vector, dat=ens)

    # Calculate 1-alpha prediction interval
    if "lgt" in scores_ens.columns:
        # Calculate corresponding via quantile function
        scores_ens["lgt"] = np.apply_along_axis(
            func1d=lambda x: np.diff(
                np.quantile(
                    a=x,
                    q=np.hstack((alpha/2, 1-alpha/2)),
                    method="median_unbiased",
                )
            ),
            axis=1,
            arr=ens,
        )

    # Calculate bias of mean forecast
    if "e_me" in scores_ens.columns:
        scores_ens["e_me"] = np.mean(a=ens, axis=1) - y

    # Calculate bias of median forecast
    if "e_md" in scores_ens.columns:
        scores_ens["e_md"] = np.median(a=ens, axis=1) - y

    ### Output ###
    # Skip evaluation measures
    if skip_evals is not None:
        scores_ens.drop(columns=skip_evals, inplace=True)

    # Return output
    return scores_ens


# Scores of a parametric distribution
def fn_scores_distr(
    f,
    y,
    distr="tlogis",
    lower=None,
    upper=None,
    alpha=0.1, 
    skip_evals=None,
    rpy_elements=None,
) -> pd.DataFrame:  # type: ignore
    """
    Scores a forecast based on the distributional parameters.
    Differentiate between 4 possible distributions:

    - tlogis: Truncated logistic distribution.
    - normal: Normal distribution.
    - 0tnorm: 0-truncated normal distribution.
    - tnorm: upper-lower truncated normal distribution.

    Parameters
    ----------
    f : n x n_par matrix
        Ensemble data for prediction
    y : n vector
        Observations
    distr : "tlogis", "norm", "0tnorm", "tnorm"
        Parametric distribution, Default: (zero-)truncated logistic
    lower : float
        Speciefies lower truncation, only needed if distr="tnorm"
    upper : float
        Specifies upper truncation, only needed if distr="tnorm"
    alpha : probability
        Significance level for PI, by default 0.1
    skip_evals : string vector
        Skip the following evaluation measures, Default: None -> Calculate
        all

    Returns
    -------
    scores_pp : n x 6 DataFrame
        DataFrame containing:
            pit : n vector
                PIT values of distributional forecasts
            crps : n vector
                CRPS of forecasts
            logs : n vector
                Log-Score of forecasts
            lgt : n vector
                Length of prediction interval
            e_md : n vector
                Bias of median forecast
            e_me : n vector
                Bias of mean forecast
    """
    ### Initiation ###
    # Load R packages
    if rpy_elements is None:
        scoring_rules = importr("scoringRules")
        crch = importr("crch")
        np_cv_rules = default_converter + numpy2ri.converter
    else:
        scoring_rules = rpy_elements["scoring_rules"]
        crch = rpy_elements["crch"]
        np_cv_rules = rpy_elements["np_cv_rules"]
    # Convert y to rpy2.robjects.vectors.FloatVector
    y_vector = vectors.FloatVector(y)

    # Input check
    if (distr not in ["tlogis", "norm"]) & any(f[:, 1] < 0):
        logging.error("Non-positive scale forecast!")

    ### Data preparation ###
    # Number of predictions
    n = f.shape[0]

    # Make data frame
    scores_pp = pd.DataFrame(
        index=range(n), columns=["pit", "crps", "logs", "lgt", "e_me", "e_md"]
    )

    ### Prediction and score calculation ###
    # Forecasts depending on distribution
    if distr == "tlogis":  # truncated logistic
        # Calculate PIT values
        if "pit" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["pit"] = crch.ptlogis(
                    q=y, location=f[:, 0], scale=f[:, 1], left=0
                )

        # Calculate CRPS of forecasts
        if "crps" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["crps"] = scoring_rules.crps_tlogis(
                    y=y_vector, location=f[:, 0], scale=f[:, 1], lower=0
                )

        # Calculate Log-Score of forecasts
        if "logs" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["logs"] = scoring_rules.logs_tlogis(
                    y=y_vector, location=f[:, 0], scale=f[:, 1], lower=0
                )

        # Calculate length of 1-alpha % prediction interval
        if "lgt" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["lgt"] = crch.qtlogis(
                    p=1-alpha/2,
                    location=f[:, 0],
                    scale=f[:, 1],
                    left=0,
                ) - crch.qtlogis(
                    p=alpha/2,
                    location=f[:, 0],
                    scale=f[:, 1],
                    left=0,
                )

        # Calculate bias of median forecast
        if "e_md" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["e_md"] = (
                    crch.qtlogis(
                        p=0.5,
                        location=f[:, 0],
                        scale=f[:, 1],
                        left=0,
                    )
                    - y
                )

        # Calculate bias of mean forecast
        if "e_me" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["e_me"] = (
                    f[:, 1]
                    - f[:, 2] * np.log(1 - crch.plogis(-f[:, 0] / f[:, 1]))
                ) / (1 - crch.plogis(-f[:, 0] / f[:, 1])) - y

    elif distr == "norm":  # normal
        # Calculate PIT values
        if "pit" in scores_pp.columns:
            scores_pp["pit"] = ss.norm.cdf(x=y, loc=f[:, 0], scale=f[:, 1])

        # Calculate CRPS of forecasts
        if "crps" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["crps"] = scoring_rules.crps_norm(
                    y=y_vector, location=f[:, 0], scale=f[:, 1]
                )

        # Calculate Log-Score of forecasts
        if "logs" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:  # noqa: F841
                scores_pp["logs"] = scoring_rules.logs_norm(
                    y=y_vector, location=f[:, 0], scale=f[:, 1]
                )

        # Calculate length of ~(n_ens-1)/(n_ens+1) % prediction interval
        if "lgt" in scores_pp.columns:
            scores_pp["lgt"] = ss.norm.ppf(
                q=1-alpha/2, loc=f[:, 0], scale=f[:, 1]
            ) - ss.norm.ppf(q=alpha/2, loc=f[:, 0], scale=f[:, 1])

        # Calculate bias of mean forecast
        if "e_md" in scores_pp.columns:
            scores_pp["e_me"] = f[:, 0] - y

        # Calculate bias of median forecast
        if "e_md" in scores_pp.columns:
            scores_pp["e_md"] = f[:, 0] - y

        ### Output ###
        # Skip evaluation measures
        if skip_evals is not None:
            scores_pp.drop(columns=skip_evals, inplace=True)

    elif distr == "0tnorm":  # 0-truncated normal
        a = (0 - f[:, 0]) / f[:, 1]
        b = np.full(shape=f[:, 1].shape, fill_value=float("inf"))
        # Calculate PIT values
        if "pit" in scores_pp.columns:
            scores_pp["pit"] = ss.truncnorm.cdf(
                x=y, a=a, b=b, loc=f[:, 0], scale=f[:, 1]
            )

        # Calculate 0 truncated CRPS of forecasts
        if "crps" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["crps"] = scoring_rules.crps_tnorm(
                    y=y_vector, location=f[:, 0], scale=f[:, 1], lower=0
                )

        # Calculate Log-Score of forecasts
        if "logs" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:  # noqa: F841
                scores_pp["logs"] = scoring_rules.logs_tnorm(
                    y=y_vector, location=f[:, 0], scale=f[:, 1], lower=0
                )

        # Calculate length of ~(n_ens-1)/(n_ens+1) % prediction interval
        if "lgt" in scores_pp.columns:
            scores_pp["lgt"] = ss.truncnorm.ppf(
                q=1-alpha/2, loc=f[:, 0], scale=f[:, 1], a=a, b=b
            ) - ss.truncnorm.ppf(
                q=alpha/2, loc=f[:, 0], scale=f[:, 1], a=a, b=b
            )

        # Calculate bias of mean forecast
        scores_pp["e_me"] = f[:, 0] - y

        # Calculate bias of median forecast
        if "e_md" in scores_pp.columns:
            scores_pp["e_md"] = (
                ss.truncnorm.ppf(q=0.5, loc=f[:, 0], scale=f[:, 1], a=a, b=b)
                - y
            )

        ### Output ###
        # Skip evaluation measures
        if skip_evals is not None:
            scores_pp.drop(columns=skip_evals, inplace=True)

    elif distr == "tnorm":  # truncated normal
        a = (lower - f[:, 0]) / f[:, 1]
        b = (upper - f[:, 0]) / f[:, 1]
        # Calculate PIT values
        if "pit" in scores_pp.columns:
            scores_pp["pit"] = ss.truncnorm.cdf(
                x=y, a=a, b=b, loc=f[:, 0], scale=f[:, 1]
            )

        # Calculate 0 truncated CRPS of forecasts
        if "crps" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:
                scores_pp["crps"] = scoring_rules.crps_tnorm(
                    y=y_vector,
                    location=f[:, 0],
                    scale=f[:, 1],
                    lower=lower,
                    upper=upper,
                )

        # Calculate Log-Score of forecasts
        if "logs" in scores_pp.columns:
            with localconverter(np_cv_rules) as cv:  # noqa: F841
                scores_pp["logs"] = scoring_rules.logs_tnorm(
                    y=y_vector,
                    location=f[:, 0],
                    scale=f[:, 1],
                    lower=lower,
                    upper=upper,
                )

        # Calculate length of ~(n_ens-1)/(n_ens+1) % prediction interval
        if "lgt" in scores_pp.columns:
            scores_pp["lgt"] = ss.truncnorm.ppf(
                q=1-alpha/2, loc=f[:, 0], scale=f[:, 1], a=a, b=b
            ) - ss.truncnorm.ppf(
                q=alpha/2, loc=f[:, 0], scale=f[:, 1], a=a, b=b
            )

        # Calculate bias of mean forecast
        scores_pp["e_me"] = f[:, 0] - y

        # Calculate bias of median forecast
        if "e_md" in scores_pp.columns:
            scores_pp["e_md"] = (
                ss.truncnorm.ppf(q=0.5, loc=f[:, 0], scale=f[:, 1], a=a, b=b)
                - y
            )

        ### Output ###
        # Skip evaluation measures
        if skip_evals is not None:
            scores_pp.drop(columns=skip_evals, inplace=True)
    # Return
    return scores_pp

### Histogram distribution ###
# CDF of a histogram distribution
def cdf_hd(y, probs, bin_edges):
    """Function that calculates the CDF of a histogram distribution

    Parameters
    ----------
    y: n vector
        Observations
    probs: n_bins vectors
        Probabilities of corresponding bins
    bin_edges: (n_bins+1) vector
        Boundaries of bins
        
    Returns
    -------
    n vector
        CDF of forecasts at observations
    """

    #### Initiation ####
    # Left/right edges of bins
    a = bin_edges[:-1]
    b = bin_edges[1:]

    #### Calculation ####
    # Calculate bin of values (Omit case that y is outside of range from here, considered later)
    k = np.searchsorted(b[:-1], y, side='right')

    # Calculate cumulative sums of bin probabilities
    # Entry k corresponds to sum of all k-1 bins below
    p_cum = np.r_[0, np.cumsum(probs[:-1])]

    # Calculate CDF value via formula dependent on bin
    res = p_cum[k] + probs[k]*(y - a[k])/(b[k] - a[k])

    # Special case: outside of range
    # res[y < a[0]] = 0
    # res[b[-1] < y] = 1
    res = res * (a[0] <= y)
    res = res * (y <= b[-1]) + (b[-1] < y)
    
    # Output
    return res

# Density of a histogram distribution
def pdf_hd(y, probs, bin_edges):
    """Function that calculates the PDF of a histogram distribution

    Parameters
    ----------
    y: n vector
        Observations
    probs: n_bins vectors
        Probabilities of corresponding bins
    bin_edges: (n_bins+1) vector
        Boundaries of bins
        
    Returns
    -------
    n vector
        PDF of forecasts at observations
    """

    #### Initiation ####
    # Left/right edges of bins
    a = bin_edges[:-1]
    b = bin_edges[1:]
    
    #### Calculation ####
    # Transform probabilities
    p_tilde = probs/(b - a)
    
    # Get transformed probability of corresponding bin
    res = p_tilde[np.searchsorted(b[:-1], y, side='right')]
    
    # Special cases: Outside of range
    # res[(y < a[0]) or (b[-1] < y)] = 0
    res = res * (a[0] <= y) * (y <= b[-1])

    # Output
    return res

# Quantile function of a histogram distribution
def quant_hd(tau, probs, bin_edges):
    """Function that calculates the PDF of a histogram distribution

    Parameters
    ----------
    tau: n vector
        Quantile levels
    probs: n_bins vectors
        Probabilities of corresponding bins
    bin_edges: (n_bins+1) vector
        Boundaries of bins
        
    Returns
    -------
    n vector
        Quantile function of forecasts at given levels
    """

    #### Initiation ####
    # Left/right edges of bins
    a = bin_edges[:-1]
    b = bin_edges[1:]

    #### Calculation ####
    # Calculate cumulative sums of bin probabilities
    # Entry k corresponds to sum of all k-1 bins below
    p_cum = np.r_[0, np.cumsum(probs)]

    # Drop after probability mass is 1
    p_cum = np.delete(p_cum, p_cum > 1-1e-5)
    
    # Calculate bin of quantiles (works for tau = 1)
    k = np.searchsorted(p_cum, tau, side='right') - 1 
    
    # Calculate quantiles via formula dependent on bin
    res = a[k] + (tau - p_cum[k])*(b[k] - a[k])/probs[k]

    # Output
    return np.minimum(np.max(b), res)

# CRPS of a histogram distribution
def crps_hd(y, f, bin_edges):
    """Function that calculates the CRPS of a histogram distribution

    Parameters
    ----------
    y: n vector
        Observations
    f: n x n_bins matrix OR n list of (different) n_bins vectors
        Probabilities of corresponding bins
    bin_edges: (n_bins+1) vector OR n x (n_bins+1) matrix OR n list of (different) (n_bins+1) vectors
        Boundaries of bins
        
    Returns
    -------
    n vector
        CRPS of forecasts and observations
    """

    #### Initiation ####
    # CRPS for a single forecast-observation pair
    def fn_single(obs, probs, bin_edges0):
        ## obs.........Observation (scalar)
        ## probs.......Bin probabilities (n_bins vector)
        ## bin_edges...Boundaries of bins ((n_bins + 1) vector)
        
        # Lower and upper edges of bins
        a = bin_edges0[:-1]
        b = bin_edges0[1:]
        
        # Realizing values for individual uniform distributions
        z = ((a <= obs) & (obs <= b))*obs + (obs < a)*a + (b < obs)*b
        if obs < a[0]:
            z[0] = obs
        if obs > b[-1]:
            z[-1] = obs
        
        # Lower and upper masses of individual uniform distributions
        L = np.cumsum(np.r_[0, probs[:-1]])
        U = 1 - np.cumsum(probs)

        # Transform from bin to unit interval ([a_i, b_i] -> [0, 1])
        w = (z - a)/(b - a)
        
        # Sum of standard uniform with lower mass L and upper mass U
        out = np.sum((b - a)*( np.abs(w - ss.uniform.cdf(w)) + 
                            (1 - L - U)*ss.uniform.cdf(w)**2 
                            - ss.uniform.cdf(w)*(1 - 2*L) 
                            + ((1 - L - U)**2)/3 + (1- L)*U ))
        
        # Output
        return out

    # Function for apply (identical bin_edges?)
    if isinstance(bin_edges, list):
        def fn_apply(i):
            return fn_single(obs=y[i], probs=f[i], bin_edges0=bin_edges[i])
    elif isinstance(bin_edges, np.ndarray):
        if np.ndim(bin_edges) == 2:
            def fn_apply(i):
                return fn_single(obs=y[i], probs=f[i,:], bin_edges0=bin_edges[i,:])
        elif np.ndim(bin_edges) == 1:
            def fn_apply(i):
                return fn_single(obs=y[i], probs=f[i,:], bin_edges0=bin_edges)

    #### Calculation ####
    # Apply function on all values
    res = [fn_apply(i) for i in range(len(y))]

    # Return
    return res

# Log-Score of a histogram distribution
def logs_hd(y, f, bin_edges):
    """Function that calculates the LogS of a histogram distribution

    Parameters
    ----------
    y: n vector
        Observations
    f: n x n_bins matrix OR n list of (different) n_bins vectors
        Probabilities of corresponding bins
    bin_edges: (n_bins+1) vector OR n x (n_bins+1) matrix OR n list of (different) (n_bins+1) vectors
        Boundaries of bins
        
    Returns
    -------
    n vector
        LogS of forecasts and observations
    """

    #### Initiation #### 
    # Function for apply (identical bin_edges?)
    if isinstance(bin_edges, list):
        def fn_apply(i):
            return -np.log(pdf_hd(y=y[i], probs=f[i], bin_edges=bin_edges[i]))
    elif isinstance(bin_edges, np.ndarray):
        if np.ndim(bin_edges) == 2:
            def fn_apply(i):
                return -np.log(pdf_hd(y=y[i], probs=f[i,:], bin_edges=bin_edges[i,:]))
        elif np.ndim(bin_edges) == 1:
            def fn_apply(i):
                return -np.log(pdf_hd(y=y[i], probs=f[i,:], bin_edges=bin_edges))

    #### Calculation ####
    # Apply function on all values
    res = [fn_apply(i) for i in range(len(y))]

    # Output
    return res

# PIT values of a histogram distribution
def pit_hd(y, f, bin_edges):
    """Function that calculates PIT values of a histogram distribution

    Parameters
    ----------
    y: n vector
        Observations
    f: n x n_bins matrix OR n list of (different) n_bins vectors
        Probabilities of corresponding bins
    bin_edges: (n_bins+1) vector OR n x (n_bins+1) matrix OR n list of (different) (n_bins+1) vectors
        Boundaries of bins

    Returns
    -------
    n vector
        PIT values of forecasts and observations
    """

    #### Initiation ####
    # Function for apply (identical bin_edges?)
    if isinstance(bin_edges, list):
        def fn_apply(i):
            return cdf_hd(y=y[i], probs=f[i], bin_edges=bin_edges[i])
    elif isinstance(bin_edges, np.ndarray):
        if np.ndim(bin_edges) == 2:
            def fn_apply(i):
                return cdf_hd(y=y[i], probs=f[i,:], bin_edges=bin_edges[i,:])
        elif np.ndim(bin_edges) == 1:
            def fn_apply(i):
                return cdf_hd(y=y[i], probs=f[i,:], bin_edges=bin_edges)

    #### Calculation ####
    # Plug in CDF
    res = [fn_apply(i) for i in range(len(y))]

    # Output (May be > 1 due to numerical reasons)
    return np.minimum(1, res)

# Prediction interval length of a histogram distribution
def lgt_hd(f, bin_edges, alpha = 0.1):
    """Function that calculates PIT values of a histogram distribution

    Parameters
    ----------
    y: n vector
        Observations
    f: n x n_bins matrix OR n list of (different) n_bins vectors
        Probabilities of corresponding bins
    bin_edges: (n_bins+1) vector OR n x (n_bins+1) matrix OR n list of (different) (n_bins+1) vectors
        Boundaries of bins
    alpha : probability
        Significance level for PI, by default 0.1

    Returns
    -------
    n vector
        Prediction interval lengths of forecasts
    """

    #### Calculation ####
    # Calculate via quantile function(identical bin_edges?)
    if isinstance(bin_edges, list):
        def fn_apply(i):
            return np.diff(quant_hd(tau=np.array([alpha/2, 1-alpha/2]), probs=f[i], bin_edges = bin_edges[i]))
    elif isinstance(bin_edges, np.ndarray):
        if np.ndim(bin_edges) == 2:
            def fn_apply(i):
                return np.diff(quant_hd(tau=np.array([alpha/2, 1-alpha/2]), probs=f[i,:], bin_edges=bin_edges[i,:]))
        elif np.ndim(bin_edges) == 1:
            def fn_apply(i):
                return np.diff(quant_hd(tau=np.array([alpha/2, 1-alpha/2]), probs=f[i,:], bin_edges=bin_edges))
    
    # Apply function
    res = [fn_apply(i) for i in range(len(f))]

    # Output
    return res

# Evaluation measures of a histogram distribution
def fn_scores_hd(
    f, 
    y, 
    bin_edges, 
    alpha = 0.1,
    skip_evals = None
) -> pd.DataFrame: #type: ignore
    """
    Scores a forecast based on the bin probabilities and edges.

    Parameters
    ----------
    f : n x n_bins matrix OR n list of (different) n_bins vectors
        Bin probabilities
    y : n vector
        Observations
    bin_edges : (n_bins+1) vector OR n x (n_bins+1) matrix OR n list of (different) (n_bins+1) vectors
        Boundaries of bins
    alpha : probability
        Significance level for PI, by default 0.1
    skip_evals : string vector
        Skip the following evaluation measures, Default: None -> Calculate
        all

    Returns
    -------
    scores_pp : n x 6 DataFrame
        DataFrame containing:
            pit : n vector
                PIT values of forecasts
            crps : n vector
                CRPS of forecasts
            logs : n vector
                Log-Score of forecasts
            lgt : n vector
                Length of prediction interval
            e_md : n vector
                Bias of median forecast
            e_me : n vector
                Bias of mean forecast
    """
    #### Initiation ####
    # Number of predictions
    n = len(y)

    #### Prediction and score calculation ####
    # Make data frame
    scores_pp = pd.DataFrame(
        index=range(n), columns=["pit", "crps", "logs", "lgt", "e_me", "e_md"]
    )

    # Skip evaluation measures
    if skip_evals is not None:
        scores_pp.drop(columns=skip_evals, inplace=True)

    # Calculate PIT values
    if "pit" in scores_pp.columns:
        scores_pp["pit"] = pit_hd(y = y,
                                  f = f,
                                  bin_edges = bin_edges)

    # Calculate CRPS
    if "crps" in scores_pp.columns:
        scores_pp["crps"] = crps_hd(y = y,
                                    f = f,
                                    bin_edges = bin_edges)

    # Calculate Log-Score
    if "logs" in scores_pp.columns:
        scores_pp["logs"] = logs_hd(y = y,
                                      f = f,
                                      bin_edges = bin_edges)

    # Calculate length of ~(n_ens-1)/(n_ens+1) % prediction interval
    if "lgt" in scores_pp.columns:
        scores_pp["lgt"] = lgt_hd(f = f,
                                  bin_edges = bin_edges,
                                  alpha = alpha)

    # Calculate bias of median forecast (identical bins?)
    if "e_md" in scores_pp.columns:
        if isinstance(bin_edges, list):
            def fn_apply(i):
                return quant_hd(tau=0.5, probs=f[i], bin_edges=bin_edges[i])
        elif isinstance(bin_edges, np.ndarray):
            if np.ndim(bin_edges) == 2:
                def fn_apply(i):
                    return quant_hd(tau=0.5, probs=f[i,:], bin_edges=bin_edges[i,:])
            elif np.ndim(bin_edges) == 1:
                def fn_apply(i):
                    return quant_hd(tau=0.5, probs=f[i,:], bin_edges=bin_edges)                

        scores_pp["e_md"] = [fn_apply(i) for i in range(len(y))] - y

    # Calculate bias of mean forecast (t(t(..)..) is column-wise multiplication) (identical bins?)
    if "e_me" in scores_pp.columns:
        if isinstance(bin_edges, list):
            def fn_apply(i):
                return np.sum(f[i]*np.diff(bin_edges[i]))/2
        elif isinstance(bin_edges, np.ndarray):
            if np.ndim(bin_edges) == 2:
                def fn_apply(i):
                    return np.sum(f[i,:]*np.diff(bin_edges[i,:]))/2
            elif np.ndim(bin_edges) == 1:
                def fn_apply(i):
                    return np.sum(f[i,:]*np.diff(bin_edges))/2   

        scores_pp["e_me"] = [fn_apply(i) for i in range(len(y))] - y

    # Return
    return scores_pp