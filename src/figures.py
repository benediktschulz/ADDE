## Script for evaluation of aggregation methods of NN methods

import json
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def plot_example_aggregation():
    ### Get CONFIG information ###
    (
        _,
        plot_path,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _get_config_info()

    ### Section 2: Example aggregation ###
    # Evaluations for plotting
    n_plot = 1_000

    # Aggregation methods to plot
    agg_meths_plot = ["lp", "vi", "vi-a", "vi-w", "vi-aw"]

    # Number of forecasts to aggregate
    n = 2

    # Lower and upper boundaries
    lower = 0.5
    upper = 13.5

    # Parameters of individual distributions
    mu_1 = 7
    mu_2 = 10
    sd_1 = 1
    sd_2 = 1

    # Data frame of distributions
    df_plot = pd.DataFrame()

    # Vector to plot on
    df_plot["y"] = np.linspace(start=lower, stop=upper, num=n_plot)

    # Calculate probabilities and densities of individual forecasts
    for temp in ["1", "2"]:
        df_plot[f"p_{temp}"] = ss.norm.cdf(
            x=df_plot["y"],
            loc=eval(f"mu_{temp}"),
            scale=eval(f"sd_{temp}"),
        )
        df_plot[f"d_{temp}"] = ss.norm.pdf(
            x=df_plot["y"],
            loc=eval(f"mu_{temp}"),
            scale=eval(f"sd_{temp}"),
        )

    # LP
    df_plot["p_lp"] = df_plot[["p_1", "p_2"]].mean(axis=1)
    df_plot["d_lp"] = df_plot[["d_1", "d_2"]].mean(axis=1)

    # For-Loop over Vincentization approaches
    for temp in [element for element in agg_meths_plot if "vi" in element]:
        # Intercept
        if temp in ["vi", "vi-w"]:
            a = 0
        else:  # temp in ["vi-a", "vi-aw"]:
            a = -6

        # Weights
        if temp in ["vi", "vi-a"]:
            w = 1 / n
        else:  # temp in ["vi-w", "vi-aw"]:
            w = 1 / n + 0.15

        # Calculate mean and sd
        mu_vi = a + w * sum([mu_1, mu_2])
        sd_vi = w * sum([sd_1, sd_2])

        # Calculate probabilities and densities
        df_plot[f"p_{temp}"] = ss.norm.cdf(
            x=df_plot["y"], loc=mu_vi, scale=sd_vi
        )

        df_plot[f"d_{temp}"] = ss.norm.pdf(
            x=df_plot["y"], loc=mu_vi, scale=sd_vi
        )

    # Name of PDF
    filename = os.path.join(plot_path, "aggregation_methods.pdf")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ## PDF
    # Empty plot
    axes[0].set_title("Probability density function (PDF)")
    axes[0].set_xlabel("y")
    axes[0].set_ylabel("f(y)")
    axes[0].set_ylim(bottom=0, top=max(df_plot[["d_1", "d_2"]].max()))
    axes[0].set_xlim(left=lower, right=upper)

    # Draw individual PDFs
    for i in range(1, n + 1):
        sns.lineplot(
            ax=axes[0],
            x=df_plot["y"],
            y=df_plot[f"d_{i}"],
            color=agg_col["ens"],
            linestyle=agg_lty["ens"],
        )

    # For-Loop over aggregation methods
    for temp in agg_meths_plot:
        sns.lineplot(
            ax=axes[0],
            x=df_plot["y"],
            y=df_plot[f"d_{temp}"],
            color=agg_col[temp],
            linestyle=agg_lty[temp],
        )

    ## CDF
    # Empty plot
    axes[1].set_title("Cumulative distribution function (CDF)")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("F(y)")
    axes[1].set_ylim(bottom=0, top=1)
    axes[1].set_xlim(left=lower, right=upper)

    # Draw individual CDFs
    for i in range(1, n + 1):
        sns.lineplot(
            ax=axes[1],
            x=df_plot["y"],
            y=df_plot[f"p_{i}"],
            color=agg_col["ens"],
            linestyle=agg_lty["ens"],
        )

    # For-Loop over aggregation methods
    for temp in agg_meths_plot:
        sns.lineplot(
            ax=axes[1],
            x=df_plot["y"],
            y=df_plot[f"p_{temp}"],
            color=agg_col[temp],
            linestyle=agg_lty[temp],
        )

    ## Quantile functions
    # Empty plot
    axes[2].set_title("Quantile function")
    axes[2].set_xlabel("p")
    axes[2].set_ylabel("Q(p)")
    axes[2].set_xlim(left=0, right=1)
    axes[2].set_ylim(bottom=lower, top=upper)

    # Draw individual quantile functions
    for i in range(1, n + 1):
        sns.lineplot(
            ax=axes[2],
            y=df_plot["y"],
            x=df_plot[f"p_{i}"],
            color=agg_col["ens"],
            linestyle=agg_lty["ens"],
        )

    # For-Loop over aggregation methods
    for temp in agg_meths_plot:
        sns.lineplot(
            ax=axes[2],
            y=df_plot["y"],
            x=df_plot[f"p_{temp}"],
            color=agg_col[temp],
            linestyle=agg_lty[temp],
        )

    # Add legend
    fig.legend()

    # Save fig
    fig.savefig(filename)
    log_message = f"Example aggregation saved to {filename}"
    logging.info(log_message)


def plot_panel_model():
    ### Get CONFIG information ###
    (
        ens_method,
        _,
        plot_path,
        data_path,
        data_ens_path,
        data_agg_path,
        dataset_ls,
        n_sim,
        n_ens,
        n_ens_vec,
        nn_vec,
    ) = _get_config_info()

    ### Simulation: Score panel ###
    # Vector of scores/quantiles to plot
    score_vec = ["crps", "crpss", "me", "lgt", "cov", "a", "w"]

    # For-Loop over data sets
    for dataset in dataset_ls:
        # Get number of repetitions
        if dataset in ["gusts", "protein", "year"]:
            temp_n_sim = 5
        else:
            temp_n_sim = n_sim
        
        ### Simulation: Load data ###
        filename = f"eval_{dataset}_{ens_method}.pkl"
        temp_data_path = data_path.replace("dataset", dataset)
        with open(os.path.join(temp_data_path, filename), "rb") as f:
            df_scores = pickle.load(f)

        # Check dataset is simulated or UCI Dataset
        optimal_scores_available = False

        ### Initialization ###
        # Only scenario
        df_sc = df_scores[df_scores["model"] == dataset]

        ### Calculate quantities ###
        df_plot = pd.DataFrame()

        # For-Loop over quantities of interest
        for temp_sr in score_vec:
            # Consider special case CRPSS
            if temp_sr == "crpss":
                temp_out = "crps"
            else:
                temp_out = temp_sr

            # Get optimal score of scenario for CRPSS
            s_opt = 0
            if optimal_scores_available and (temp_sr not in ["a", "w"]):
                s_opt = df_sc[df_sc["type"] == "ref"][temp_out].mean()

            # For-Loop over network variants
            for temp_nn in nn_vec:
                # Only network type
                df_nn = df_sc[df_sc["nn"] == temp_nn]

                # For-Lop over ensemble sizes and aggregation methods
                for i_ens in n_ens_vec:
                    for temp_agg in ["opt", "ens"] + agg_meths:
                        # Skip ensemble for skill
                        if (temp_sr == "crpss") and (temp_agg == "ens"):
                            continue
                        elif (temp_sr == "crpss") and (temp_agg == "opt"):
                            continue
                        elif (temp_sr == "a") and (temp_agg == "opt"):
                            continue
                        elif (temp_sr == "w") and (temp_agg == "opt"):
                            continue

                        # Fill in data frame
                        new_row = {
                            "nn": temp_nn,
                            "metric": temp_sr,
                            "n_ens": i_ens,
                            "agg": temp_agg,
                        }

                        # Reference: Average score of ensemble members
                        s_ref = df_nn[
                            (df_nn["n_rep"] < i_ens) # n_rep starts at 0!!
                            & (df_nn["type"] == "ind")
                        ][temp_out].mean()

                        # Special case: Average ensemble score
                        if temp_agg == "ens":
                            new_row["score"] = s_ref
                        elif temp_agg == "opt":
                            new_row["score"] = s_opt
                        else:
                            # Read out score
                            new_row["score"] = df_nn[
                                (df_nn["n_ens"] == i_ens)
                                & (df_nn["type"] == temp_agg)
                            ][temp_out].mean()

                            # Special case: CRPSS
                            if temp_sr == "crpss":
                                # Calcuate skill
                                new_row["score"] = (
                                    100
                                    * (s_ref - new_row["score"])
                                    / (s_ref - s_opt)
                                )

                            # Relative weight difference to equal weights in %
                            if temp_sr == "w":
                                new_row["score"] = 100 * (
                                    i_ens * new_row["score"] - 1
                                )

                        df_plot = pd.concat(
                            [
                                df_plot,
                                pd.DataFrame(new_row, index=[0]),
                            ],
                            ignore_index=True,
                        )

        ### Prepare data ###
        # Rename networks
        df_plot["nn"] = df_plot["nn"].str.upper()

        # Rename metrics
        score_labels = {
            "crps": "CRPS",
            "crpss": "CRPSS in %",
            "me": "Bias",
            "lgt": "PI length",
            "cov": "PI coverage in %",
            "a": "Intercept",
            "w": "Rel. weight diff. in %",
        }

        # Intercept values
        hline_vec0 = {
            # "crpss": 0,
            "me": 0,
            "cov": 90,
            "a": 0,
            "w": 0,
        }

        # Legend labels
        leg_labels = {"ens": "DE", "opt": r"$F^*$", **agg_abr}

        ### PDF ###
        fig, axes = plt.subplots(
            len(score_labels),
            len(nn_vec),
            figsize=(8.27, 11.69),
            squeeze=False,  # Always return 2d array even if only 1d
        )

        if not optimal_scores_available:
            df_plot = df_plot[df_plot["agg"] != "opt"]

        # For-Loop over networks
        for i, temp_nn in enumerate([x.upper() for x in nn_vec]):
            if any(
                (df_plot["nn"] == temp_nn)
                & (df_plot["metric"] == "crpss")
                & (df_plot["score"] <= 0)
            ):
                hline_vec = {"crpss": 0, **hline_vec0}
            else:
                hline_vec = hline_vec0

            y_label = True
            if i > 0:
                y_label = False
            # For-Loop over metrics
            for j, metric in enumerate(score_labels.keys()):
                sns.lineplot(
                    ax=axes[j][i],
                    data=df_plot[
                        (df_plot["nn"] == temp_nn)
                        & (df_plot["metric"] == metric)
                    ],
                    x="n_ens",
                    y="score",
                    hue="agg",
                    palette=agg_col,
                )
                if metric in hline_vec.keys():
                    axes[j][i].axhline(y=hline_vec[metric], linestyle="dashed")
                if y_label:
                    axes[j][i].set_ylabel(score_labels[metric])
                else:
                    axes[j][i].set_ylabel("")
                if j == 0:
                    axes[j][i].title.set_text(temp_nn)  # type: ignore

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            [leg_labels[label] for label in labels],
            loc="upper center",
            ncol=len(score_labels),
        )
        fig.suptitle(
            f"{ens_method} - Model {dataset}"
        )
        for ax in axes.flatten():  # type: ignore
            ax.legend().set_visible(False)

        # Save fig
        filename = os.path.join(plot_path, f"{dataset}_panel.pdf")
        fig.savefig(filename)
        log_message = f"Panel saved to {filename}"
        logging.info(log_message)
        plt.close(fig)


def plot_panel_boxplot():
    ### Get CONFIG information ###
    (
        ens_method,
        _,
        plot_path,
        data_path,
        data_ens_path,
        data_agg_path,
        dataset_ls,
        n_sim,
        n_ens,
        n_ens_vec,
        nn_vec,
    ) = _get_config_info()

    ### Simulation: CRPSS boxplot panel ###
    # For-Loop over scenarios
    for dataset in dataset_ls:
        ### Simulation: Load data ###
        filename = f"eval_{dataset}_{ens_method}.pkl"
        temp_data_path = data_path.replace("dataset", dataset)
        with open(os.path.join(temp_data_path, filename), "rb") as f:
            df_scores = pickle.load(f)

        ### Initialization ###
        # Only scenario
        df_sc = df_scores[df_scores["model"] == dataset]

        ### Calculate quantities ###
        df_plot = pd.DataFrame()

        # For-Loop over network variants, ensemble sizes and aggregation methods
        for temp_nn in nn_vec:
            for i_ens in n_ens_vec:
                for temp_agg in agg_meths:
                    # Get subset of scores data
                    df_sub = df_sc[
                        (df_sc["n_ens"] == i_ens)
                        & (df_sc["type"] == temp_agg)
                        & (df_sc["nn"] == temp_nn)
                    ]

                    new_row = {
                        "n_ens": i_ens,
                        "nn": temp_nn,
                        "agg": temp_agg,
                        "crpss": [list(100 * df_sub["crpss"])],
                    }

                    df_plot = pd.concat(
                        [df_plot, pd.DataFrame(new_row, index=[0])],
                        ignore_index=True,
                    )

        ### Prepare data ###
        # Rename networks
        df_plot["nn"] = df_plot["nn"].str.upper()

        # Get y-limits
        y_lim = df_plot.explode(column="crpss", ignore_index=True)["crpss"].agg(['min', 'max']) + (-0.1, 0.1)

        ### PDF ###
        # Make plot
        fig, axes = plt.subplots(
            len(nn_vec),
            len(agg_names.keys()),
            figsize=(15, 10),
            squeeze=False,
        )
        sample_size = len(df_plot["crpss"].iloc[0])
        fig.suptitle(
            f"{ens_method} - Model {dataset} - Sample size: {sample_size}"
        )

        # For-Loop over networks
        for i, temp_nn in enumerate([x.upper() for x in nn_vec]):
            # For-Loop over metrics
            for j, agg_method in enumerate(agg_names.keys()):
                # Create boxplots
                sns.boxplot(
                    data=df_plot[
                        (df_plot["nn"] == temp_nn)
                        & (df_plot["agg"] == agg_method)
                    ].explode(
                        column="crpss"  # type: ignore
                    ),
                    y="crpss",
                    x="n_ens",
                    color=agg_col[agg_method],
                    ax=axes[i][j],
                )

                # No skill
                axes[i][j].axhline(y=0, color="grey", linestyle="dashed")
                
                # y-limits
                axes[i][j].set(ylim=y_lim)
                
                # Method title
                if i == 0:
                    axes[i][j].set_title(agg_abr[agg_method])
                
                # x-label
                if i == (len(nn_vec)-1):
                    axes[i][j].set_xlabel("Ensemble Size")
                else:
                    axes[i][j].set_xlabel("")

                # y-label
                if j == 0:
                    axes[i][j].set_ylabel(f"{temp_nn}: CRPSS in %")
                else:
                    axes[i][j].set_ylabel("")

        # Save fig
        filename = os.path.join(plot_path, f"{dataset}_crpss_boxplots.pdf")
        fig.savefig(filename)
        log_message = f"CRPSS boxplots saved to {filename}"
        logging.info(log_message)
        plt.close(fig)


def plot_pit_ens():
    ### Get CONFIG information ###
    (
        ens_method,
        _,
        plot_path,
        data_path,
        data_ens_path,
        data_agg_path,
        dataset_ls,
        n_sim,
        n_ens,
        n_ens_vec,
        nn_vec,
    ) = _get_config_info()

    ### Simulation: PIT histograms ###
    # Network ensemble size
    n_ens = 10

    # Number of bins in histogram
    n_bins = 21

    # For-Loop over scenarios
    for dataset in dataset_ls:
        if dataset in ["gusts", "protein", "year"]:
            temp_n_sim = 5
        else:
            temp_n_sim = n_sim
        
        ### Simulation: Load data ###
        filename = f"eval_{dataset}_{ens_method}.pkl"
        temp_data_path = data_path.replace("dataset", dataset)
        with open(os.path.join(temp_data_path, filename), "rb") as f:
            df_scores = pickle.load(f)
        
        ### Initialization ###
        # Only scenario
        df_sc = df_scores[df_scores["model"] == dataset]

        ### Get PIT values ###
        # List for PIT values
        pit_ls = {}

        # For-Loop over network variants
        for temp_nn in nn_vec:
            ### PIT values of ensemble member ###
            # Vector for PIT values
            temp_pit = []

            # For-Loop over ensemble member and simulation
            for i_rep in range(n_ens):
                for i_sim in range(temp_n_sim):
                    # Load ensemble member
                    filename = (
                        f"{temp_nn}_sim_{i_sim}_ens_{i_rep}.pkl"  # noqa: E501
                    )
                    temp_data_ens_path = data_ens_path.replace(
                        "dataset", dataset
                    )
                    with open(
                        os.path.join(temp_data_ens_path, filename), "rb"
                    ) as f:
                        [pred_nn, y_valid, y_test] = pickle.load(f)
                    
                    # Check
                    if np.isnan(pred_nn["scores"]["pit"]).any():
                        log_message = f"NaN in PIT detected in {filename}"
                        logging.info(log_message)

                    # Index vector for validation and testing (pred_nn["n_test"] = n_valid + n_test)
                    i_test = range(pred_nn["n_valid"], pred_nn["n_test"])

                    # Read out
                    temp_pit.extend(pred_nn["scores"]["pit"][i_test])

            # Save PIT
            pit_ls[f"{temp_nn}_ens"] = temp_pit

            ### Aggregation methods ###
            # For-Loop over aggregation methods
            for temp_agg in agg_meths:
                # Vector for PIT values
                temp_pit = []

                # For-Loop over simulations
                for i_sim in range(temp_n_sim):
                    # Load aggregated forecasts
                    filename = f"{temp_nn}_sim_{i_sim}_{temp_agg}_ens_{n_ens}.pkl"  # noqa: E501
                    temp_data_agg_path = data_agg_path.replace(
                        "dataset", dataset
                    )
                    with open(
                        os.path.join(temp_data_agg_path, filename), "rb"
                    ) as f:
                        pred_agg = pickle.load(f)
                    
                    # Check
                    if np.isnan(pred_agg["scores"]["pit"]).any():
                        log_message = f"NaN in PIT detected in {filename}"
                        logging.info(log_message)

                    # Read out PIT-values
                    temp_pit.extend(pred_agg["scores"]["pit"])

                # Save PIT
                pit_ls[f"{temp_nn}_{temp_agg}"] = temp_pit

        ### Calculate histograms ###
        df_plot = pd.DataFrame()

        # For-Loop over network variants and aggregation methods
        for temp_nn in nn_vec:
            for temp_agg in ["ens", *agg_meths]:
                # Calculate histogram and read out values (see pit function)
                temp_hist, temp_bin_edges = np.histogram(
                    pit_ls[f"{temp_nn}_{temp_agg}"], bins=n_bins, density=True
                )

                new_row = {
                    "nn": temp_nn,
                    "agg": temp_agg,
                    "breaks": [temp_bin_edges],
                    "pit": [temp_hist],
                }

                df_plot = pd.concat(
                    [df_plot, pd.DataFrame(new_row, index=[0])],
                    ignore_index=True,
                )

        ### Prepare data ###
        # Rename networks
        df_plot["nn"] = df_plot["nn"].str.upper()

        ### PDF ###
        # Make plot
        fig, axes = plt.subplots(
            len(nn_vec),
            len(["ens", *agg_meths]),
            figsize=(15, 5),
            squeeze=False,
        )
        fig.suptitle(
            f"{ens_method} - Model {dataset}"
            f" (n_sim {temp_n_sim}, n_ens {n_ens})"
        )

        leg_labels = {"ens": "DE", "opt": r"$F^*$", **agg_abr}

        # For-Loop over networks
        for i, temp_nn in enumerate([x.upper() for x in nn_vec]):
            # Get a y-limit depending on network variant
            y_lim = max([pit for pitlist in df_plot[df_plot["nn"] == temp_nn]["pit"] for pit in pitlist]) + 0.1

            # For-Loop over metrics
            for j, metric in enumerate(["ens", *agg_meths]):
                df_curr = df_plot[
                    (df_plot["nn"] == temp_nn) & (df_plot["agg"] == metric)
                ]

                axes[i][j].bar(
                    x=df_curr["breaks"].iloc[0][:-1],
                    align = "edge",
                    height=df_curr["pit"].iloc[0],
                    width=np.diff(df_curr["breaks"].iloc[0]),
                    color="lightgrey",
                    edgecolor="black",
                )
                axes[i][j].axhline(y=1, linestyle="dashed", linewidth=1.5, color="black")
                axes[i][j].set_ylim(top = y_lim)

                if i == 0:
                    axes[i, j].title.set_text(leg_labels[metric])
                if j == 0:
                    axes[i][j].set_ylabel(temp_nn)

        # Save fig
        filename = os.path.join(plot_path, f"{dataset}_pit_ens.pdf")
        fig.savefig(filename)
        log_message = f"PIT saved to {filename}"
        logging.info(log_message)
        plt.close(fig)


def plot_ensemble_members():
    ### Get CONFIG information ###
    (
        ens_method,
        _,
        plot_path,
        data_path,
        data_ens_path,
        data_agg_path,
        dataset_ls,
        n_sim,
        n_ens,
        n_ens_vec,
        nn_vec,
    ) = _get_config_info()
    temp_n_sim = n_sim

    ### Plot CRPS and quantile function of ensemble members
    # Prepare quantile calculation
    # Size of quantile samples
    n_q_samples = 100
    # Quantile levels for evaluation
    q_levels = np.arange(
        start=1 / (n_q_samples + 1), stop=1, step=1 / (n_q_samples + 1)
    )

    df_plot = pd.DataFrame()
    idx = 0

    ### Collect ensemble member data ###
    for dataset in dataset_ls:
        if dataset in ["gusts", "protein", "year"]:
            temp_n_sim = 5
        else:
            temp_n_sim = n_sim
        for temp_nn in nn_vec:
            for i_sim in range(temp_n_sim):
                for i_ens in range(n_ens):
                    # Load ensemble member
                    filename = os.path.join(
                        f"{temp_nn}_sim_{i_sim}_ens_{i_ens}.pkl",  # noqa: E501
                    )
                    temp_data_ens_path = data_ens_path.replace(
                        "dataset", dataset
                    )
                    with open(
                        os.path.join(temp_data_ens_path, filename), "rb"
                    ) as f:
                        pred_nn, y_valid, y_test = pickle.load(
                            f
                        )  # pred_nn, y_valid, y_test

                    # Get indices of validation and test set
                    i_test = [x + len(y_valid) for x in range(len(y_test))]

                    new_row = {
                        "model": dataset,
                        "nn": temp_nn,
                        "n_sim": i_sim,
                        "n_ens": i_ens,
                        "idx": idx,
                        "agg": 0,
                    }
                    idx += 1

                    # Store q_levels and quantile samples in dataframe
                    new_row["x"] = q_levels
                    if temp_nn == "drn":
                        mu, std = np.mean(
                            pred_nn["f"][i_test,],
                            axis=0,
                        )
                        quantiles = ss.norm.ppf(q=q_levels, loc=mu, scale=std)
                        new_row["quantiles"] = quantiles
                    elif temp_nn == "bqn":
                        new_row["quantiles"] = np.mean(
                            pred_nn["f"][i_test, :], axis=0
                        )
                    elif temp_nn == "hen":
                        pass

                    df_plot = pd.concat(
                        [df_plot, pd.DataFrame([new_row], index=[0])],
                        ignore_index=True,
                    )

    ### Collect aggregated data ###
    for dataset in dataset_ls:
        if dataset in ["gusts", "protein", "year"]:
            temp_n_sim = 5
        else:
            temp_n_sim = n_sim
        for temp_nn in nn_vec:
            for i_sim in range(temp_n_sim):
                for i_ens in range(n_ens):
                    for temp_agg in agg_meths:
                        filename = f"{temp_nn}_sim_{i_sim}_{temp_agg}_ens_{n_ens}.pkl"  # noqa: E501
                        temp_data_agg_path = data_agg_path.replace(
                            "dataset", dataset
                        )
                        with open(
                            os.path.join(temp_data_agg_path, filename),
                            "rb",
                        ) as f:
                            pred_agg = pickle.load(f)

                        new_row = {
                            "model": dataset,
                            "nn": temp_nn,
                            "n_sim": i_sim,
                            "n_ens": i_ens,
                            "idx": idx,
                            "agg": temp_agg,
                        }
                        idx += 1

                        new_row["x"] = q_levels
                        if temp_nn == "drn":
                            if temp_agg == "lp":
                                new_row["quantiles"] = np.quantile(
                                    a=np.mean(pred_agg["f"], axis=0),
                                    q=q_levels,
                                )
                            else:
                                mu, std = np.mean(pred_agg["f"], axis=0)
                                quantiles = ss.norm.ppf(
                                    q=q_levels, loc=mu, scale=std
                                )
                                new_row["quantiles"] = quantiles
                        elif temp_nn == "bqn":
                            new_row["quantiles"] = np.mean(
                                pred_agg["f"], axis=0
                            )
                        elif temp_nn == "hen":
                            new_row["nn"] = "hen"

                        df_plot = pd.concat(
                            [df_plot, pd.DataFrame([new_row], index=[0])],
                            ignore_index=True,
                        )

    # Make plot
    fig, axes = plt.subplots(
        4,
        len(nn_vec),
        figsize=(15, 15),
        squeeze=False,
    )
    for dataset in dataset_ls:
        filename = f"eval_{dataset}_{ens_method}.pkl"
        temp_data_path = data_path.replace("dataset", dataset)
        with open(os.path.join(temp_data_path, filename), "rb") as f:
            df_scores = pickle.load(f)
        df_ens = df_scores[df_scores["type"] == "ind"]
        df_agg = df_scores[
            (df_scores["type"] != "ref") & (df_scores["type"] != "ind")
        ]
        for idx, temp_nn in enumerate(nn_vec):
            df_ens_temp = df_ens[
                (df_ens["nn"] == temp_nn) & (df_ens["model"] == dataset)
            ]
            df_agg_temp = df_agg[
                (df_agg["nn"] == temp_nn) & (df_agg["model"] == dataset)
            ]
            sns.boxplot(
                x=df_ens_temp["n_sim"],
                y=df_ens_temp["crps"],
                ax=axes[0][idx],
                palette="Dark2",
            )

            sns.boxplot(
                x=df_agg_temp["n_sim"],
                y=df_agg_temp["crps"],
                hue=df_agg_temp["type"],
                ax=axes[1][idx],
                palette="Dark2",
            )

            df_plot_temp = df_plot[
                (df_plot["nn"] == temp_nn) & (df_plot["agg"] == 0)
            ]
            sns.lineplot(
                data=df_plot_temp.explode(["quantiles", "x"]),  # type: ignore
                x="x",
                y="quantiles",
                hue="n_sim",
                ax=axes[2][idx],
                palette="Dark2",
            )

            df_plot_temp = df_plot[
                (df_plot["nn"] == temp_nn)
                & (df_plot["agg"] != 0)
                & (df_plot["agg"] != "lp")
            ]
            sns.lineplot(
                data=df_plot_temp.explode(["quantiles", "x"]),  # type: ignore
                x="x",
                y="quantiles",
                hue="agg",
                ax=axes[3][idx],
                palette="Dark2",
            )

            axes[0][idx].title.set_text(temp_nn.upper())  # type: ignore

        fig.suptitle(
            f"Model {dataset} - CRPS and predicted quantile functions"
            f" (n_sim {temp_n_sim}, n_ens {n_ens})"
        )

        # Save fig
        filename = os.path.join(plot_path, f"{dataset}_ensemble_members.pdf")
        fig.savefig(filename)
        log_message = f"Ensemble members saved to {filename}"
        logging.info(log_message)
        plt.close(fig)


def skill_tables():
    ### Get CONFIG information ###
    (
        ens_method,
        ens_method_ls,
        plot_path,
        data_path,
        data_ens_path,
        data_agg_path,
        dataset_ls,
        n_sim,
        n_ens,
        n_ens_vec,
        nn_vec,
    ) = _get_config_info()
    
    ### 
    # Calculate CRPSS
    temp_sr = "crps"

    ### Dependent on ensemble methods ###
    # Generate file
    tbl = open(f"plots/results/skill_table_ens_meths.txt", "w+")

    # Write table header 
    tbl.write("\\begin{tabular}{l*{7}{r}|*{6}{r}|*{5}{r}} \n")
    tbl.write("\\toprule \n")
    tbl.write("&& \\multicolumn{5}{c}{DRN} & \\multicolumn{1}{c}{} & \\multicolumn{5}{c}{BQN} & \\multicolumn{1}{c}{} & \\multicolumn{5}{c}{HEN} \\\\  \n")
    tbl.write("\\cmidrule{3-7} \\cmidrule{9-13} \\cmidrule{15-19}  \n")

    # 3x For-Loop over abbreviations
    for i in range(3):
        tbl.write("& \\multicolumn{1}{c}{} ")
        for temp_agg in agg_abr.keys():
            tbl.write(f"& {agg_abr[temp_agg]} ")
    tbl.write("\\\\ \n")

    # For-Loop over ensemble methods
    for ens_method in ens_method_ls:        
        # End line in table
        tbl.write("\\midrule \n")

        # Add line for ensemble method
        tbl.write(f"\multicolumn{{7}}{{l}}{{\\texttt{{{ens_names[ens_method]}}}}} & & & & & & & & & & & & \\\\ \n")

        # Replace ensemble method (required as data_path depends on last ens. method called)
        for temp_ens_method in ens_method_ls:
            data_path = data_path.replace(temp_ens_method, ens_method)
        
        # For-Loop over ensemble methods
        for dataset in dataset_ls:
            # Load data
            filename = f"eval_{dataset}_{ens_method}.pkl"
            temp_data_path = data_path.replace("dataset", dataset)
            with open(os.path.join(temp_data_path, filename), "rb") as f:
                df_scores = pickle.load(f)
                
            # Write dataset in file
            tbl.write(f"{set_names[dataset]} \n") 

            # For-Loop over network variants
            for temp_nn in nn_vec:
                # Start with empty column
                tbl.write(f" &")

                # Get reference score
                sc_ref = df_scores[
                            (df_scores["model"] == dataset)
                            & (df_scores["nn"] == temp_nn)
                            & (df_scores["type"] == "ind")
                        ][temp_sr].mean()
                
                # Calculate skill
                sc_skill = [(1 - df_scores[
                                (df_scores["model"] == dataset)
                                & (df_scores["nn"] == temp_nn)
                                & (df_scores["type"] == temp_agg)
                            ][temp_sr].mean()/sc_ref) for temp_agg in agg_abr.keys()]
                
                # Get best method
                agg_bold = list(agg_abr.keys())[np.argmax(sc_skill)]

                # Round skill to four decimals and multiply by 100
                sc_skill = ["{:.2f}".format(100*round(x, 4)) for x in sc_skill]

                # For-Loop over aggregation methods
                for i_agg in range(len(agg_abr)):    
                    # Best method in bold
                    if list(agg_abr.keys())[i_agg] == agg_bold:
                        tbl.write(f" & \\cellcolor{{lightgray}} \\textbf{{{sc_skill[i_agg]}}}")
                    else:                   
                        tbl.write(f" & {sc_skill[i_agg]}")
                        
                # End line in file
                tbl.write(f" \n")
                    
            # End line in table
            tbl.write(f" \\\\ \n")

    # End table
    tbl.write("\\bottomrule \n")
    tbl.write("\\end{tabular} \n")

    # Close file
    tbl.close()
    logging.info("Table Strategies done.")

    ### Dependent on datasets ###
    # Generate file
    tbl = open(f"plots/results/skill_table_data_sets.txt", "w+")

    # Write table header 
    tbl.write("\\begin{tabular}{l*{7}{r}|*{6}{r}|*{5}{r}} \n")
    tbl.write("\\toprule \n")
    tbl.write("&& \\multicolumn{5}{c}{DRN} & \\multicolumn{1}{c}{} & \\multicolumn{5}{c}{BQN} & \\multicolumn{1}{c}{} & \\multicolumn{5}{c}{HEN} \\\\  \n")
    tbl.write("\\cmidrule{3-7} \\cmidrule{9-13} \\cmidrule{15-19}  \n")

    # 3x For-Loop over abbreviations
    for i in range(3):
        tbl.write("& \\multicolumn{1}{c}{} ")
        for temp_agg in agg_abr.keys():
            tbl.write(f"& {agg_abr[temp_agg]} ")
    tbl.write("\\\\ \n")

    # For-Loop over data sets
    for dataset in dataset_ls:      
        # End line in table
        tbl.write("\\midrule \n")

        # Add line for data set
        tbl.write(f"\multicolumn{{7}}{{l}}{{\\texttt{{{set_names[dataset]}}}}} & & & & & & & & & & & & \\\\ \n")

        # For-Loop over ensemble methods
        for ens_method in ens_method_ls:  
            # Replace ensemble method (required as data_path depends on last ens. method called)
            for temp_ens_method in ens_method_ls:
                data_path = data_path.replace(temp_ens_method, ens_method)
        
            # Load data
            filename = f"eval_{dataset}_{ens_method}.pkl"
            temp_data_path = data_path.replace("dataset", dataset)
            with open(os.path.join(temp_data_path, filename), "rb") as f:
                df_scores = pickle.load(f)
                
            # Write ensembl method in file
            tbl.write(f"{ens_abbr[ens_method]} \n") 

            # For-Loop over network variants
            for temp_nn in nn_vec:
                # Start with empty column
                tbl.write(f" &")

                # Get reference score
                sc_ref = df_scores[
                            (df_scores["model"] == dataset)
                            & (df_scores["nn"] == temp_nn)
                            & (df_scores["type"] == "ind")
                        ][temp_sr].mean()
                
                # Calculate skill
                sc_skill = [(1 - df_scores[
                                (df_scores["model"] == dataset)
                                & (df_scores["nn"] == temp_nn)
                                & (df_scores["type"] == temp_agg)
                            ][temp_sr].mean()/sc_ref) for temp_agg in agg_abr.keys()]
                
                # Get best method
                agg_bold = list(agg_abr.keys())[np.argmax(sc_skill)]

                # Round skill to four decimals and multiply by 100
                sc_skill = ["{:.2f}".format(100*round(x, 4)) for x in sc_skill]

                # For-Loop over aggregation methods
                for i_agg in range(len(agg_abr)):    
                    # Best method in bold
                    if list(agg_abr.keys())[i_agg] == agg_bold:
                        tbl.write(f" & \\cellcolor{{lightgray}} \\textbf{{{sc_skill[i_agg]}}}")
                    else:                   
                        tbl.write(f" & {sc_skill[i_agg]}")
                        
                # End line in file
                tbl.write(f" \n")
                    
            # End line in table
            tbl.write(f" \\\\ \n")

    # End table
    tbl.write("\\bottomrule \n")
    tbl.write("\\end{tabular} \n")

    # Close file
    tbl.close()
    logging.info("Table Data Sets done.")


def _get_config_info():
    ### Get Config ###
    with open("src/config_eval.json", "rb") as f:
        CONFIG = json.load(f)
    ens_method = CONFIG["ENS_METHOD"]

    # Get available ensemble methods
    ens_method_ls = CONFIG["_available_ENS_METHOD"]
    
    # Path for figures
    plot_path = os.path.join(CONFIG["PATHS"]["PLOTS_DIR"], ens_method)

    # Path of data
    data_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
        "dataset",
        ens_method,
    )

    # Path of network ensemble data
    data_ens_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
        "dataset",
        ens_method,
        CONFIG["PATHS"]["ENSEMBLE_F"],
    )

    # Path of aggregated network data
    data_agg_path = os.path.join(
        CONFIG["PATHS"]["DATA_DIR"],
        CONFIG["PATHS"]["RESULTS_DIR"],
        "dataset",
        ens_method,
        CONFIG["PATHS"]["AGG_F"],
    )

    # Models considered
    dataset_ls = CONFIG["DATASET"]

    # Number of simulations
    n_sim = CONFIG["PARAMS"]["N_SIM"]

    # Ensemble size
    n_ens = CONFIG["PARAMS"]["N_ENS"]

    # Vector of ensemble members
    step_size = 2
    n_ens_vec = np.arange(
        start=step_size, stop=n_ens + step_size, step=step_size
    )

    # Network variants
    nn_vec = CONFIG["PARAMS"]["NN_VEC"]

    return (
        ens_method,
        ens_method_ls,
        plot_path,
        data_path,
        data_ens_path,
        data_agg_path,
        dataset_ls,
        n_sim,
        n_ens,
        n_ens_vec,
        nn_vec,
    )

### Initialize ###
# Vector for plotting on [0,1]
x_plot = np.arange(0, 1, 0.001)
x_plot50 = np.arange(0, 50, 0.01)

# Evaluation measures
sr_eval = ["crps", "me", "lgt", "cov"]

# Skill scores
sr_skill = ["crps"]

# Names of aggregation methods
agg_names = {
    "lp": "Linear Pool",
    "vi": "Vincentization",
    "vi-a": "Vincentization (a)",
    "vi-w": "Vincentization (w)",
    "vi-aw": "Vincentization (a, w)",
}
agg_abr = {
    "lp": r"LP",
    "vi": r"$V_0^=$",
    "vi-a": r"$V_a^=$",
    "vi-w": r"$V_0^w$",
    "vi-aw": r"$V_a^w$",
}

# Aggregation methods
agg_meths = list(agg_names.keys())

# Methods with coefficient estimation
coeff_meths = ["vi-a", "vi-w", "vi-aw"]

# Get colors
cols = sns.color_palette("Dark2", 8, as_cmap=True)

# Colors of aggregation methods
agg_col = {
    "lp": cols.colors[5],  # type: ignore
    "vi": cols.colors[4],  # type: ignore
    "vi-a": cols.colors[0],  # type: ignore
    "vi-w": cols.colors[2],  # type: ignore
    "vi-aw": cols.colors[3],  # type: ignore
    "ens": cols.colors[7],  # type: ignore
    "opt": cols.colors[3],  # type: ignore
}

# Line types of aggregation methods
agg_lty = {
    "lp": "dashed",
    "vi": "solid",
    "vi-a": "solid",
    "vi-w": "solid",
    "vi-aw": "solid",
    "ens": "dashdot",
    "opt": "dashdot",
}

# Names of datasets
set_names = {
    "gusts": "Wind",
    "scen_1": "Scenario 1",
    "scen_4": "Scenario 2",
    "protein": "Protein",
    "naval": "Naval",
    "power": "Power",
    "kin8nm": "Kin8nm",
    "wine": "Wine",
    "concrete": "Concrete",
    "energy": "Energy",
    "boston": "Boston",
    "yacht": "Yacht",
}

# Names of ensemble methods
ens_names = {
    "rand_init": "Random Initialization",
    "bagging": "Bagging",
    "batchensemble": "BatchEnsemble",
    "mc_dropout": "MC Dropout",
    "variational_dropout": "Variational Dropout",
    "concrete_dropout": "Concrete Dropout",
    "bayesian": "Bayesian",
}

# Abbrevaitions of ensemble methods
ens_abbr = {
    "rand_init": "Naive Ensemble",
    "bagging": "Bagging",
    "batchensemble": "BatchEnsemble",
    "mc_dropout": "MC Dropout",
    "variational_dropout": "Var.\\ Dropout",
    "concrete_dropout": "Con.\\ Dropout",
    "bayesian": "Bayesian",
}


if __name__ == "__main__":
    ### Call plot functions ###
    # plot_example_aggregation()
    plot_panel_model()
    plot_panel_boxplot()
    plot_pit_ens()
    plot_ensemble_members()
