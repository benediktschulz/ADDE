import json
import os

def main():
    # Get config file
    with open("src/config_hpar.json", "rt") as f:
        CONFIG = json.load(f)

    # General hyperparameters
    hpar_ls0 = ["layers", "actv", "n_batch", "lr_adam"]

    # Names of hyperparamters
    hpar_names = {
        "16": "16",
        "32": "32",
        "64": "64",
        "256": "256",
        "relu": "Relu",
        "softplus": "Soft",
        "-1": ".0010",
        "0.0005": ".0005",
        "[64, 32]": "2--\\phantom{{0}}64", 
        "[512, 256]": "2--512", 
        "[64, 64, 32]": "3--\\phantom{{0}}64", 
        "[512, 512, 256]": "3--512", 
        "[512, 512, 256, 128]": "4--512",
        "0.05": "5\\%",
        "0.1": "10\\%",
        "0.2": "20\\%",
        "0.5": "50\\%",
        "0.8": "80\\%",
        "uniform": "Unif",
        "standard_normal": "Norm",
        "laplace": "Lapl",
    }

    # Names of datasets
    set_names = {
        "gusts": "Gusts",
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

    # Names of datasets
    ens_names = {
        "rand_init": "Naive Ensemble / Bagging / BatchEnsemble",
        # "bagging": "Bagging",
        # "batchensemble": "BatchEnsemble",
        "mc_dropout": "MC Dropout",
        "variational_dropout": "Variational Dropout",
        "concrete_dropout": "Concrete Dropout",
        "bayesian": "Bayesian",
    }

    # Get list of datasets
    nn_vec = ["drn", "bqn", "hen"]

    # Get list of datasets
    dataset_ls = set_names.keys()

    # Generate file
    tbl = open(f"plots/results/hpars.txt", "w+")

    # Write table header 
    tbl.write("\\begin{tabular}{l*{7}{r}|*{7}{r}|*{6}{r}} \n")
    tbl.write("\\toprule \n")
    tbl.write("&& \\multicolumn{5}{c}{DRN} & \\multicolumn{1}{c}{} & \\multicolumn{6}{c}{BQN} & \\multicolumn{1}{c}{} & \\multicolumn{6}{c}{HEN} \\\\  \n")
    tbl.write("\\cmidrule{3-7} \\cmidrule{9-14} \\cmidrule{16-21}  \n")
    tbl.write("&& Arch & Actv & BA & LR & DR/PR & \\multicolumn{1}{c}{} & $d$ & Arch & Actv & BA & LR & DR/PR & \\multicolumn{1}{c}{} & $N$ & Arch & Actv & BA & LR & DR/PR \\\\ \n")

    # List of Ensembling methods
    ens_methods_tbl = ["rand_init", "mc_dropout", "variational_dropout", "concrete_dropout", "bayesian"]
    
    # For-Loop over ensemble methods
    for ens_method in ens_methods_tbl:        
        # End line in table
        tbl.write("\\midrule \n")

        # Add line for ensemble method
        tbl.write(f"\multicolumn{{7}}{{l}}{{\\texttt{{{ens_names[ens_method]}}}}} & & & & & & & & & & & & \\\\ \n")

        # Hyperparameter list depends on ensemble method
        if ens_method == "bayesian":
            hpar_ls = hpar_ls0 + ["prior"]
        elif ens_method == "mc_dropout":
            hpar_ls = hpar_ls0 + ["p_dropout"]
        else:
            hpar_ls = hpar_ls0

        # For-Loop over ensemble methods
        for dataset in dataset_ls:
            # Write dataset in file
            tbl.write(f"{set_names[dataset]} \n") 

            # For-Loop over network variants
            for temp_nn in nn_vec:
                # Path of hyperparameter files
                hpar_file = os.path.join(
                    CONFIG["PATHS"]["DATA_DIR"],
                    CONFIG["PATHS"]["RESULTS_DIR"],
                    dataset,
                    ens_method,
                    CONFIG["PATHS"]["HPAR_F"],
                    f"{temp_nn}_hpar_final.json"
                )

                # Load file
                with open(hpar_file, "rt") as f:
                    HPARS = json.load(f)["HPARS"]

                # Variant parameter for BQN and HEN
                if temp_nn == "bqn":
                    tbl.write(f" & & {HPARS['p_degree']} ") 
                elif temp_nn == "hen":
                    tbl.write(f" & & {HPARS['N_bins']} ")
                else:
                    tbl.write(f" &")

                # For-Loop over hyperparameters
                for temp_hpar in hpar_ls:                    
                    # Write in file
                    tbl.write(f" & {hpar_names[str(HPARS[temp_hpar])]}")
                
                # One column for no additional parameter
                if ens_method not in ["bayesian", "mc_dropout"]:
                    tbl.write(f" & -")
                        
                # End line in file
                tbl.write(f" \n")
                    
            # End line in table
            tbl.write(f" \\\\ \n")

    # End table
    tbl.write("\\bottomrule \n")
    tbl.write("\\end{tabular} \n")

    # Close file
    tbl.close()
    
if __name__ == "__main__":
    # np.seterr(all="raise")
    main()
