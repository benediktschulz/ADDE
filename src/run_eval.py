import json
import logging

import figures
import ss_3_scores
import ss_4_diversity

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

if __name__ == "__main__":
    #### More specifications ####
    methods = [
        "rand_init",
        "bagging",
        "batchensemble",
        "mc_dropout",
        "variational_dropout",
        "concrete_dropout",
        "bayesian",
    ]

    # Run multiple ensemble methods
    for ens_method in methods:
        ### Get Config ###
        with open("src/config_eval.json", "rt") as f:
            CONFIG = json.load(f)
        CONFIG["ENS_METHOD"] = ens_method
        with open("src/config_eval.json", "wt") as f:
            json.dump(CONFIG, f)

        logging.log(msg=f"#### Running {ens_method} ####", level=25)

        # Score results
        ss_3_scores.main()

        # Score results
        ss_4_diversity.main()

        # Plot results
        figures.plot_panel_model()
        figures.plot_panel_boxplot()
        figures.plot_pit_ens()
        # figures.plot_ensemble_members()

    # Generate skill table
    figures.skill_tables()
