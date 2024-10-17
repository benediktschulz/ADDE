import json
import logging

import hpar_summary
import hpar_eval
import hpar_tables

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

if __name__ == "__main__":
    #### More specifications ####
    methods = [
        "rand_init",
        "mc_dropout",
        "variational_dropout",
        "concrete_dropout",
        "bayesian",
    ]

    # Generate tables
    hpar_tables.main()

    # For-Loop over ensemble methods
    for ens_method in methods:
        ### Get Config ###
        with open("src/config_hpar_eval.json", "rt") as f:
            CONFIG = json.load(f)
        CONFIG["ENS_METHOD"] = ens_method
        with open("src/config_hpar_eval.json", "wt") as f:
            json.dump(CONFIG, f)

        logging.log(msg=f"#### Running {ens_method} ####", level=25)

        # 1. Summarize results
        hpar_summary.main()

        # 2. Evaluation figures
        hpar_eval.main()