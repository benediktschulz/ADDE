import json
import logging

import hpar_network

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
    # Run multiple ensemble methods
    for ens_method in methods:
        ### Get Config ###
        with open("src/config_hpar.json", "rt") as f:
            CONFIG = json.load(f)
        CONFIG["ENS_METHOD"] = ens_method
        with open("src/config_hpar.json", "wt") as f:
            json.dump(CONFIG, f)

        logging.log(msg=f"#### Running {ens_method} ####", level=25)

        # Run hyperparameter runs
        hpar_network.main()