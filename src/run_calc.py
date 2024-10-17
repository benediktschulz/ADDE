import json
import logging

import ss_1_ensemble
import ss_2_aggregation

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
        with open("src/config.json", "rt") as f:
            CONFIG = json.load(f)
        CONFIG["ENS_METHOD"] = ens_method
        with open("src/config.json", "wt") as f:
            json.dump(CONFIG, f)

        logging.log(msg=f"#### Running {ens_method} ####", level=25)

        # 1. Run ensemble prediction
        ss_1_ensemble.main()

        # 2. Run aggregation
        ss_2_aggregation.main()