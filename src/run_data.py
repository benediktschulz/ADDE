import json
import logging

import ss_0_data
import process_datasets

### Set log Level ###
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

if __name__ == "__main__":
    # UCI datasets
    process_datasets.main()
    
    # Data of simulations study
    ss_0_data.main()