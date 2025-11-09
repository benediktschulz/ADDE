import json
import sys
import logging

# import src.ss_5_aggregation_costs as ss_5_aggregation_costs
import ss_5_aggregation_times

### Set log Level ###
logging.basicConfig(
    format="%(asctime)s - %(message)s", 
    level=logging.INFO,
    stream=sys.stdout,  # ensure logs go to stdout
    force=True,         # overrides previous basicConfig calls (Python 3.8+)
)

if __name__ == "__main__":
    ss_5_aggregation_times.main()