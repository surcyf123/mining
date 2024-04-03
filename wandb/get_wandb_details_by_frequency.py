import wandb
import csv
import logging
from collections import defaultdict
import datetime
import json

# Store the original json.loads
original_json_loads = json.loads

def relaxed_json_loads(s, *args, **kwargs):
    kwargs.pop("strict", None)  # remove the strict argument if it exists
    return original_json_loads(s, strict=False, *args, **kwargs)

# Patch the library's json.loads
json.loads = relaxed_json_loads

logging.basicConfig(level=logging.INFO)
api = wandb.Api()
project_path = "opentensor-dev/openvalidators"

# Extract nested dictionary to a flat dictionary
def flatten(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

# for current runs
def gather_frequency_data(api, project_path):
    frequency_data = defaultdict(lambda: defaultdict(int))
    total_runs = 0
    processed_runs = 0
    
    logging.info("Starting data retrieval...")
    
    try:
        for run in api.runs(project_path):
            total_runs += 1
            
            run_details = {
                "Run ID": run.id,
                "Created": run.created_at,
                "Tags": "|".join(sorted(set(run.tags))) if run.tags else None,
                "User": str(run.user)
            }

            config_details = flatten(run.config)
            run_details = {**run_details, **config_details}
            run_details = {k: str(v) for k, v in run_details.items() if v not in [None, '']}  # Convert all values to strings

            for key, value in run_details.items():
                frequency_data[key][value] += 1

            processed_runs += 1
            if processed_runs % 100 == 0:
                logging.info(f"Processed {processed_runs}/{total_runs} runs...")

    except Exception as e:
        logging.error(f"Error occurred: {e}")

    logging.info("Sorting frequency data...")
    
    for key in frequency_data:
        frequency_data[key] = dict(sorted(frequency_data[key].items(), key=lambda item: item[1], reverse=True))
    
    logging.info("Writing data to CSV...")
    
    with open("wandb_details_by_frequency_current.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Detail Key", "Detail Value", "Frequency"])
        for key, inner_dict in frequency_data.items():
            for value, freq in inner_dict.items():
                writer.writerow([key, value, freq])

    logging.info("CSV generation complete!")

gather_frequency_data(api, project_path)
