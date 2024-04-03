import wandb
import csv
from collections import defaultdict
import json

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

# Create or overwrite the CSV file
with open('wandb_details.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    headers_written = False  # Flag to check if headers are already written

    # Try to iterate over runs
    try:
        for run in api.runs(project_path):
            run_id = run.id
            state = run.state
            netuid = run.config.get('netuid')

            # Extracting run details
            run_details = {
                "Run ID": run_id,
                "Created": run.created_at,
                "Tags": "|".join(sorted(set(run.tags))) if run.tags else None,
                "User": str(run.user)
            }

            # Flatten config
            config_details = flatten(run.config)

            # Merge details
            run_details = {**run_details, **config_details}

            # Filter out None or empty values
            run_details = {k: v for k, v in run_details.items() if v not in [None, '']}

            # Write to CSV
            if not headers_written:
                writer.writerow(run_details.keys())
                headers_written = True
            writer.writerow(run_details.values())

    except Exception as e:
        print(f"Unexpected error occurred while processing run {run_id}. Error: {str(e)}")
