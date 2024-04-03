import wandb
import time
from datetime import datetime
import sys
import json

# Store the original json.loads
original_json_loads = json.loads

def relaxed_json_loads(s, *args, **kwargs):
    kwargs.pop("strict", None)  # remove the strict argument if it exists
    return original_json_loads(s, strict=False, *args, **kwargs)

# Patch the library's json.loads
json.loads = relaxed_json_loads

# # for all data for one run
# api = wandb.Api()
# run = api.run("/opentensor-dev/openvalidators/runs/uqqz344j")

# # Fetch all keys from the run's history
# history_data = run.history()

# # Convert DataFrame to JSON
# new_data = json.loads(history_data.to_json(orient="records"))

# # Save updated data to the file
# with open("full_data_0806.json", "w") as file:
#     json.dump(new_data, file)



api = wandb.Api()
project_path = "opentensor-dev/openvalidators"

all_data = []

time_now = time.time()
start_timestamp = time_now - (60 * 60 * 3)  #last number hours ago

successful_runs = 0
error_count = 0
run_states = {"running": 0, "finished": 0, "failed": 0, "crashed": 0, "other": 0, "killed": 0}
corrected_keys = ['diversity_reward_model', 'reward', 'completions', 'prompt', 'rlhf_reward_model', 'reciprocate_reward_model', 'name', "neuron_hotkeys", "nsfw_filter", "uids", "task_validator_filter", "blacklist_filter"]

def clean_dict(d):
    """Recursively filter out None or empty values from a dictionary."""
    if not isinstance(d, dict):
        return d
    cleaned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            nested = clean_dict(value)
            if nested:  # Check if nested dictionary is not empty
                cleaned[key] = nested
        elif value not in [None, '', [], {}]:  # Exclude None, empty strings, empty lists, and empty dicts
            cleaned[key] = value
    return cleaned

try:
    for run in api.runs(project_path):
        run_id = run.id
        state = run.state
        netuid = run.config.get('netuid')
        created_timestamp = datetime.fromisoformat(run.created_at).timestamp()

        run_states[state] = run_states.get(state, 0) + 1

        if start_timestamp < created_timestamp and state == "running": 
            try:
                # Fetching the full history data of the run
                full_history_data = run.history(samples=5000)
                
                # Filtering the data based on corrected_keys
                history_data = [{key: entry.get(key) for key in corrected_keys} for entry in full_history_data.to_dict(orient="records")]

                if history_data:
                    all_data += history_data
                    data_size = sys.getsizeof(history_data) / (1024)  # Convert bytes to MB
                    print(f"Extracted {data_size:.2f} MB from run ID: {run_id}")
                else:
                    print(f"No data extracted for run ID: {run_id}. Skipping...")
                    continue
                
                    # # Extracting and cleaning run details
                # run_details = {
                #     "Created": run.created_at,
                #     "Tags": run.tags,
                #     "User": str(run.user),
                #     "Notes": run.notes,
                #     "Config": run.config
                # }
                # cleaned_details = clean_dict(run_details)
                # print(f"Run details: {cleaned_details}")
                successful_runs += 1

            except json.decoder.JSONDecodeError as e:
                print(f"JSONDecodeError for run ID {run_id}. Error: {str(e)}")
                error_count += 1
        else:
            print(f"Run {run_id} skipped based on criteria.")
            break

except Exception as e:
    print(f"Unexpected error occurred in the loop: {str(e)}")
    print(f"Problematic run ID: {run_id}")

# Outside loop operations
try:
    total_runs = len(api.runs(project_path))
    print(f"\nTotal number of runs in the project: {total_runs}")
    print(f"Number of successful data retrievals: {successful_runs}")
    print(f"Number of runs with errors: {error_count}")
    print("Run states:", run_states)

    # Save all data to a file
    with open("all_run_data_updated.json", "w") as file:
        json.dump(all_data, file, indent=2)
        print("json dump successful")

except Exception as e:
    print(f"Error while saving to file: {str(e)}")



