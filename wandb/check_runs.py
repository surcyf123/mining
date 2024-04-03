import json

# Store the original json.loads
original_json_loads = json.loads

def relaxed_json_loads(s, *args, **kwargs):
    kwargs.pop("strict", None)  # remove the strict argument if it exists
    return original_json_loads(s, strict=False, *args, **kwargs)

# Patch the library's json.loads
json.loads = relaxed_json_loads

# import wandb
# api = wandb.Api()

# project_path = "opentensor-dev/openvalidators"
# all_runs = []
# BATCH_SIZE = 500  # Adjust as needed
# all_run_details = []
# page_number = 1
# for frequency data for wandb runs
# try:
#     page = api.runs(project_path)
#     while page:
#         print(f"Processing page {page_number}... (Total runs in this page: {len(page)})")
        
#         for i in range(0, len(page), BATCH_SIZE):
#             batch = page[i:i+BATCH_SIZE]
#             all_runs.extend(batch)
#             print(f"Processed {i + len(batch)} runs out of {len(page)} in current page...")
        
#         page = page.next_page()
#         page_number += 1
# except json.decoder.JSONDecodeError:
#     print("Error while decoding JSON, skipping this batch.")

# total_runs = len(all_runs)
# print(f"Total number of runs processed: {total_runs}")


import wandb
import json

api = wandb.Api()

project_path = "opentensor-dev/openvalidators"
BATCH_SIZE = 500  # Adjust as needed
all_run_details = []


# for complete run data
try:
    runs = api.runs(project_path)  # Fetch all runs in the single page

    print(f"Processing runs... (Total runs: {len(runs)})")
    
    for i in range(0, len(runs), BATCH_SIZE):
        batch = runs[i:i+BATCH_SIZE]
        
        # Collecting run details for each run in the batch
        for run in batch:
            all_run_details.append(run._attrs)
            
        print(f"Processed {i + len(batch)} runs out of {len(runs)}...")

except json.decoder.JSONDecodeError:
    print("Error while decoding JSON, skipping this batch.")

# Save the collected run details to a JSON file
with open("all_run_details.json", "w") as f:
    json.dump(all_run_details, f)

print(f"Total number of runs processed: {len(all_run_details)}")