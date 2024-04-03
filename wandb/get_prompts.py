import json

# Load the data from the JSON file
with open("/home/fsuser/mining2/wandb/run_data_0805.json", "r") as file:
    data = json.load(file)

# Get the first 500 prompts
prompts = [dictionary['prompt'] for dictionary in data[:1000]]

# Save the prompts to a text file
with open("only_prompts.json", "w") as file:
    json.dump(prompts, file, indent=2)
