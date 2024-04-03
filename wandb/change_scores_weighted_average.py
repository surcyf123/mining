import json

# Load the data from the JSON file
with open("/home/fsuser/mining2/wandb/all_run_data_updated.json", "r") as file:
    data = json.load(file)

# Modify the data according to the requirements
for dictionary in data:
    rlhf_scores = dictionary['rlhf_reward_model']
    reciprocate_scores = dictionary['reciprocate_reward_model']
    current_rewards = dictionary.get('rewards', [0]*50)  # Defaults to a list of 50 zeroes if 'rewards' key does not exist

    # Calculate the new rewards
    new_rewards = [0.6 * rlhf + 0.4 * reciprocate if current != 0 else 0 for rlhf, reciprocate, current in zip(rlhf_scores, reciprocate_scores, current_rewards)]
    
    # Update the 'rewards' key in the dictionary
    dictionary['rewards'] = new_rewards

# Save the modified data back to the JSON file
with open("run_data_0805.json", "w") as file:
    json.dump(data, file, indent=4)
