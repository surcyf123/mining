import json
import heapq

# Load the modified data from the JSON file
with open("/home/fsuser/mining2/wandb/run_data_0805.json", "r") as file:
    data = json.load(file)

# Get all completions along with their scores and names
# Get all completions along with their scores, names, and prompts, in the desired order
completions = [{'name': name, 'prompt': dictionary['prompt'], 'completions': completion, 'rewards': reward, 'rlhf_reward_model': rlhf, 'reciprocate_reward_model': reciprocate} for dictionary in data for completion, rlhf, reciprocate, reward, name in zip(dictionary['completions'], dictionary['rlhf_reward_model'], dictionary['reciprocate_reward_model'], dictionary['rewards'], [dictionary['name']]*50)]

# Get the top 100 completions based on 'rewards'
top_100_rewards = heapq.nlargest(100, completions, key=lambda x: x['rewards'])

# Get the top 100 completions based on 'rlhf_reward_model'
top_100_rlhf = heapq.nlargest(100, completions, key=lambda x: x['rlhf_reward_model'])

# Get the top 100 completions based on 'reciprocate_reward_model'
top_100_reciprocate = heapq.nlargest(100, completions, key=lambda x: x['reciprocate_reward_model'])

# Save the top completions to JSON files
with open("top_completions_rewards.json", "w") as file:
    json.dump(top_100_rewards, file, indent=3)

with open("top_completions_rlhf.json", "w") as file:
    json.dump(top_100_rlhf, file, indent=3)

with open("top_completions_reciprocate.json", "w") as file:
    json.dump(top_100_reciprocate, file, indent=3)

# Now, get the top 100 completions for each unique 'name'
unique_names = set(dictionary['name'] for dictionary in data)

for name in unique_names:
    # Filter the completions for the current name
    completions_for_name = [comp for comp in completions if comp['name'] == name]
    
    # Get the top 100 completions for the current name
    top_100_for_name = heapq.nlargest(100, completions_for_name, key=lambda x: x['rewards'])
    
    # Save the top 100 completions for the current name to a JSON file
    with open(f"top_completions_{name}.json", "w") as file:
        json.dump(top_100_for_name, file, indent=3)
