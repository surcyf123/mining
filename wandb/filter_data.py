import json

# Load your data
with open('full_data_0728.json', 'r') as f:
    data = json.load(f)

# Create a list to hold the new dictionaries
new_data = []

# Loop through your data to extract and rearrange the necessary information
for entry in data:
    # If "rlhf_reward_model" is present and not empty, reorder the values
    if "rlhf_reward_model" in entry and entry["rlhf_reward_model"]:
        # Get the sorted order of indices based on rlhf_reward_model
        sorted_order = sorted(range(len(entry["rlhf_reward_model"])), key=lambda k: entry["rlhf_reward_model"][k], reverse=True)

        # Reorder the lists based on this order
        entry["completions"] = [entry["completions"][i] for i in sorted_order]
        entry["rewards"] = [entry["rewards"][i] for i in sorted_order]
        entry["diversity_reward_model"] = [entry["diversity_reward_model"][i] for i in sorted_order]
        entry["reciprocate_reward_model"] = [entry["reciprocate_reward_model"][i] for i in sorted_order]
        entry["rlhf_reward_model"] = [entry["rlhf_reward_model"][i] for i in sorted_order] 
        entry["relevance_filter"] = [entry["relevance_filter"][i] for i in sorted_order]


    # Add the updated entry to the new_data list
    new_data.append(entry)

# Convert the new_data list of dictionaries into a new JSON file
with open('completions_data_sorted.json', 'w') as f:
    json.dump(new_data, f, indent=3)
