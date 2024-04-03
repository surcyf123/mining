import os
import json
import re

def get_file_size(file_name):
    """Get the size of the file in bytes."""
    return os.path.getsize(file_name)

def standardize_prompt(prompt):
    """Standardize the prompt by converting numerical sentence counts to a placeholder 'X'."""
    return re.sub(r'in \d+ sentences', 'in X sentences', prompt)

def print_file_size(file_name):
    """Print the size of the given file in both MB and GB."""
    file_size_bytes = get_file_size(file_name)
    file_size_MB = file_size_bytes / (1024 * 1024)
    file_size_GB = file_size_bytes / (1024 * 1024 * 1024)
    print(f"Size of '{file_name}': {file_size_MB:.2f} MB ({file_size_GB:.2f} GB)")

def flatten_data(data):
    """Flatten the data structure to ease processing."""
    flattened = []
    for entry in data:
        prompt = entry.get('prompt', '')
        for reward, completion in zip(entry.get('rewards', []), entry.get('completions', [])):
            flattened.append({
                'prompt': prompt,
                'reward': reward,
                'completion': completion
            })
    return flattened

def main():
    # Check the file size of 'run_data_0728.json'
    print_file_size('all_run_data_updated.json')

    # Load and flatten data from json file
    with open('all_run_data_updated.json', 'r') as f:
        raw_data = json.load(f)
    flattened_data = flatten_data(raw_data)

    # Filter data based on reward score and sort
    high_score_data = [entry for entry in flattened_data if entry['reward'] > .82]
    sorted_data = sorted(high_score_data, key=lambda x: x['reward'], reverse=True)

    # Save the sorted data
    with open('top_answers_0729.json', 'w') as f:
        json.dump(sorted_data, f, indent=4)

    # Organize data by prompt type
    grouped_data = {}
    for entry in sorted_data:
        standardized = standardize_prompt(entry['prompt'])
        ending = standardized[-6:]
        grouped_data.setdefault(ending, []).append(entry)

    # Handle small groups (less than 10 entries)
    random_group = [entry for key, entries in grouped_data.items() if len(entries) < 10 for entry in entries]
    if random_group:
        grouped_data['random'] = random_group
        for key in list(grouped_data.keys()):  # Make a list of keys to avoid runtime error during iteration
            if len(grouped_data[key]) < 10:
                del grouped_data[key]

    # Extract unique completions prioritizing higher scores
    unique_completions = {}
    for entry in high_score_data:
        completion = entry['completion']
        reward = entry['reward']
        if completion not in unique_completions or unique_completions[completion]['reward'] < reward:
            unique_completions[completion] = entry

    high_score_data_unique = list(unique_completions.values())

    # Organize UNIQUE data by prompt type
    grouped_data_unique = {}
    for entry in high_score_data_unique:
        standardized = standardize_prompt(entry['prompt'])
        ending = standardized[-10:]
        grouped_data_unique.setdefault(ending, []).append(entry)

    # Handle small groups (less than 10 entries) using the UNIQUE data
    random_group = [entry for key, entries in grouped_data_unique.items() if len(entries) < 10 for entry in entries]
    if random_group:
        grouped_data_unique['random'] = random_group
        for key in list(grouped_data_unique.keys()):  # Make a list of keys to avoid runtime error during iteration
            if len(grouped_data_unique[key]) < 10:
                del grouped_data_unique[key]

    # Ensure all groups have the same size, based on the smallest group size
    min_size = min(len(entries) for entries in grouped_data_unique.values())
    for key in grouped_data_unique:
        grouped_data_unique[key] = grouped_data_unique[key][:min_size]

    # Print grouped data statistics using the UNIQUE data
    print("\nAll Groups (from unique data):")
    for group, entries in grouped_data_unique.items():
        print(f"{group} - {len(entries)} entries")

    # Save the grouped and unique completions
    with open('grouped_answers0729.json', 'w') as f:
        # if I want to see completion and score, ranked by order
        # json.dump({group: sorted([{'prompt': entry['prompt'], 'completion': entry['completion'], 'reward': entry['reward']} for entry in entries], key=lambda x: x['reward'], reverse=True) for group, entries in grouped_data_unique.items()}, f, indent=4)
        # this just makes a list of the answers
        json.dump({group: [entry['completion'] for entry in entries] for group, entries in grouped_data_unique.items()}, f, indent=4)

if __name__ == '__main__':
    main()
