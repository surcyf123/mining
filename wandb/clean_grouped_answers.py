import json

# Load your file
with open('grouped_answers0730.json', 'r') as f:
    data = json.load(f)

# Define the substrings to remove
substrings_to_remove = ['Answer:', 'Question:', 'Summary:', 'question:', 'answer:', 'summary']

# Iterate over every key and list in the dictionary
for key in data:
    for i in range(len(data[key])):
        # Iterate over every substring to remove
        for substring in substrings_to_remove:
            # If the substring is in the list item, remove it
            data[key][i] = data[key][i].replace(substring, '')

def replace_substrings(data, replacements):
    for key in data:
        for i in range(len(data[key])):
            for old, new in replacements.items():
                data[key][i] = data[key][i].replace(old, new)
    return data

# Define your replacement dictionary here. For example:
replacements = {
    'Answer:': 'Response:',
    'Question:': 'Query:',
    'Summary:': 'Overview:',
    'answer:': 'response:',
    'question:': 'query:',
    'summary:': 'overview:',
}

data = replace_substrings(data, replacements)

# Save the cleaned data back into the file
with open('grouped_answers0802.json', 'w') as f:
    json.dump(data, f)
