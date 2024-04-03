import requests

data = {
    "key1": "Your first paragraph here.",
    "key2": "Your second paragraph here."
}

response = requests.post('http://localhost:7500/process', json=data)
print(response.json())

