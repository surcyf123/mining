import requests

def test_flask_server():
    url = 'http://localhost:7500/process_paragraph' 
    paragraph = 'This is a sample paragraph for testing.' 
    response = requests.post(url, json={'paragraph': paragraph})

    if response.status_code == 200:
        print("Response from server:")
        print(response.json())
    else:
        print(f"Request failed with status code {response.status_code}")

if __name__ == "__main__":
    test_flask_server()
