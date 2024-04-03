import requests

def upload_to_fileio(file_content):
    upload_url = "https://file.io"
    files = {'file': ('output_part.csv', file_content)}
    response = requests.post(upload_url, files=files)
    
    if response.status_code == 200:
        json_data = response.json()
        if json_data.get('success'):
            return json_data.get('link')
        else:
            return "Error uploading file: " + json_data.get('message')
    else:
        return "Error uploading file. Status code: " + str(response.status_code)

def get_last_fourth_of_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    one_fourth_length = len(lines) // 6
    return "".join(lines[-one_fourth_length:])

if __name__ == "__main__":
    file_path = "/home/fsuser/mining2/output0805.csv"
    last_fourth_content = get_last_fourth_of_file(file_path)
    download_link = upload_to_fileio(last_fourth_content)
    
    print(f"Download link: {download_link}")

