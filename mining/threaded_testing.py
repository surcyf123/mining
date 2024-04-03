import concurrent.futures
import logging
import threading
import csv
import time
import queue
import requests
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dpo_generation_urls = ["http://50.173.30.254:23076", "http://50.173.30.254:23057", "http://50.173.30.254:23007", "http://50.173.30.254:23004"]
dpo_scoring_urls = ["http://50.173.30.254:23050", "http://50.173.30.254:23084"]

with open("/home/fsuser/text_generation/dataset/only_prompts.json", "r") as f:
    prompts = json.load(f)

# Create a queue for scoring servers
scoring_servers = queue.Queue()
for url in dpo_scoring_urls:
    scoring_servers.put(url)

fieldnames = ['Prompt', 'Completion', 'Reward', 'Normalized Reward', 'Token Probabilities']
csv_filename = 'output0825_dpo.csv'
csv_lock = threading.Lock()
def write_to_csv(data):
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        with csv_lock:
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

def request_to_dpo(endpoint_url, messages):
    prompt = messages[0]
    try:
        start_time = time.time()
        response = requests.post(endpoint_url, json={"prompt": prompt})
        end_time = time.time()
        logging.info(f"Request to {endpoint_url} took {end_time - start_time:.2f} seconds")

        # Print the entire response from DPO
        logging.info(f"DPO Response: {response.text}")  # or use print(f"DPO Response: {response.text}")

        response.raise_for_status()  # Raises an exception if the request returned an error status code
        response_data = response.json()
        completions = response_data.get("completions", [])
        return completions
    except Exception as e:
        logging.error(f"Error making request to dpo_generator: {e}")
        return None


def score_answers(session, data, all_answers, url):
    logging.info("Scoring answers")
    start_time = time.time()
    req_data = {
        'prompt': data['prompt'],
        'responses': all_answers
    }

    try:
        response = session.post(url, json=req_data)
        end_time = time.time()
        logging.info(f"Scoring request took {end_time - start_time:.2f} seconds")

        response.raise_for_status()  # This will raise an error if the request fails
        response_json = response.json()
        logging.info(f"Scores received: {response_json['rewards']}")
        return response_json
    except Exception as e:
        logging.error(f"Error scoring answers: {e}")
        return {'rewards': None}

def generate_and_score(executor, session, data, writer, scoring_url, question_number):
    logging.info("Starting generate_and_score function.")

    tasks = [executor.submit(request_to_dpo, url, [data['prompt']]) for url in dpo_generation_urls]
    results = [future.result() for future in tasks]

    all_answers = [result[0] for result in results if result]

    logging.info("Scoring generated answers...")
    response_data = score_answers(session, data, all_answers, scoring_url)
    generated_scores = response_data.get('rewards', [])

    for answer, score in zip(all_answers, generated_scores):
        csv_data = {
            'Prompt': data['prompt'],
            'Completion': answer,
            'Reward': score,
            'Normalized Reward': '',  # Placeholder, adjust as needed
            'Token Probabilities': ''  # Placeholder, adjust as needed
        }
        writer(csv_data)

    # Return the scoring server back to the queue for reuse
    scoring_servers.put(scoring_url)

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    with requests.Session() as session:
        for i, prompt in enumerate(prompts):
            data = {'prompt': prompt}
            logging.info(f"Generating Answers for Prompt {i+1}...")
            try:
                scoring_url = scoring_servers.get(block=True, timeout=10)  # wait 10 seconds
                generate_and_score(executor, session, data, write_to_csv, scoring_url, i + 1)
            except queue.Empty:
                logging.error("No scoring server available after waiting for 10 seconds.")
                continue
