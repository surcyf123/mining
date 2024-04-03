from flask import Flask, request, jsonify
from datastore_manager import DatastoreManager
from diversity_model import greedy_select_completions
from csv_logger import CSVLogger
from utils import send_discord_alert, task_validate, cleanup_prompt
import threading
import queue
import requests
import logging
import concurrent.futures
import time
from collections import deque
from queue import PriorityQueue
import openai

openai.api_key = ''
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class SuppressPostGetFilter(logging.Filter):
    def filter(self, record):
        return 'POST' not in record.getMessage() and 'GET' not in record.getMessage()

# Apply the filter to the 'werkzeug' logger
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(SuppressPostGetFilter())

scores_fields = ["Timestamp", "Model_Name", "Total_Reward", "Bert", "Mpnet", "RLHF", "Reciprocate", "DPO", "Type", "Completion"]
time_diffs_fields = ["Timestamp", "Prompt", "Time Difference"]
unique_prompts_fields = ["Timestamp", "Type", "Client_Identifier", "Question_Value", "Max_Time_Allowed", "Prompt"]
scores_logger = CSVLogger("scores_data0927.csv", scores_fields)
time_diffs_logger = CSVLogger("time_diffs.csv", time_diffs_fields)
unique_prompts_logger = CSVLogger("unique_prompts.csv", unique_prompts_fields)
app = Flask(__name__)
VERIFY_TOKEN = "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n"
datastore_lock = threading.Lock()
COMPLETION_ENDPOINTS = [
    # 'http://211.21.106.81:23391/generate', # 4090s models4x2p1 good
    # 'http://211.21.106.81:23508/generate',
    # 'http://211.21.106.81:23542/generate',
    # 'http://211.21.106.81:23470/generate',
    # 'http://172.218.204.83:2323/generate', # 4090s models4x2p2 good
    # 'http://172.218.204.83:2127/generate',
    # 'http://172.218.204.83:2116/generate',
    # 'http://172.218.204.83:2810/generate',
    # "http://46.246.68.45:20782/generate", # 3090s models8x2 good
    # "http://46.246.68.45:20724/generate",
    # "http://46.246.68.45:20419/generate",
    # "http://46.246.68.45:20901/generate",
    # "http://46.246.68.45:20791/generate",
    # "http://46.246.68.45:20976/generate",
    # "http://46.246.68.45:20623/generate",
    # "http://46.246.68.45:20366/generate",
    'http://50.173.30.254:21027/generate', #3090s models8x1 good
    'http://50.173.30.254:21088/generate',
    'http://50.173.30.254:21086/generate',
    'http://50.173.30.254:21051/generate',
    'http://50.173.30.254:21078/generate',
    'http://50.173.30.254:21090/generate',
    'http://50.173.30.254:21050/generate',
    'http://50.173.30.254:21075/generate',
    
]

REWARD_ENDPOINTS = [
    # numbers after GPU type is ranked by speed of that type
    # "http://47.189.79.46:50159", # 3090s1
    # "http://47.189.79.46:50108",
    # "http://47.189.79.46:50193",
    # "http://47.189.79.46:50060",
    'http://211.21.106.84:57414', # 3090s2
    'http://211.21.106.84:57515',
    'http://211.21.106.84:57298',
    'http://211.21.106.84:57445',
]

model_to_char = {
 'TheBloke/Huginn-13B-v4-AWQ': 'a',
 'TheBloke/UndiMix-v2-13B-AWQ': 'b',
 'TheBloke/Huginn-13B-v4.5-AWQ': 'c',
 'TheBloke/Huginn-v3-13B-AWQ': 'd',
 'TheBloke/Stheno-Inverted-L2-13B-AWQ': 'e',
 'TheBloke/Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-AWQ': 'f',
 'TheBloke/Speechless-Llama2-13B-AWQ': 'g',
 'TheBloke/Mythical-Destroyer-V2-L2-13B-AWQ': 'h',
 'TheBloke/Mythical-Destroyer-L2-13B-AWQ': 'i',
 'TheBloke/MythoBoros-13B-AWQ': 'j',
 'TheBloke/StableBeluga-13B-AWQ': 'k',
 'TheBloke/CodeUp-Llama-2-13B-Chat-HF-AWQ': 'l',
 'TheBloke/Baize-v2-13B-SuperHOT-8K-AWQ': 'm',
 'TheBloke/orca_mini_v3_13B-AWQ': 'n',
 'TheBloke/Chronoboros-Grad-L2-13B-AWQ': 'o',
 'TheBloke/Project-Baize-v2-13B-AWQ': 'p',
 'TheBloke/PuddleJumper-13B-AWQ': 'q',
 'TheBloke/Luban-13B-AWQ': 't',
 'TheBloke/LosslessMegaCoder-Llama2-13B-Mini-AWQ': 's',
 'TheBloke/OpenOrca-Platypus2-13B-AWQ': 'u',
 'TheBloke/Llama2-13B-MegaCode2-OASST-AWQ': 'v',
 'TheBloke/Chronos-Hermes-13B-SuperHOT-8K-AWQ': 'w',
 'TheBloke/OpenOrcaxOpenChat-Preview2-13B-AWQ': 'x',
 
 }


def create_queue_from_list(url_list):
    q = queue.Queue()
    for url in url_list:
        q.put(url)
    return q

class OrderedEndpointQueue:
    def __init__(self, endpoints):
        self.endpoints = endpoints
        self.queue = PriorityQueue()
        self.endpoint_set = set(endpoints)
        self.lock = threading.Lock()
        
        for i, endpoint in enumerate(endpoints):
            self.queue.put((i, endpoint))

    def get(self, block=True, timeout=None):
        with self.lock:
            if self.queue.empty():
                if block:
                    raise ValueError("Blocking not supported in this implementation.")
                    raise queue.Empty()

            while not self.queue.empty():
                _, endpoint = self.queue.get()
                if endpoint in self.endpoint_set:
                    self.endpoint_set.remove(endpoint)
                    return endpoint
            return None  # No available endpoints

    def put(self, endpoint):
        with self.lock:
            index = self.endpoints.index(endpoint)
            self.endpoint_set.add(endpoint)
            self.queue.put((index, endpoint))

    def empty(self):
        with self.lock:
            return self.queue.empty()

    def qsize(self):
        with self.lock:
            return self.queue.qsize()

reward_queue = OrderedEndpointQueue(REWARD_ENDPOINTS)
# reward_queue = create_queue_from_list(REWARD_ENDPOINTS)
completion_queue = create_queue_from_list(COMPLETION_ENDPOINTS)

def send_request(endpoint, data, endpoint_queue, queue_name):
    """Send a POST request to the given endpoint with provided data and log the outcome."""
    # logging.info(f"Sending question to {endpoint}")
    try:
        response = requests.post(endpoint, json=data, timeout=9.5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Endpoint {endpoint} returned {response.status_code}"}
    except requests.RequestException as e:
        logging.error(f"Request to {endpoint} failed: {str(e)}")
        return {"error": str(e)}
    finally:
        endpoint_queue.put(endpoint)  # Add the endpoint back into the queue

def get_scores_from_reward_model(original_prompt, response, endpoint):
    """Get scores from the reward model for the given response."""
    data = {
        "verify_token": VERIFY_TOKEN,
        "prompt": original_prompt,
        "completions": response
    }
    # logging.info(f"data_to_send = {data}")
    reward_response = send_request(endpoint, data, reward_queue, "reward_queue")
    if "error" in reward_response:
        logging.error(f"Request to {endpoint} failed! {reward_response}")
        return {"error": f"Failed to get data: {reward_response.get('error')}"}
    # logging.info(f"reward_response is :{reward_response}")
    return reward_response

def get_completion_or_default(question_type, prompt, datastore_manager):    
    # Ensure the prompt exists in the datastore
    if prompt not in datastore_manager.datastore:
        datastore_manager.datastore[prompt] = {}
    
    # Ensure the 'Completions List' key exists
    if 'Completions List' not in datastore_manager.datastore[prompt]:
        datastore_manager.datastore[prompt]['Completions List'] = []
    
    completions_list = datastore_manager.datastore[prompt]['Completions List']
    completions_left = len(completions_list)

    if not completions_left:
        logging.error(f"No completions found for prompt {datastore_manager.datastore[prompt]['Client Identifier'][:10]}")
        return None, None

    try:
        completion_data = completions_list.pop(0)  # get the first completion and remove it from the list
        cleaned_response = task_validate(question_type, completion_data['completion'])

        return cleaned_response, completion_data['score']
    except IndexError:
        logging.error("No completions available")

    return None, None

def get_completion(question_type, prompt, timeout, datastore_manager):
    time.sleep(timeout)
    with datastore_manager.datastore_lock:
        completion, score = get_completion_or_default(question_type, prompt, datastore_manager)
        if completion:
            return completion, score
    
    return "error in model generation", None

def fetch_and_score_completions(prompt, completions_needed, datastore_manager, timeout, question_type, device="cpu"):
    """Fetch and score completions for the given prompt with diversity penalty and logging."""
    
    # Ensure the prompt exists in the datastore
    if prompt not in datastore_manager.datastore:
        datastore_manager.datastore[prompt] = {}
    
    # Ensure the 'Completions List' key exists
    if 'Completions List' not in datastore_manager.datastore[prompt]:
        datastore_manager.datastore[prompt]['Completions List'] = []
    
    responses_needed = datastore_manager.datastore[prompt]['Count']
    completions_list = datastore_manager.datastore[prompt]['Completions List']
    best_completions = completions_list  # Initialize best_completions

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_and_score_single_completion, prompt, datastore_manager, timeout, question_type) 
                   for _ in range(min(datastore_manager.completion_queue.qsize()-1, completions_needed))]

        for future in concurrent.futures.as_completed(futures):
            scored_responses_list = future.result()
            if scored_responses_list:
                with datastore_manager.datastore_lock:
                    completions_list.extend(scored_responses_list)
    
    completions_received = len(completions_list)
    # logging.info(f"Fetched {completions_received} completions for prompt: {datastore_manager.datastore[prompt]['Client Identifier']}")

    if completions_received != 0 and responses_needed != 1:
        start_time = time.time()
        best_completions = greedy_select_completions(completions_list, responses_needed, device=device)
        datastore_manager.datastore[prompt]['Completions List'] = best_completions
        logging.info(f"Time taken to compute diversity: {round(time.time() - start_time, 3)} seconds")

    return {"responses": best_completions}

def fetch_and_score_single_completion(prompt, datastore_manager, timeout, question_type):
    start_time = time.time()
    
    if datastore_manager.completion_queue.empty():
        logging.error("completions queue empty!")
        return None

    # logging.info(f"Completion queue size before retrieval: {datastore_manager.completion_queue.qsize()}")
    completion_endpoint = datastore_manager.completion_queue.get(block=True, timeout=.5)
    # logging.info(f"URLs left in model queue: {datastore_manager.completion_queue.qsize()}")
    completion_data = {
        "prompt": prompt,
        "n": 16,
        "num_tokens": 160,
        "top_p": .95,
        "temperature": .95,
        "top_k": 1000,
    }
    completion = send_request(completion_endpoint, completion_data, datastore_manager.completion_queue, "completion_queue")
    checkpoint_time = time.time() - start_time
    # logging.info(f"speed is {round(completion['tokens_per_second'], 2)}t/s from {completion_endpoint} in {round(checkpoint_time, 2)}s")

    if not datastore_manager.reward_queue.empty() and "response" in completion:
        time_left = timeout - checkpoint_time - 1.5
        # logging.info(f"Reward queue size before retrieval: {datastore_manager.reward_queue.qsize()}")   
        reward_endpoint = datastore_manager.reward_queue.get(block=True, timeout=max(time_left, 0))
        logging.info(f"URLs left in reward queue: {datastore_manager.reward_queue.qsize()}, got {reward_endpoint} as the best endpoint")
        # Get scores for all completions
        scores_list = get_scores_from_reward_model(prompt, completion["response"], reward_endpoint)
        # logging.info(f"Prompt: {prompt}\n Completions: {completion['response']}")
        logging.info(f"received scores from {reward_endpoint} in {round(time.time() - start_time - checkpoint_time, 3)} seconds")
        data_to_return_list = []
        for i, (single_completion, scores) in enumerate(zip(completion["response"], scores_list)):
            if "error" not in scores:
                # logging.info(f"Scores is {scores}"
                # logging.info(f"single_completion is {single_completion}")
                scores_logger.log(
                    Model_Name=completion["model"],
                    Total_Reward=scores["Total Reward"],
                    Bert=scores["Bert"],
                    Mpnet=scores["MPNet"],
                    RLHF=scores["RLHF"],
                    Reciprocate=scores["Reciprocate"],
                    DPO=scores["DPO"],
                    Type=question_type,
                    # Completion=repr(single_completion),
                )

                data_to_return = {
                    "completion": single_completion + " " + model_to_char[completion["model"]],
                    "score": scores["Total Reward"],
                    "model": completion["model"]
                }
                # logging.info(f"data to return is {data_to_return}")
                data_to_return_list.append(data_to_return)

        if data_to_return_list:
            duration = time.time() - start_time
            # if duration > timeout:
                # logging.error(f"didn't fetch and score completions in time!")
            # logging.info(f"Total fetch and score took {duration} seconds")
            return data_to_return_list

    return None

def process_question(prompt, datastore_manager, completions_needed, timeout, client_identifier, question_type):
    try:
        scored_responses = fetch_and_score_completions(prompt, completions_needed, datastore_manager, timeout, question_type)
        sorted_responses = scored_responses.get('responses', [])
        # logging.info(f"Added {len(sorted_responses)} out of {completions_needed} completions to datastore for question from {client_identifier}")

        return {"responses": sorted_responses}
    except Exception as e:
        logging.error(f"Failed to get response from primary endpoint: {e}", exc_info=True)
        return {"error": "Failed to process question."}

def send_openai_request(prompt):
    start_time = time.time()
    try:
        # Format the messages parameter as a list of dictionaries
        messages = [{"role": "user", "content": prompt}]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=1.3,
            max_tokens=100,
            top_p=.95,
            n=1,
        )
        duration = round(time.time() - start_time, 4)
        answer = response["choices"][0]["message"]["content"].strip()
        logging.info(f"OpenAI API request took {duration} seconds: {answer}")
        return answer
    except Exception as e:
        logging.info(f"got exception when calling openai {e}")
        return "error calling model"


datastore_manager = DatastoreManager(completion_queue, reward_queue, process_question)
@app.route("/bittensor", methods=["POST"])
def chat():
    # logging.info("request received!")
    try:
        required_fields = ["prompt", "type", "client_identifier", "question_value"]
        request_data = request.get_json()
        # Authentication check
        if request_data.get("verify_token") != VERIFY_TOKEN:
            logging.info("invalid verify token of request")
            return jsonify({"error": "Invalid authentication token"}), 401
        # Check for required fields
        missing_fields = [field for field in required_fields if field not in request_data]
        if missing_fields:
            logging.info("request has missing fields")
            send_discord_alert("Request format changed!")
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        prompt = request_data["prompt"]
        completions_needed = request_data.get('completions_needed', 1)
        question_type = request_data["type"]
        client_identifier = request_data["client_identifier"]
        question_value = request_data["question_value"]
        timeout = request_data.get("max_time_allowed", 9.8) - .2
        if question_type == "unknown":
            unique_prompts_logger.log(
                Prompt=prompt.replace('\n', '\\n'),
                Type=question_type,
                Client_Identifier=client_identifier,
                Question_Value=question_value,
                Max_Time_Allowed=timeout
            )
            if client_identifier in ["5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3", "5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2", "5HNQURvmjjYhTSksi8Wfsw676b4owGwfLR2BFAQzG7H3HhYf"]:
                response = send_openai_request(prompt)
            else:
                response = "I'm sorry, I do not know the answer to your question. Can you please rephrase it so I can better assist you?"

            logging.info(f"unknown prompt answer: {response}")
            return response

        else:
            datastore_manager.update(prompt, client_identifier, question_value, question_type, timeout)
            datastore_manager.maybe_start_processing_thread(prompt, question_value, timeout, question_type) #starts process_question
            response, score = get_completion(question_type, prompt, timeout, datastore_manager)
            if score:
                logging.info(f"Response to {client_identifier[:10]}: {repr(response[:30])} | Score: {round(score, 3)}, value={round(question_value, 3)}")
            else:
                if datastore_manager.datastore[prompt]["Completions to Retrieve"] != 0:
                    logging.error(f"attempted to retrieve {datastore_manager.datastore[prompt]['Completions to Retrieve']} completions for {client_identifier[:10]}, but no completions available")
                else:
                    logging.info(f"giving no answer on purpose to {client_identifier[:10]}")

            cleanup_prompt(prompt, datastore_manager)
            return response
    except Exception:
        logging.exception("An error occurred while processing the chat request")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)