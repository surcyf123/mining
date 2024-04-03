import concurrent.futures
from concurrent.futures import wait, FIRST_COMPLETED
import requests
import openai
import json
import logging
import threading
import csv
import time
from collections import defaultdict, deque
import os
import random
from flask import Flask, jsonify, request
import traceback
import sys
import queue
from queue import Queue
import traceback
from cachetools import TTLCache
import pickle
from collections import deque
from contextlib import contextmanager
import inspect
from itertools import cycle


app = Flask(__name__)
DATASTORE_FILE = "datastore.pkl"
SAVE_INTERVAL = 10
stop_saving_event = threading.Event()

def save_datastore_periodically():
    while not stop_saving_event.is_set():
        save_datastore()
        time.sleep(SAVE_INTERVAL)  # Wait for the specified interval

def save_datastore():
    with open(DATASTORE_FILE, 'wb') as f:
        pickle.dump(datastore, f)

def load_datastore():
    try:
        with open(DATASTORE_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return {}  # Return an empty dictionary if there's an issue loading

datastore = load_datastore()

# Start the background thread right after creating the Flask application and loading the datastore
threading.Thread(target=save_datastore_periodically, daemon=True).start()

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logger = logging.getLogger('Sender')
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear existing handlers
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False  # Prevent logger from passing messages to the root logger
logger.info("Test message")


n = 50
min_question_length = 100
start_prompt = 3100
datastore_lock = threading.Lock()
known_keys = ['g context\n', 'r thoughts', 'ntences.\n\n', 'questions\n']
temperatures = [0.33]
last_written_time = {}
openai.api_key = ''
scoring_server_urls = [
    "http://213.173.102.136:10400", "http://213.173.102.136:10401", "http://213.173.102.136:10402",
    "http://213.173.99.35:18693","http://213.173.99.35:18694", "http://213.173.99.35:18695",
    "http://213.173.98.7:17703","http://213.173.98.7:17704", "http://213.173.98.7:17705",
    "http://213.173.102.150:19948", "http://213.173.102.150:19949", "http://213.173.102.150:19950",
    "http://213.173.98.204:11649", "http://213.173.98.204:11650","http://213.173.98.204:11651",
    "http://213.173.102.164:16239","http://213.173.102.164:16240","http://213.173.102.164:16241",
    "http://213.173.102.161:12765", "http://213.173.102.161:12766", "http://213.173.102.161:12767",
    # "http://213.173.98.84:26980", "http://213.173.98.84:26981", "http://213.173.98.84:26982",
    "http://64.247.206.99:43183", "http://64.247.206.99:43184", "http://64.247.206.99:43185", "http://64.247.206.99:43186", "http://64.247.206.99:43187",
]

verify_token_testing_validator = 'SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n'
verify_token = "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n" 
system_prompts = [
    "Be a helpful AI assitant. In case asked, you are a custom model trained with gpt4 data. Act like you are some super complex model that can do any task.",
    "Identify an important trend or development from the prompt and compose a question asking about the implications of this trend. Provide an answer discussing its implications, contributing factors, and potential future impact.",
    "From the prompt, identify a significant theory or concept within it. Formulate a question asking for a detailed explanation of this theory or concept. Provide an answer that discusses its principles, applications, and any research or discoveries related to it.",
    "Identify an important trend or future development within it relating to the prompt. Construct a relevant question discussing the potential implications of this trend. Provide an answer that discusses potential outcomes, contributing factors, and research aimed at exploring these implications.",
    "Consider a complex concept or a significant finding in an area relevant to the prompt. Formulate a question that asks about the methodologies used to validate these findings or the concept. Provide an answer detailing the different measures taken to ensure accuracy and reliability, including sample size, statistical tests, and comparison with previous studies.",
    "Pick a topic where a specific procedure or process is followed. Formulate a relevant question to the prompt that probes into the steps and considerations involved in this procedure or process. Provide an answer that discusses aspects such as the objectives, implementation steps, precautions taken, and expected outcomes.",
    "Provide a question about a random scientific or historical subject. Enumerate your answer clearly and exaplain the process for solving the question with the scientific method.",
    "provide a summary of the following information in 8 sentences",
    "Start by asking about the context or specifics of the topic. Then, provide a detailed, step-by-step tutorial on how to understand or approach it, focusing on the relevance and implications of each step.",
]

strategies = ['start', 'middle', 'end', 'full', 'r_thoughts', 'g_context', 'questions', 'summary', "summary2", 'start_and_end']

default_answer = "I deeply apologise. I was unable to process the response in time. HTTP///:::>>> Error code: H100/POD9/ID:271765 unavailable"

with open('/home/fsuser/mining2/wandb/run_data_0728.json', 'r') as f:
    dataset = json.load(f)

with open('/home/fsuser/mining2/wandb/grouped_answers0730.json', 'r') as f:
    all_grouped_predefined_answers = json.load(f)

scoring_servers = queue.Queue()
for url in scoring_server_urls:
    scoring_servers.put(url)

queue_lock = threading.Lock()

@contextmanager
def get_scoring_url(queue):
    url = queue.get(block=True, timeout=5)
    try:
        queue_info(inspect.currentframe().f_back.f_code.co_name, url)
        yield url
    finally:
        queue.put(url)
        queue_info(inspect.currentframe().f_back.f_code.co_name, url)

def queue_info(function, scoring_url):
    temp_list = []
    with queue_lock: 
        queue_size = scoring_servers.qsize()
        
        # Copy the queue without modifying its order
        for i in range(queue_size):
            item = scoring_servers.get()
            temp_list.append(item)
            scoring_servers.put(item)

        # logger.info(f"Queue Size: {queue_size}") #| Available URLs: {temp_list} | Function: {function} | Used URL: {scoring_url}")
        min_queue_size = 5
        if queue_size < min_queue_size:
            logger.info(f"Queue size is less than {min_queue_size}, it is {queue_size}")

csv_lock = threading.Lock()
fieldnames = ['In memory', 'Count', 'Error', 'Task', 'Random', 'Short', 'Unknown', 'Score', 'Relevance', 'RLHF', 'Reciprocate', 'Source', 'Index', 'Strategy', 'System Prompt', 'Predefined Scores', 'Openai Scores', 'Question', 'Answer', 'Hotkeys']
csv_filename = 'output0802.csv'
def write_to_csv(data):
    rounded_data = {}
    # Fetch the count from the datastore
    count = datastore.get(data.get('Question'), {}).get('count', 0)
    data['Count'] = count

    for k, v in data.items():
        if isinstance(v, float):
            rounded_data[k] = round(v, 5)
        else:
            rounded_data[k] = v

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(rounded_data)


def select_prompt_portion(prompt, strategy, portion_size=200):
    content_length = len(prompt[-1]['content'])

    if strategy == "start":
        prompt[-1]['content'] = prompt[-1]['content'][:200]
    elif strategy == "middle":
        start = max(0, content_length // 2 - portion_size // 2)
        prompt[-1]['content'] = prompt[-1]['content'][start:start+portion_size]
    elif strategy == "end":
        prompt[-1]['content'] = prompt[-1]['content'][-portion_size:]
    elif strategy == "full":
        pass
    elif strategy == "150-300":
        prompt[-1]['content'] = prompt[-1]['content'][150:350]
    elif strategy == "300-450":
        prompt[-1]['content'] = prompt[-1]['content'][-450:-250]
    elif strategy == "450-600":
        prompt[-1]['content'] = prompt[-1]['content'][450:650]
    elif strategy == "600-750":
        prompt[-1]['content'] = prompt[-1]['content'][-750:-550]
    elif strategy == "750-900":
        prompt[-1]['content'] = prompt[-1]['content'][750:950]
    elif strategy == "last250-50":
        prompt[-1]['content'] = prompt[-1]['content'][-250:-50]
    elif strategy == "900-1050":
        prompt[-1]['content'] = prompt[-1]['content'][900:1050]
    elif strategy == "first100-250_last100-250":
        prompt[-1]['content'] = prompt[-1]['content'][100:225] + prompt[-1]['content'][-225:-100]
    else:
        raise ValueError("Unknown strategy", strategy)

    return prompt



def request_to_openai(model_name, messages, temperature, max_tokens, top_p, system_prompt_index, strategy, frequency_penalty=1, presence_penalty=0.1):
    start_time = time.time()
    # logger.info(f"Calling OpenAI API with system prompt index {system_prompt_index}, temperature {temperature}, and strategy {strategy}")
    portion = select_prompt_portion(messages, strategy)
    # logger.info("portion is: ", portion)
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=portion,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=1,
            )
            duration = round(time.time() - start_time, 4)
            # logger.info(f"OpenAI API request took {duration} seconds")
            answers = [choice["message"]["content"].strip() for choice in response["choices"]]
            return (answers, system_prompt_index, temperature, strategy)
        except Exception as e:
            retries += 1
            logger.exception("Error making request to OpenAI API. Retrying in 1 second.", e)
            time.sleep(1)
    logger.error("Max retries reached. Failed to get response from OpenAI API.")
    return None


def score_answers(session, data, all_answers, url):
    start_time = time.time()
    # logger.info("Scoring answers")
    req_data = {
        'verify_token': 'SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n',
        'prompt': data[-1]['content'],
        'responses': all_answers
    }

    try:
        response = session.post(url, json=req_data)
        response.raise_for_status()  # This will raise an error if the request fails
        response_json = response.json()
        # logger.info(f"Scores received: {response_json['rewards']}")
        # logger.info(f"Reward details: {response_json['reward_details']}")
        # logger.info(f"Answer scoring took {round(time.time() - start_time, 4)} seconds!")
        return response_json
    except Exception as e:
        try:
            logger.exception(f"Error scoring answers: {e}. Response: {response.text}")
        except Exception:
            logger.exception("Error scoring answers, and could not log response text.")
        return {'rewards': None, 'reward_details': {}}

def get_predefined_answers_for_prompt(prompt, grouped_answers):
    # Extract the last 10 characters from the prompt
    key = prompt[-:]
    # logger.info("check key:", prompt[-10:], key)
    return grouped_answers.get(key, []), key

def score_predefined(session, data, predefined_answers_for_current_prompt, result_queue, predefined_event, timeout_event, scoring_url, question_number, predefined_finished_event, datastore):
    # logger.info("score_predefined started")
    start_time = time.time()
    if predefined_event.is_set():
        result_queue.put(None)
        return  # Exit if the event is set
    try:
        if timeout_event.is_set():
            return
        # Split predefined answers into chunks
        chunk_size = 8
        chunks = [predefined_answers_for_current_prompt[i:i + chunk_size] for i in range(0, len(predefined_answers_for_current_prompt), chunk_size)]
        total_elements = sum(len(chunk) for chunk in chunks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(score_answers, session, data, chunk, scoring_url)
                futures.append(future)

            # Process all futures and store results
            for future in futures:
                predefined_response_data = future.result()  # Get the results from each future as they complete
                # if question_number not in datastore:
                #     with datastore_lock:
                #         datastore[question_number] = {}
                    
                with datastore_lock:   
                    rewards = predefined_response_data.get('rewards')
                    if rewards is not None:
                        datastore.setdefault(question_number, {}).setdefault('predefined', {}).setdefault('rewards', []).extend(predefined_response_data['rewards'])
                        datastore[question_number]['predefined'].setdefault('answers', []).extend(predefined_answers_for_current_prompt)
                        datastore[question_number]['predefined'].setdefault('reward_details', {}).setdefault('rlhf_reward_model', []).extend(predefined_response_data['reward_details']['rlhf_reward_model'])
                        datastore[question_number]['predefined'].setdefault('reward_details', {}).setdefault('reciprocate_reward_model', []).extend(predefined_response_data['reward_details']['reciprocate_reward_model'])
                        datastore[question_number]['predefined'].setdefault('reward_details', {}).setdefault('relevance_filter', []).extend(predefined_response_data['reward_details']['relevance_filter'])
                    else:
                        logger.warning("No rewards found in predefined_response_data.")
    
        logger.info(f"Score_predefined finished in {round(time.time() - start_time, 4)} seconds for {total_elements} elements.")
    except Exception as e:
        logger.exception("An error occurred in generate_and_score function.")
    finally:
        predefined_finished_event.set()



def generate_and_score_openai(session, data, temperatures, system_prompt_indices, strategies, openai_event, scoring_url, datastore, question_number, openai_finished_event, max_tokens):
    # logger.info("Starting generate_and_score_openai")  
    start_time = time.time() 

    api_request_count = 0  # Counter for OpenAI API requests

    try:
        # Exit if the event is set
        if openai_event.is_set():
            logger.info("Exiting early due to openai_event being set")
            return  

        # Generate tasks for each combination of temperature, system prompt, and strategy
        tasks = []
        for temperature in temperatures:
            for system_prompt_index in system_prompt_indices:
                for strategy in strategies:
                    messages = [
                        {"role": "system", "content": system_prompts[system_prompt_index]},
                        {"role": "user", "content": data[-1]['content']}
                    ]
                    task = executor.submit(request_to_openai, 'gpt-3.5-turbo', messages, temperature, max_tokens, 0.8, system_prompt_index, strategy)
                    tasks.append(task)

        # Wait for all tasks to complete and gather results
        results = [future.result() for future in tasks]
        api_request_count += len(results)  # Update the count of API requests made
        all_answers = [result[0][0] for result in results]
        response_data = score_answers(session, data, all_answers, scoring_url)

        # Update datastore with scores and parameters
        with datastore_lock:
            # logger.info("adding openai data to datastore")
            # if question_number not in datastore:
            #     datastore[question_number] = {}

            datastore[question_number]['openai'] = {
                'rewards': response_data.get('rewards', [-1]),
                'parameters': [result[1:] for result in results],
                'answers': all_answers
            }
        # logger.info(f"after appending to datastore, info for openai is {datastore[question_number]['openai']}")
        total_time = round(time.time() - start_time, 4)
        logger.info(f"Finished generate_and_score_openai in {total_time} seconds for {api_request_count} OpenAI API requests.")

        # Notify other threads that this function is finished
        openai_finished_event.set()
    except Exception as e:
        logger.exception("An error occurred in generate_and_score_openai function.", exc_info=True)
        with datastore_lock:
            if question_number not in datastore:
                datastore[question_number] = {}
            datastore[question_number]['openai'] = {
                'rewards': [-1],  # Add a default reward in case of exception
                'parameters': [],
                'answers': all_answers
            }


def extract_best_data_from_datastore(question_number, datastore, timeout_event, used_key, data):
    start_time = time.time()
    best_score = None
    system_prompt = None
    strategy = None
    
    # Helper function to get reward details
    def get_reward_detail(source, detail_type, index):
        if source == "predefined":
            index = index % 40  # or whatever chunk size you're using
        reward_detail_dict = datastore.get(question_number, {}).get(source, {})
        if not reward_detail_dict:
            # logging.error(f"No reward details found for source {source}.")
            return None
        reward_detail_list = reward_detail_dict.get('reward_details', {}).get(detail_type, [])
        if index < len(reward_detail_list):
            return reward_detail_list[index]
        else:
            # logging.error(f"Index {index} out of range for reward details of type {detail_type} for source {source}.")
            return None

    
    # Helper function to get best data
    def get_best_data(source, answers, rewards, params=None):
        best_index = rewards.index(max(rewards))  # get the index of the best reward
        best_answer = answers[best_index]  # get the best answer from the list
        best_score = rewards[best_index]  # get the best score from the list
        if params:
            openai_params = params[best_index]  # get the corresponding parameters
            return best_answer, best_score, openai_params
        return best_answer, best_score, None  # return None if there are no parameters



    combined_scores, all_predefined_data, all_openai_data = [], [], []
    source_data_map = {
        'predefined': all_predefined_data,
        'openai': all_openai_data
    }

    for source, details_list in source_data_map.items():
        if source in datastore[question_number]:
            rewards = datastore[question_number][source].get("rewards", [0])
            answers = deque(datastore[question_number][source]["answers"])
            if not rewards or not answers:
                logger.error(f"No rewards or answers found for source {source}")
                continue
            for idx, score in enumerate(rewards):
                identifier = None
                if source == "openai":
                    identifier = datastore[question_number]['openai']['parameters'][idx]
                details_list.append({
                    "score": score,
                    "rlhf_reward_model": get_reward_detail(source, 'rlhf_reward_model', idx),
                    "reciprocate_reward_model": get_reward_detail(source, 'reciprocate_reward_model', idx),
                    "relevance_filter": get_reward_detail(source, 'relevance_filter', idx),
                    "Index": idx if source == "predefined" else None,
                    "Identifier": identifier if source == "openai" else None
                })

    source_data = {
            'openai': datastore[question_number].get('openai', {}),
            'predefined': datastore[question_number].get('predefined', {}),
        }

    best_data_dict = {}
    for source, data_dict in source_data.items():
        if 'rewards' in data_dict:
            try:
                best_answer, best_score, params = get_best_data(source, data_dict['answers'], data_dict['rewards'], data_dict.get('parameters'))
                if not best_data_dict or best_score > best_data_dict['Score']:
                    best_data_dict = {
                        'Question': replace_newlines(data[-1]['content']),
                        'Task': replace_newlines(used_key),
                        'Short': len(data[-1]['content']) < min_question_length,
                        'Score': best_score,
                        'RLHF': get_reward_detail(source, 'rlhf_reward_model', 0),
                        'Reciprocate': get_reward_detail(source, 'reciprocate_reward_model', 0),
                        'Relevance': get_reward_detail(source, 'relevance_filter', 0),
                        'Random': False,
                        'Source': source,
                        'Index': None if source == "openai" else 0,
                        'System Prompt': params[0] if params else None,
                        'Strategy': params[2] if params else None,
                        'Predefined Scores': all_predefined_data,
                        'Openai Scores': all_openai_data,
                        'Answer': replace_newlines(best_answer),
                        # 'Hotkeys': ' '.join(datastore[question_number]['Hotkeys'])
                    }
            except Exception as e:
                logger.exception(f"An error occurred while processing {source} data")

    if not best_data_dict:
        best_data_dict = {
            'Question': replace_newlines(data[-1]['content']),
            'Task': replace_newlines(used_key),
            'Short': len(data[-1]['content']) < min_question_length,
            'Score': 0,
            'Answer': default_answer,
            'Random': True,
            'Source': 'default',
            'RLHF': 0,
            'Reciprocate': 0,
            'Relevance': 0,
            'Index': None,
            'System Prompt': None,
            'Strategy': None,
            'Predefined Scores': all_predefined_data,
            'Openai Scores': all_openai_data,
            # 'Hotkeys': ' '.join(datastore[question_number]['Hotkeys'])
        }
        logger.info("No answers scored in time!")

    return best_data_dict



start_time = time.time()
def replace_newlines(string):
    return string.replace('\r\n', '\\n').replace('\n', '\\n')


def handle_short_question(data, system_prompts, temperatures, system_prompt_indices, strategies, question_number, used_key):
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": system_prompts[0]},
        {"role": "user", "content": data[-1]['content']}
    ]
    result = request_to_openai('gpt-3.5-turbo', messages, 1.5, 140, 0.8, system_prompt_indices[0], strategies[3] if len(data[-1]['content']) < 1000 else strategies[9])

    short_data = {
        'Question': replace_newlines(data[-1]['content']),
        'Short': True if len(data[-1]['content']) < min_question_length else False,
        'Unknown': True if used_key not in known_keys else False,
        'Answer': replace_newlines(result[0][0])
    }
    
    with csv_lock:
        write_to_csv(short_data)    
    
    # Add the answer to the datastore
    with datastore_lock:
        if question_number not in datastore:
            datastore[question_number] = {}
        datastore[question_number]['Answer'] = [short_data['Answer']]
        datastore[question_number]['rewards'] = [2]  # Add a default reward of 2
        datastore[question_number]['status'] = 'done'  # Set the status to done for this question


    logger.info(f"Short or Unknown Question took {round(time.time() - start_time, 4)} seconds!")
    return [short_data['Answer']]



strategies = ['start', 'middle', 'end', 'full', '150-300', "300-450", "450-600", "600-750", "750-900", 'first100-250_last100-250', 'last250-50']
new_endings_to_keys = {
    "and insightful question about the preceding context\n. Do not try to return an answer or a summary:": "g context\n",
    "and insightful question about the preceding context and previous questions. Do not try to return an answer or a summary:\n": "g context\n",
    "\nAnswer the question step by step and explain your thoughts. Do not include questions or summaries in your answer." : "r thoughts",
    "sentences. Do not try to create questions or answers for your summarization.\n\n": "ntences.\n\n"
}

def handle_key(strategies, data, system_prompt_indices, stake_amt):
    strategies_without_full = [strategy for strategy in strategies if strategy != 'full']
    # Determine the used key based on the last part
    used_key = None
    for ending, key in new_endings_to_keys.items():
        if data[-1]['content'].endswith(ending):
            used_key = key
            break

    if used_key is None:
        return None

    if stake_amt > 300000:
        if used_key == "r thoughts":
            return [1, 2, 3], [strategies[9], strategies[6], strategies[6], strategies[8]], 180
        elif used_key == "g context\n":
            return [5, 4, 6], [strategies[8], strategies[4]], 225
        elif used_key == "questions\n":
            return [8], [strategies[1], strategies[8]], 170
        elif used_key == "ntences.\n\n":
            return [7], [strategies[9], strategies[1], strategies[0], strategies[8], strategies[7], strategies[6], strategies[5], strategies[4], strategies[10], strategies[11]], 140
        else:
            return None

    elif stake_amt > 70000:
        if used_key == "r thoughts":
            return [1, 2], [strategies[9], strategies[6]], 160
        elif used_key == "g context\n":
            return [5, 4], [strategies[8], strategies[4]], 200
        elif used_key == "questions\n":
            return [8], [strategies[1], strategies[8]], 150
        elif used_key == "ntences.\n\n":
            return [7], [strategies[9], strategies[1], strategies[0], strategies[8], strategies[7], strategies[6]], 125
        else:
            logger.error("used_key not found")
            return None

    elif stake_amt > 20000:
        if used_key == "r thoughts":
            return [1, 2], [strategies[9], strategies[6]], 140
        elif used_key == "g context\n":
            return [5, 4], [strategies[8], strategies[4]], 175
        elif used_key == "questions\n":
            return [8], [strategies[1], strategies[8]], 130
        elif used_key == "ntences.\n\n":
            return [7], [strategies[9], strategies[1], strategies[0], strategies[8], strategies[7]], 115
        else:
            logger.error("used_key not found")
            return None
    else:
        logger.error("key or stake not found")


def update_datastore_status_and_answer(question_number, status, answer):
    with datastore_lock:
        datastore[question_number]['status'] = status
        datastore[question_number]['Answer'] = [answer]

def legit_answer(executor, session, data, temperatures, system_prompt_indices, strategies, scoring_url, question_number, scoring_servers):
    best_data = {}
    global min_question_length, system_prompts
    timeout_event = threading.Event()
    timer = threading.Timer(9.5, timeout_event.set)
    timer.start()
    start_time = time.time()
    result_queue = Queue()
    openai_event = threading.Event()
    openai_finished_event = threading.Event()
    stake_amt = datastore[question_number]['stake_amt']
    key_result = handle_key(used_key, strategies, data, system_prompt_indices) 
    system_prompt_indices, strategies, max_tokens = key_result
    openai_future = executor.submit(generate_and_score_openai, session, data, temperatures, system_prompt_indices, strategies, openai_event, scoring_url, datastore, question_number, openai_finished_event, max_tokens)

    if timeout_event.is_set():
        logger.error(f"Thread for question type {used_key} timed out.")
        return

    done, not_done = wait([predefined_future, openai_future], timeout=9.5) 
    if not done:
        logger.warning(f"Thread for question type {used_key} timed out. Redundant!")
    else:
        best_data = extract_best_data_from_datastore(question_number, datastore, timeout_event, used_key, data)
        if predefined_future in not_done:
            best_data['Error'] = "Only OpenAI data was available for scoring."
        elif openai_future in not_done:
            best_data['Error'] = "Only predefined data was available for scoring."

        # Sort and update datastore
        for source in ["predefined", "openai"]:
            if source in datastore[question_number]:
                rewards = datastore[question_number][source]["rewards"]
                answers = datastore[question_number][source]["answers"] 

                sorted_rewards, sorted_answers = zip(*sorted(zip(rewards, answers), reverse=True))

                # Updating the sorted rewards and answers in the datastore
                datastore[question_number][source]["rewards"] = list(sorted_rewards)
                datastore[question_number][source]["answers"] = deque(sorted_answers)

        # Combine the answers and rewards from both sources
        combined_rewards = datastore[question_number].get("predefined", {}).get("rewards", []) + datastore[question_number].get("openai", {}).get("rewards", [])
        combined_answers = datastore[question_number].get("predefined", {}).get("answers", deque([])) + datastore[question_number].get("openai", {}).get("answers", deque([]))

        # Combine rewards and answers into tuples
        combined_answers_rewards = list(zip(combined_answers, combined_rewards))

        # Sort all answers based on the combined rewards
        sorted_combined_answers_rewards = sorted(combined_answers_rewards, key=lambda x: x[1], reverse=True)

        # Filter out answers with reward 0 or less
        filtered_answers_rewards = [(answer, reward) for answer, reward in sorted_combined_answers_rewards if reward > 0]

        # Only create a cycle if there are answers
        if filtered_answers_rewards:
            # Create a cycle from the filtered and sorted answers
            answers_cycle = cycle(filtered_answers_rewards)
            # Update the datastore with the answers cycle
            datastore[question_number]["all_answers"] = answers_cycle
        else:
            logger.error(f"No answers with reward > 0 for question: {replace_newlines(question_number[:50])}")
            datastore[question_number]["all_answers"] = None

        answer = best_data.get('Answer')
        update_datastore_status_and_answer(question_number, 'done', answer)


        try:
            with csv_lock:
                write_to_csv(best_data)
        except Exception as e:
            logger.exception(f"Error writing to CSV for question {replace_newlines(question_number[:50])}: {e}")

        logger.info(f"Question finished processing in {round(time.time() - start_time, 4)} seconds")
        return answer

    except requests.Timeout:
        logger.error("A network call timed out.")
    except Exception as e:
        update_datastore_status_and_answer(question_number, 'error', 'unavailable')
        logger.exception("An error occurred in generate_and_score function.")
    finally:
        if datastore[question_number]['status'] != 'error':
            update_datastore_status_and_answer(question_number, 'done', best_data.get('Answer'))
        if timeout_event.is_set():
            logger.error("Thread did not finish within 9.5 seconds. Stopping.")

        timeout_event.clear()
        timer.cancel()
        predefined_event.clear()
        openai_event.clear()
        predefined_finished_event.clear()
        openai_finished_event.clear()

    return best_data.get('Answer')

def generate_and_score(executor, session, data, temperatures, system_prompt_indices, strategies, scoring_url, question_number, scoring_servers):
    best_data = {}
    global min_question_length, all_grouped_predefined_answers, system_prompts
    timeout_event = threading.Event()
    timer = threading.Timer(9.5, timeout_event.set)
    timer.start()
    start_time = time.time()
    result_queue = Queue()
    predefined_event = threading.Event()
    openai_event = threading.Event()
    predefined_finished_event = threading.Event()
    openai_finished_event = threading.Event()
    predefined_answers_for_current_prompt, used_key = get_predefined_answers_for_prompt(data[-1]['content'], all_grouped_predefined_answers)
    key_result = handle_key(used_key, strategies, data, system_prompt_indices) 
    system_prompt_indices, strategies, max_tokens = key_result
    predefined_future = executor.submit(score_predefined, session, data, predefined_answers_for_current_prompt, result_queue, predefined_event, timeout_event, scoring_url, question_number, predefined_finished_event, datastore)
    openai_future = executor.submit(generate_and_score_openai, session, data, temperatures, system_prompt_indices, strategies, openai_event, scoring_url, datastore, question_number, openai_finished_event, max_tokens)

    if timeout_event.is_set():
        logger.error(f"Thread for question type {used_key} timed out.")
        return

    done, not_done = wait([predefined_future, openai_future], timeout=9.5) 
    if not done:
        logger.warning(f"Thread for question type {used_key} timed out. Redundant!")
    else:
        best_data = extract_best_data_from_datastore(question_number, datastore, timeout_event, used_key, data)
        if predefined_future in not_done:
            best_data['Error'] = "Only OpenAI data was available for scoring."
        elif openai_future in not_done:
            best_data['Error'] = "Only predefined data was available for scoring."

            # Sort and update datastore
            for source in ["predefined", "openai"]:
                if source in datastore[question_number]:
                    rewards = datastore[question_number][source]["rewards"]
                    answers = datastore[question_number][source]["answers"] 

                    sorted_rewards, sorted_answers = zip(*sorted(zip(rewards, answers), reverse=True))

                    # Updating the sorted rewards and answers in the datastore
                    datastore[question_number][source]["rewards"] = list(sorted_rewards)
                    datastore[question_number][source]["answers"] = deque(sorted_answers)

        # Combine the answers and rewards from both sources
        combined_rewards = datastore[question_number].get("predefined", {}).get("rewards", []) + datastore[question_number].get("openai", {}).get("rewards", [])
        combined_answers = datastore[question_number].get("predefined", {}).get("answers", deque([])) + datastore[question_number].get("openai", {}).get("answers", deque([]))

        # Combine rewards and answers into tuples
        combined_answers_rewards = list(zip(combined_answers, combined_rewards))

        # Sort all answers based on the combined rewards
        sorted_combined_answers_rewards = sorted(combined_answers_rewards, key=lambda x: x[1], reverse=True)

        # Filter out answers with reward 0 or less
        filtered_answers_rewards = [(answer, reward) for answer, reward in sorted_combined_answers_rewards if reward > 0]

        # Only create a cycle if there are answers
        if filtered_answers_rewards:
            # Create a cycle from the filtered and sorted answers
            answers_cycle = cycle(filtered_answers_rewards)
            # Update the datastore with the answers cycle
            datastore[question_number]["all_answers"] = answers_cycle
        else:
            logger.error(f"No answers with reward > 0 for question: {replace_newlines(question_number[:50])}")
            datastore[question_number]["all_answers"] = None

        answer = best_data.get('Answer')
        update_datastore_status_and_answer(question_number, 'done', answer)


    try:
        with csv_lock:
            write_to_csv(best_data)
    except Exception as e:
        logger.exception(f"Error writing to CSV for question {replace_newlines(question_number[:50])}: {e}")

    logger.info(f"Question finished processing in {round(time.time() - start_time, 4)} seconds")
    return answer

    except requests.Timeout:
        logger.error("A network call timed out.")
    except Exception as e:
        update_datastore_status_and_answer(question_number, 'error', 'unavailable')
        logger.exception("An error occurred in generate_and_score function.")
    finally:
        if datastore[question_number]['status'] != 'error':
            update_datastore_status_and_answer(question_number, 'done', best_data.get('Answer'))
        if timeout_event.is_set():
            logger.error("Thread did not finish within 9.5 seconds. Stopping.")

        timeout_event.clear()
        timer.cancel()
        predefined_event.clear()
        openai_event.clear()
        predefined_finished_event.clear()
        openai_finished_event.clear()

    return best_data.get('Answer')


executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
session = requests.Session()
datastore = TTLCache(maxsize=5000, ttl=21600)
datastore_lock = threading.Lock() 

@app.route("/bittensor", methods=["POST"])
def chat():
    request_data = request.get_json()
    auth_token = request_data.get("verify_token")

    if auth_token != "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n":
        return jsonify({"error": "Invalid authentication token"}), 401

    data = request_data.get("messages", [])
    question_number = data[-1]['content']
    system_prompt_indices = [0]
    src_hotkey = request_data.get('src_hotkey')
    stake_amt = request_data.get('stake_amt')
    timeout = request_data.get('timeout', 10)  # Fallback to 9.5 seconds if timeout is not specified
    timeout -= 0.3

    with datastore_lock:
        if question_number not in datastore:
            # logger.info(f"New request. Not in datastore: {replace_newlines(question_number[:50])}")
            datastore[question_number] = {
                'status': 'processing',
                'count': 1,
                'Hotkeys': [src_hotkey],
                'stake_amt': stake_amt
            }
            thread = threading.Thread(target=process_question, args=(question_number, data, system_prompt_indices))
            thread.start()
        else:
            # logger.info(f"New request. Already in datastore: {replace_newlines(question_number[:50])}")
            datastore[question_number]['count'] += 1
            datastore[question_number]['Hotkeys'].append(src_hotkey)

        if datastore[question_number]['status'] == 'error':
            logger.info(f"New request. Errored question reprocessing: {replace_newlines(question_number[:50])}")
            datastore[question_number]['status'] = 'processing'
            thread = threading.Thread(target=process_question, args=(question_number, data, system_prompt_indices))
            thread.start()

    start_time = time.time()
    best_answer = None
    reward = None
    while True:
        with datastore_lock:
            status = datastore[question_number]['status']
            if status == 'done':
                if 'all_answers' in datastore[question_number] and hasattr(datastore[question_number]['all_answers'], '__next__'):
                    response, reward = next(datastore[question_number]['all_answers'])
                    best_answer = response  # Store the best answer
                else:
                    logger.error(f"No answers left for the question: {replace_newlines(question_number[:50])}")
                    if best_answer:
                        # If no answers are left, but a best answer was previously set, return that
                        logger.info(f"Returning best answer from previous iterations: {best_answer}")
                        return jsonify({'response': best_answer})
                    else:
                        logger.info(f"returning default because no answers available")
                        logger.info(f"datastore is: {datastore[question_number]}")
                        return jsonify({'response': default_answer})
                logger.info(f"response: {response[:20]}, reward: {reward}")
                return jsonify({'response': response})
            elif status == 'error':
                if best_answer:
                    logger.info(f"response: {response[:20]}, reward: {reward}")
                    return jsonify({'response': best_answer})
                else:
                    logger.info(f"returning default because of errored status")
                    return jsonify({'response': default_answer})

        if round(time.time() - start_time, 4) > timeout:
            logger.error(f"Question processing exceeded timeout, returning default")
            if best_answer:
                logger.info(f"response: {response[:20]}, reward: {reward}")
                return jsonify({'response': best_answer})
            else:
                logger.info(f"returning default because of timeout")
                return jsonify({'response': default_answer})



def process_question(question_number, data, system_prompt_indices):
    try:
        used_key = question_number[-10:]
        if len(data[-1]['content']) < min_question_length or used_key not in known_keys:  # Question is short or unknwown type
            handle_short_question(data, system_prompts, temperatures, system_prompt_indices, strategies, question_number, used_key)
        elif "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3" in datastore[question_number]['Hotkeys'] or "5Hddm3iBFD2GLT5ik7LZnT3XJUnRnN8PoeCFgGQgawUVKNm8" in datastore[question_number]['Hotkeys']:
            # logger.info("foundation or taostats sent request: %s", replace_newlines(question_number[:50]))
            # logger.info(f"datastore data for this question is {datastore[question_number]}")
            try:
                with get_scoring_url(scoring_servers) as scoring_url:
                    logger.info("temp sending all to generate and score")
                    generate_and_score(executor, session, data, temperatures, system_prompt_indices, strategies, scoring_url, question_number, scoring_servers)
            except Exception as e:
                scoring_servers.put(scoring_url)
                logger.exception("An error occurred in the process_question function.", exc_info=True)
        else:
            try:
                with get_scoring_url(scoring_servers) as scoring_url:
                    generate_and_score(executor, session, data, temperatures, system_prompt_indices, strategies, scoring_url, question_number, scoring_servers)
            except Exception as e:
                scoring_servers.put(scoring_url)
                logger.exception("An error occurred in the process_question function.", exc_info=True)
        with datastore_lock:
            datastore[question_number]['status'] = 'done' 
    except Exception as e:
        logger.exception(f"Error processing question {replace_newlines(question_number[:50])}: {e}")
        with datastore_lock:
            datastore[question_number]['status'] = 'error'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)

