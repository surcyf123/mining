import threading
import logging
import time
from utils import send_discord_alert
from csv_logger import CSVLogger
import numpy as np

prompts_fields = ["Timestamp", "Type", "Client_Identifier", "Question_Value", "Max_Time_Allowed"]
prompts_logger = CSVLogger("prompts_data.csv", prompts_fields)
timestamps_fields = ["Timestamp", "Average_Difference", "Total_Difference"]
timestamps_logger = CSVLogger("timestamps.csv", timestamps_fields)

class DatastoreManager:
    def __init__(self, completion_queue, reward_queue, process_question_function):
        self.datastore = {}
        self.timers = {}
        self.prompt_timestamps = {}
        self.prompt_timestamp_collections = {}
        self.datastore_lock = threading.Lock()
        self.completion_queue = completion_queue
        self.reward_queue = reward_queue
        self.process_question_function = process_question_function
        self.responses_needed = 0
    
    def get_info(self, prompt):
        with self.datastore_lock:
            return self.datastore.get(prompt, {})

    def update(self, prompt, client_identifier, question_value, question_type, timeout):
        # logging.info("Called datastore.update")
        current_time = time.time()
        with self.datastore_lock:
            if prompt not in self.datastore or self.datastore[prompt]['Status'] == 'error':
                prompts_logger.log(
                    Type=question_type,
                    Client_Identifier=client_identifier,
                    Question_Value=question_value,
                    Max_Time_Allowed=timeout
                )

                self.datastore[prompt] = {
                    'Status': 'Processing',
                    'Count': 1,
                    'Client Identifier': client_identifier,
                    'Question Value': question_value,
                    "Timeout" : timeout,
                }
                self.prompt_timestamps[prompt] = current_time
                self.prompt_timestamp_collections[prompt] = [current_time]
            else:
                if client_identifier == self.datastore[prompt]["Client Identifier"]:
                    self.datastore[prompt]['Count'] += 1
                    self.prompt_timestamps[prompt] = current_time 
                    self.prompt_timestamp_collections[prompt].append(current_time)
                else: 
                    logging.warning(f"Relay Attack by {client_identifier} with question_value {question_value} for prompt {prompt}")
                    send_discord_alert(f"Relay Attack by {client_identifier} with question_value {question_value} for prompt {prompt}")

    def maybe_start_processing_thread(self, prompt, question_value, timeout, question_type):
        # logging.info("called datastore.maybeprocess")
        with self.datastore_lock:
            if self.timers.get(prompt):
                self.timers[prompt].cancel()

            delay_time = 0.15
            timeout -= delay_time
            timer = threading.Timer(delay_time, self.start_processing_thread, args=(prompt, question_value, timeout, question_type))
            timer.start()
            self.timers[prompt] = timer
        
    def start_processing_thread(self, prompt, question_value, timeout, question_type):
        def completions_to_retrieve(count, qv, max_completions=25, proportionality_factor=1.75):
            def get_slope(count, lambda_value=0.03, base_slope=0.3):
                return base_slope * np.exp(-lambda_value * count)

            if self.datastore[prompt]["Client Identifier"] == "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3":
                max_completions == self.completion_queue.qsize() - 2

            slope = qv * get_slope(count)
            initial_value = qv * proportionality_factor
            if count == 1:
                return int(round(min(initial_value, max_completions)))
            return min(int(np.ceil(initial_value + slope * count)), max_completions)

        # logging.info(f"Processing {self.datastore[prompt]['Count']} completions for {self.datastore[prompt]['Client Identifier'][:10]} ({round(self.datastore[prompt]['Question Value'], 3)}) after {round(time.time() - self.prompt_timestamps[prompt], 3)}s with timeout {self.datastore[prompt]['Timeout']}")
        with self.datastore_lock:
            self.datastore[prompt]["Completions to Retrieve"] = completions_to_retrieve(self.datastore[prompt]["Count"], question_value)
            if self.datastore[prompt]["Completions to Retrieve"] > self.completion_queue.qsize():
                self.datastore[prompt]["Completions to Retrieve"] = self.completion_queue.qsize()
            self.responses_needed = self.datastore[prompt]["Count"]
            self.datastore[prompt]['Status'] = 'ThreadStarted'
            self.log_timestamps(self.prompt_timestamp_collections[prompt])
            completions_to_retrieve = self.datastore[prompt]["Completions to Retrieve"]
            
            logging.info(f"{self.responses_needed} responses needed to {self.datastore[prompt]['Client Identifier'][:10]}, gathering {completions_to_retrieve} completions")
            if completions_to_retrieve > 0:
                thread = threading.Thread(target=self.process_question_function, args=(prompt, self, completions_to_retrieve, timeout, self.datastore[prompt]['Client Identifier'][:10], question_type))
                thread.start()

    def log_timestamps(self, timestamps):
        # Calculate the differences between consecutive timestamps
        differences = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        # Calculate the average difference
        avg_diff = sum(differences) / len(differences) if differences else 0
        
        # Calculate the total difference
        total_diff = timestamps[-1] - timestamps[0] if timestamps else 0

        # Log the values
        timestamps_logger.log(
            Timestamp=time.time(),
            Average_Difference=avg_diff,
            Total_Difference=total_diff
        )
