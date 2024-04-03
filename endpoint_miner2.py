import argparse
import bittensor
from typing import List, Dict
import requests
import json
import traceback
import logging

default_answer = '''
I deeply apologise. I was unable to process the response in time. HTTP///:::>>> Error code: H100/POD9/ID:271765 unavailable
'''

logging.basicConfig(level=logging.INFO, format='%(message)s')
class EndpointMiner(bittensor.BasePromptingMiner):

    @classmethod
    def check_config(cls, config: 'bittensor.Config'):
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument('--endpoint.verify_token', default="SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n", type=str, help='Auth')
        parser.add_argument('--endpoint.url', default="http://216.153.49.230:5000/bittensor", type=str, help='Endpoint')
        # parser.add_argument( '--back_up_endpoint.url', default="http://216.153.62.153:1130/bittensor", type=str, help='Endpoint')

    def init(self):
        super(EndpointMiner, self).init()
        logging.info(f"Config: {self.config}")

    def forward(self, messages: List[Dict[str, str]]) -> str:
        resp = default_answer
        src_hotkey = self.src_hotkey  # Extract the hotkey
        logging.info(f"Extracted src_hotkey: {src_hotkey}")

        
        # Construct the data payload to send
        data_to_send = {
            "messages": messages,
            "verify_token": self.config.endpoint.verify_token,
            "src_hotkey": src_hotkey
        }

        try:
            logging.info(f"Sending request to endpoint: {self.config.endpoint.url}")
            response = requests.post(
                self.config.endpoint.url,
                data=json.dumps(data_to_send),
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()  # Raise exception for non-2xx status codes
            resp = response.json()["response"]
            logging.info(f"Successful response from endpoint. First 100 chars of answer: {resp[0:100]}")
        except Exception as e:
            logging.error(f"Failed to get response from primary endpoint: {e}")
            if self.config.back_up_endpoint is not None:
                try:
                    logging.info(f"Trying backup endpoint: {self.config.back_up_endpoint.url}")
                    response = requests.post(
                        self.config.back_up_endpoint.url,
                        data=json.dumps({"messages": messages, "verify_token": self.config.endpoint.verify_token}),
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )
                    response.raise_for_status()  # Raise exception for non-2xx status codes
                    resp = response.json()["response"]
                    logging.info(f"Successful response from backup endpoint. First 100 chars of answer: {resp[0:100]}")
                except requests.exceptions.RequestException as e:
                    logging.error(f"Failed to get response from backup endpoint: {e}")
                    resp = default_answer
            else:
                logging.info("No backup endpoint available.")
        logging.info(f"Returning response. First 100 chars: {resp[0:100]}")

        return resp

if __name__ == "__main__":
    bittensor.utils.version_checking()
    EndpointMiner().run()
