import logging
import requests
import time
import threading

def send_discord_alert(message):
    webhook_url = "https://discord.com/api/webhooks/1149956694642794506/Axy2w4Md5YfGq-tpkaGPrZgtUVN8wxMNPMofOGX9vF2PzCvxaVYhtSgYhctdXKDZFt7Y"
    data = {
        "content": f"@everyone {message}",
        "username": "AlertBot"
    }
    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            logging.info("Discord alert sent successfully!")
        else:
            logging.error(f"Failed to send Discord alert. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Failed to send Discord alert: {e}", exc_info=True)


#test what is better
def task_validate(question_type, completion):
    # Define keywords and their categories
    keywords = {
        'summary': ['summary:', 'paraphrase:', 'paraphrasing:', 'paraphrased:'],
        'question': ['question:', 'query:', 'q:'],
        'answer': ['answer:', 'response:', 'a:', 'completion:']
    }
    
    # Define the conditions for each category
    conditions = {
        'summary': not question_type == "augment",
        'question': question_type == "augment" or question_type.startswith("answer"),
        'answer': question_type == "augment" or question_type.startswith("followup")
    }
    
    # Function to replace keywords based on conditions
    def replace_keywords(category, completion):
        if conditions[category] and any(kw in completion.lower() for kw in keywords[category]):
            for kw in keywords[category]:
                # Check both the lowercase and the capitalized version
                for variant in [kw, kw.capitalize()]:
                    if variant in completion:
                        completion = completion.replace(variant, variant.rstrip(':') + '-') # remove + '-' to just remove the :
        return completion
    
    # Process each category
    for category in keywords:
        completion = replace_keywords(category, completion)
    
    return completion

def delayed_cleanup(prompt, datastore_manager):
    with datastore_manager.datastore_lock:
        # Cancel the timer associated with the prompt
        if prompt in datastore_manager.timers:
            datastore_manager.timers[prompt].cancel()
            del datastore_manager.timers[prompt]
        
        # Remove the prompt from various collections
        if prompt in datastore_manager.datastore:
            del datastore_manager.datastore[prompt]
        if prompt in datastore_manager.prompt_timestamps:
            del datastore_manager.prompt_timestamps[prompt]
        if prompt in datastore_manager.prompt_timestamp_collections:
            del datastore_manager.prompt_timestamp_collections[prompt]

def cleanup_prompt(prompt, datastore_manager):
    # Start a timer to call the actual cleanup after 10 seconds
    timer = threading.Timer(10.0, delayed_cleanup, args=[prompt, datastore_manager])
    timer.start()
