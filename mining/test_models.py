import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

prompt1 = """
What is the Schwinger-DeWitt approach? How does it relate to vacuum polarization effects of massive fields in curved backgrounds? Provide an overview of its application in obtaining analytical expressions for the one-loop quantum effective action.\nPrevious Question \nQuestion:ask a new question instead. \n\nWhat are some potential applications of this result? How might it impact our understanding of other physical phenomena?\nThis result could have significant implications for our understanding of other physical phenomena such as electromagnetism, weak nuclear force, and strong nuclear force. By studying the behavior of these forces in curved backgrounds, we may gain new insights into their underlying principles and potentially make more accurate predictions about their effects on matter and energy. Additionally, this result could lead to improved models of particle interactions and cosmological scenarios, which could ultimately help us better understand the fundamental nature of the universe.\nAnswer:BREAKING NEWS! \ud83d\udea8 New study reveals that #vacuumpolarization effects of massive fields in curvedbackgrounds can be calculated using the Schwinger-DeWitt approach. This game-changing technique gives us ANALYTICAL EXPRESSIONS for the one-loop quantum effective action! \ud83c\udf0d Impactful implications for our undersatnding of other physics phenoma like Electromagnetism, Weak Nuclear Force & Strong Nuclear Force! \ud83d\udca1 Could lead to preciser models of particle interactions & cosmo scenarios! \u2728 Keep 'em peeled for more updates from @PhysicsHub \ud83d\udc40\nQuestion:What are some specific examples of physical phenomena that could be studied using the Schwinger-DeWitt approach?\nSome specific examples of physical phenomena that could be studied using the Schwinger-DeWitt approach include:\n1. Electromagnetic fields in curved spacetime: The behavior of electromagnetic fields, such as light waves or electric and magnetic fields, can be analyzed using the Schwinger-DeWitt approach to gain insights into their propagation and interactions in curved backgrounds.\n2. Weak nuclear force: The weak nuclear force, which is responsible for radioactive decay and neutrino interactions, can also be investigated using this technique to understand its behavior in different gravitational environments.\n3. Strong nuclear force: Similarly, the strong nuclear force, which binds protons and neutrons together within atomic nuclei\nAnswer:Here is a question. What applications could the Schwinger-DeWitt approach have in the field of electromagnetism?\nThe answer is as follows. The Schwinger-DeWitt approach could have significant applications in electromagnetism, such as providing a deeper understanding of light waves and the propagation of electric and magnetic fields in curved spacetime. This could potentially reveal new insights into the behavior of light and other electromagnetic phenomena, leading to more accurate predictions and models of electromagnetic interactions in varied gravitational environments.\n\nQuestion:Here is a question. How does the Schwinger-DeWitt approach help us understand electromagnetic phenomena better?\nParaphrased The answer is as follows. The Schwinger-DeWitt approach can provide deeper insights into electromagnetic phenomena by allowing us to study their behavior in curved backgrounds, which can reveal new insights into the underlying principles and make more accurate predictions about their effects on matter and energy.\nAnswer the question step by step and explain your thoughts. Do not include questions or summaries in your answer.
"""

prompt2 = "How are you?"


urls = [
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

start_time = time.time()
prompts = [prompt1, prompt1]

def send_request(args):
    url, prompt = args
    data = {
        "prompt": prompt,
        "n": 16,
        "num_tokens": 160,
        "top_p": .92,
        "temperature": .8,
        "top_k": 1000,
    }
    
    try:
        response_start_time = time.time()
        response = requests.post(url, json=data, timeout=60)
        response_data = response.json()
        response_time = time.time() - response_start_time

        # Print results excluding 'response' key
        for key, value in response_data.items():
            if key != 'response':
                print(f"{key}: {value}")

        return response_data, response_time
    except requests.Timeout:
        print(f"Request to {url} timed out.")
    except requests.RequestException as e:
        print(f"Request to {url} failed with error: {e}")

tasks = [(url, prompt) for url in urls for prompt in prompts]

# Use ThreadPoolExecutor to send the requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    # Submit tasks for execution
    futures = [executor.submit(send_request, task) for task in tasks]
    
    # Process results as they become available
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Task failed with error: {e}")

print(f"Total time taken= {time.time() - start_time}")


