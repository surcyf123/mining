import openai
openai.api_key = ""
import time

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
        print(f"OpenAI API request took {duration} seconds")
        answer = response["choices"][0]["message"]["content"].strip()
        return (answer)
    except Exception as e:
        print(f"got exception when calling openai {e}")
        return "error calling model"


prompt = "what is your name?"
print(send_openai_request(prompt))