from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

MODEL_NAME = "cerebras/btlm-3b-8k-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

@app.route('/generate', methods=['POST'])
def generate():
    content = request.json
    prompt = content.get('prompt', '')
    
    if not prompt:
        return jsonify(error="Prompt not provided"), 400

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    completions = model.generate(input_ids, max_length=50, num_return_sequences=5, pad_token_id=tokenizer.eos_token_id, do_sample=True)

    completions_texts = [tokenizer.decode(completion, skip_special_tokens=True) for completion in completions]
    
    return jsonify(completions=completions_texts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
