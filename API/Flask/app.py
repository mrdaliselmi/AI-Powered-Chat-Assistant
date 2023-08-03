from flask import Flask, jsonify, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

model_name="microsoft/GODEL-v1_1-base-seq2seq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/api/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    instruction = data.get('instruction', "Instruction: given a dialog context, you need to respond formally, in an office environment.")
    knowledge = data.get('knowledge', "")
    dialog = data['dialog']
    num_responses = data.get('num_responses', 3)
    max_length = data.get('max_length', 10)
    min_length = data.get('min_length', 1)
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids

    # Move tensors to CUDA device
    input_ids = input_ids.to(device)

    # Generate outputs on GPU
    outputs = model.generate(input_ids, max_length=max_length, min_length=min_length, top_p=0.9, do_sample=True, num_return_sequences=num_responses)

    # Move outputs back to CPU
    outputs = outputs.to("cpu")

    responses = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        responses.append(response)

    return jsonify(response=responses)


if __name__ == '__main__':
    app.run(debug=True)
