from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import os
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import argparse

parser = argparse.ArgumentParser(description='GODEL based chatbot')
parser.add_argument('--model_path', type=str,default="",
                    help='path to the fine-tuned model')


args = parser.parse_args()

root =os.getcwd()
version = "1.0.0"
if args.model_path != "":
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
else:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
def preprocessing(text):
   text = text.translate(str.maketrans("","",string.punctuation))
   text = text.lower().split()
   text = [wordnet_lemmatizer.lemmatize(i) for i in text if i not in nltk.corpus.stopwords.words('french')]
   return " ".join(text)

def generate(knowledge, dialog, num_return_sequences):
    if knowledge != '':
        instruction = f'Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.'
        knowledge = '[KNOWLEDGE] ' + knowledge
    else:
        instruction = f'Instruction: given a dialog context, you need to response empathically.'
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True,num_return_sequences=num_return_sequences)
    responses = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        responses.append(response)
    return responses

if __name__ =="__main__":
    dialog = []
    print("Enter 'quit' to exit")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            break
        dialog.append(message)
        knowledge = ""
        answer = generate(knowledge, dialog, num_return_sequences=3)
        print("Model: "+ str(answer[0]))
        dialog.append(answer)