print('setting up libraries... ')
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import time

#from ipywidgets import widgets
import torch
from transformers import *
from flask_cors import CORS
import requests

import json


app=Flask(__name__)
CORS(app)

@app.route('/')
def home():
    ip = requests.get('https://checkip.amazonaws.com').text.strip()
    return render_template('index.html', server_ip=ip)



@app.route('/api/', methods=['POST'])
def ans():
    data = request.get_json()

    question = data['question']
    text=data['paragraph']
    st=time.time()

    input_text = question + " [SEP] " + text
    input_ids = tokenizer.encode(input_text)
    start_scores, end_scores = model(torch.tensor([input_ids]).to(device))
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)  
    ans=' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    ans = ans.replace(' ##', "")

    time_taken = time.time() - st
    print('answer: ', ans)

    j_ans = {'answer': ans}

    #print(ans) 

    return jsonify(j_ans)

def get_gpu_status(device):

    #Additional Info when using cuda
    if device.type =='cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

if __name__ == '__main__':
    
    print('Loading models... ')
    tokenizer = tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
    print('model loaded!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    model.to(device)
    get_gpu_status(device)

    #@title String fields

    app.run(host='0.0.0.0')






