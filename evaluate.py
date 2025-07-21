from flask import Flask, render_template, request, jsonify, make_response
import os
import re
import json
import torch
import requests
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import sys
from werkzeug.utils import secure_filename
import random
from bert_score import score  
from pyswip import Prolog, Atom
from datasets import load_dataset
from datasets import Dataset
from difflib import SequenceMatcher
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import io
from contextlib import redirect_stdout


app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

dataset_format = {
    "qid": "",
    "category": "",
    "reasoning": "",
    "entityid": "",
    "entity": "",
    "description": "",
    "question": "",
    "answer": "",
    "evidence": []
}

current_log_file = None

def get_wikidata_id_from_wikipedia_url(wikipedia_url):
    title = wikipedia_url.split("/")[-1]
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "format": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_info in pages.items():
            if "pageprops" in page_info and "wikibase_item" in page_info["pageprops"]:
                return {
                    "entity": f"http://www.wikidata.org/entity/{page_info['pageprops']['wikibase_item']}",
                    "entityLabel": title.replace("_", " ")
                }
    except requests.exceptions.SSLError:
        print(f"SSL error，skip {wikipedia_url}")
    except requests.exceptions.RequestException as e:
        print(f"request {wikipedia_url} fail: {e}")
    
    return None  




def calculate_similarity(sentence1, sentence2):
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')    
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)    
    similarity = util.cos_sim(embedding1, embedding2)   
    similarity_percentage = similarity.item() * 100

    return similarity_percentage

def get_average_similarity(array1, array2):
   
    if len(array1) != len(array2):
        raise ValueError("error")
    
    total_similarity = 0.0

    
    for str1, str2 in zip(array1, array2):
        similarity = calculate_similarity(str1, str2)
        total_similarity += similarity

    
    average_similarity = total_similarity / len(array1)

    return average_similarity



def loadset(rule, domain):
    base_dir = 'data/generated'
    questions = []
    standard_answers = []
    
    rule_dir = os.path.join(base_dir, rule)
    if not os.path.exists(rule_dir):
        return questions, standard_answers
    
    for file_name in os.listdir(rule_dir):
        if file_name.endswith(f'{domain}_qa.json'):
            file_path = os.path.join(rule_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                new_facts_data = json.load(f)
            
            sampled_facts = random.sample(new_facts_data, min(50, len(new_facts_data)))
            for fact in sampled_facts:
                full_question = (
                    f"Description:\n{fact['description']}\n\n"
                    f"Question:\n{fact['question']}\n\n"
                    "Answer me with ONE word 'yes' or 'no'."
                )
                questions.append(full_question)
                standard_answers.append(fact["answer"])
    
    return questions, standard_answers


