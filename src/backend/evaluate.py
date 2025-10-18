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







def loadset(rule, domain, file_path=None):
    questions = []
    standard_answers = []

    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                new_facts_data = json.load(f)

            # 保持原有的采样逻辑（最多50条）
            sampled_facts = random.sample(new_facts_data, min(50, len(new_facts_data)))
            for fact in sampled_facts:
                full_question = (
                    f"Question:\n{fact['question']}\n\n"  
                    "Answer me with ONE word 'yes' or 'no'." 
                )
                questions.append(full_question)
                standard_answers.append(fact["answer"])
            print(f"Loaded {len(questions)} samples from specified path: {file_path}")
            return questions, standard_answers

        except Exception as e:
            print(f"Error loading dataset from {file_path}: {str(e)}")
            return questions, standard_answers


    print(f"No valid dataset path provided or file not found: {file_path}")
    return questions, standard_answers