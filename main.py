import os
import re
import io
import json
import torch
import requests
from tqdm import tqdm
import psutil
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import threading
import sys
from werkzeug.utils import secure_filename
import random
import traceback
from datetime import datetime
import numpy as np
from flask import Flask, render_template, request, jsonify, make_response
from contextlib import redirect_stdout

from bert_score import score  
from pyswip import Prolog, Atom
from datasets import load_dataset
from datasets import Dataset
from difflib import SequenceMatcher
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline , AutoModelForSequenceClassification
from data_preprocess import rag_preprocess
from download import download_rag
from embed_faiss import build_faiss_index,load_embedding_model,load_faiss_index


from get_wiki_cat_id import get_category_pages, save_links_to_file, get_category_members, extract_entity_pages, save_entity_links_to_file
from transitive_entity_extract import get_entity_info, get_predicate_labels, replace_predicates_with_labels
from transitive_pl_build import replace_special_characters, save_to_pl_file, process_prolog_file
from wiki_pl_build import get_related_entity_list, get_prop_list, replace_special_characters1, save_to_pl_file1, process_prolog_file1
from rule_generation import RuleGenerator
from prolog_inference import normalize_path, safe_consult, get_wikipedia_summary, replace_special_characters2, is_q_followed_by_digits, negation_prolog_inference, composite_prolog_inference, inverse_prolog_inference
from question_generation import inverse_template, negation_template, composite_template
from evaluate import  loadset, get_wikidata_id_from_wikipedia_url

app = Flask(__name__, static_folder='static', template_folder='templates')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = 'RAG_raw'  
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

class BinaryAnswerClassifier:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.labels = {0: "no", 1: "yes"}  
    
    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        
        
        probabilities = torch.softmax(logits, dim=1)
        confidence = probabilities[0][prediction].item()
        
        return self.labels[prediction], confidence
def extract_valid_answer(full_output: str, prompt: str) -> str:

    prompt_end_idx = full_output.find(prompt) + len(prompt)

    valid_answer = full_output[prompt_end_idx:].strip()

    return valid_answer if valid_answer else full_output

# init
answer_classifier = BinaryAnswerClassifier()

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
# global
current_log_file = None
rag_processes = {}  
current_rag_log = None

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  

def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        gpu_utilization = int(result.stdout.strip())
        return gpu_utilization
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
        return 0

def get_device():
    
    if torch.cuda.is_available():
        return 0  
    elif torch.backends.mps.is_available():
        return "mps"  
    else:
        return -1
      
@app.route('/')
def index():
    
    return render_template('index.html')
@app.route('/basic_evaluation.html')
def basic_evaluation():
    return render_template('basic_evaluation.html')
@app.route('/rag_evaluation.html')
def rag_evaluation():
    return render_template('rag_evaluation.html')
@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    
    try:
        data = request.json
        rule_choice = data.get('rule_choice')
        domain_choice = data.get('domain_choice')
        model_choice = data.get('model_choice')
        dataset_source = data.get('dataset_source', 'existing')  
        
        if not all([rule_choice, domain_choice, model_choice]):
            return jsonify({"status": "error", "message": "Missing required parameters"}), 400
        
        timestamp = int(time.time())
        log_filename = f"evaluation_{timestamp}.log"  
        log_file = os.path.join("logs", log_filename)  
        
        
        thread = threading.Thread(
            target=run_evaluation, 
            args=(rule_choice, domain_choice, model_choice, log_file, dataset_source)
        )
        thread.daemon = True
        thread.start()
        

        return jsonify({
            "status": "processing", 
            "message": "Evaluation started. Check logs for details.",
            "log_file": log_filename, 
            "dataset_source": dataset_source
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing request: {str(e)}"}), 500
    
def generate_answers(model_name, questions):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=False, torch_dtype=torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, is_decoder=True)

    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    answers = []
    response_times = []
    memory_usages = []
    gpu_utilizations = []

    for question in questions:
        start_time = time.time()
        start_memory = get_memory_usage()
        start_gpu = get_gpu_utilization()

        response = text_generator(question, max_new_tokens=300, num_return_sequences=1, truncation=True)

        end_time = time.time()
        end_memory = get_memory_usage()
        end_gpu = get_gpu_utilization()

        response_time = end_time - start_time
        response_times.append(response_time)
        memory_usages.append(end_memory - start_memory)
        gpu_utilizations.append((start_gpu + end_gpu) / 2)

        generated_text = response[0]['generated_text'].strip()

        if generated_text.startswith(question):
            answer = generated_text[len(question):].strip()
        else:
            answer = generated_text
        if not answer:
            answer = "No answer"
        answers.append(answer)

    return answers, response_times, memory_usages, gpu_utilizations  



def evaluate_model(model_name, questions, standard_answers):
    model_answers, response_times, memory_usages, gpu_utilizations = generate_answers(model_name, questions)

    basic_correct = 0
    total = len(questions)
    total_response_time = sum(response_times)
    average_response_time = total_response_time / total if total > 0 else 0

    positive_memory_usages = [mem for mem in memory_usages if mem > 0]
    if positive_memory_usages:
        average_memory_usage = sum(positive_memory_usages) / len(positive_memory_usages)
    else:
        average_memory_usage = 0
    average_gpu_utilization = sum(gpu_utilizations) / total if total > 0 else 0

  

    results = {
        "model_name": model_name,
        "questions": [],
        "basic_accuracy": 0.0,
        "average_response_time": average_response_time,
        "average_memory_usage": average_memory_usage,
        "average_gpu_utilization": average_gpu_utilization,
    }

    for i, (question, model_answer, standard_answer, response_time, memory_usage, gpu_utilization) in enumerate(
            zip(questions, model_answers, standard_answers, response_times, memory_usages, gpu_utilizations)):
        raw_reference = standard_answer.strip().lower()
        reference_answer = re.sub(r'[^a-z]', '', raw_reference)

        predicted_label, confidence = answer_classifier.predict(model_answer)

        if not model_answer or model_answer in ["No answer"]:
            predicted_label = "none"
            is_correct = False
        else:
            is_correct = (predicted_label == reference_answer)
            if is_correct:
                basic_correct += 1

        results["questions"].append({
            "question": question,
            "model_answer": model_answer,
            "reference_answer": reference_answer,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "is_correct": is_correct,
            "response_time": response_time,
            "memory_usage": memory_usage,
            "gpu_utilization": gpu_utilization
        })

    basic_accuracy = (basic_correct / total) * 100 if total > 0 else 0
    results["basic_accuracy"] = basic_accuracy

    return json.dumps(results, indent=4, ensure_ascii=False)
  
def run_evaluation(rule_choice, domain_choice, model_choice, log_file, dataset_source):
   
    global current_log_file
    current_log_file = log_file
    
    os.makedirs("results", exist_ok=True)  
    
   
    with open(log_file, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {device}")
                
               
                rule_map = {"1": "inverse", "2": "negation", "3": "composite"}
                reasoning_type = rule_map.get(rule_choice, "inverse")
                print(f"Selected reasoning rule: {reasoning_type}")
                
                
                domains = ["geography", "history", "health", "mathematics", "nature", "people", "society", "technology", "culture"]
                domain = domains[int(domain_choice) - 1] if domain_choice.isdigit() and 1 <= int(domain_choice) <= 9 else "geography"
                print(f"Selected domain: {domain}")
                print(f"Processing domain: {domain}")
                
                
                if dataset_source == "existing":
                    print("Using existing dataset...")
                    
                    questions, standard_answers = loadset(reasoning_type, domain)
                    print(f"Loaded {len(questions)} questions")
                else:

                    print("Dynamically generating new dataset...")
    
                    # Get all the category links
                    category_name = f"{domain}"  
                    category_file = f"category/{domain.lower()}_links.txt"
                    save_links_to_file(category_name, category_file)
                    print(f"Category links saved to {category_file}")
                    
                    # Get the entity pages from the selected categories
                    selected_categories = []
                    with open(category_file, 'r') as file:
                        for line in file:
                            selected_categories.append(line.split('/Category:')[-1].strip())
                    selected_file = f"category/selected/{domain.lower()}_wiki_links.txt"
                    max_pages_limit = 200
                    wikipedia_urls = save_entity_links_to_file(selected_categories, selected_file, max_pages=max_pages_limit)
                    print(f"Entity links saved to {selected_file}, with a total of {len(wikipedia_urls)} links")
                    
                    # Get Wikidata IDs for the entity pages
                    wikidata_entities = []
                    for url in tqdm(wikipedia_urls, desc=f"Getting Wikidata IDs for {domain}"):
                        result = get_wikidata_id_from_wikipedia_url(url)
                        if result:
                            wikidata_entities.append(result)
                        time.sleep(1)
                    
                    output_file = f'wiki/{domain}_useful_entities.json'
                    if not os.path.exists(os.path.dirname(output_file)):
                        os.makedirs(os.path.dirname(output_file))
                    with open(output_file, 'w') as f:
                        json.dump(wikidata_entities, f, indent=4)
                    print(f"Wikidata entities saved to {output_file}, with a total of {len(wikidata_entities)} entities")
                    
                    # transitive_entity_extract.py
                    print(f"transitive_entity_extract.py Processing topic: {domain}")
                    try:
                        with open(f'wiki/{domain}_useful_entities.json', 'r', encoding='utf-8') as f:
                            data = json.load(f)
                                                     
                        class_list = []
                        for item in data:
                            class_list.append((item['entity'].split('/')[-1], item['entityLabel']))
                        
                        useful_list = []
                        for entity_id, entity_name in tqdm(class_list, desc=f"Processing entities for {domain}"):
                            useful_entities = get_entity_info(entity_id, entity_name)
                            if len(useful_entities) == 0:
                                continue
                            useful_list.extend(useful_entities)
                        
                        
                        predicate_set = {entity["predicate"] for entity in useful_list}
                        predicate_label_map = get_predicate_labels(predicate_set)
                        updated_entities_list = replace_predicates_with_labels(useful_list, predicate_label_map)
                        
                        with open(f'wiki/transitive/{domain}_useful_entities.json', 'w', encoding='utf-8') as f:
                            json.dump(updated_entities_list, f, indent=4)
                        print(f"Transitive entities processed and saved to wiki/transitive/{domain}_useful_entities.json")
                    except FileNotFoundError:
                        print(f"Error: Could not find entity file for topic '{domain}'")
                        return {"status": "error", "message": f"Error: Could not find entity file for topic '{domain}'"}
                    except Exception as e:
                        print(f"Error processing topic '{domain}': {str(e)}")
                        return {"status": "error", "message": f"Error processing topic '{domain}': {str(e)}"}
                    
                    # transitive_pl_build.py
                    print(f"transitive_pl_build.py Processing topic: {domain}")
                    output_folder = 'wiki/transitive/pl_files'
                    backup_list = {}
                    log_entities = []
                    wiki_entity_path = f'wiki/transitive/{domain}_useful_entities.json'
                    with open(wiki_entity_path, 'r') as file:
                        entities = json.load(file)
                    for entity in tqdm(entities):
                        output_file = os.path.join(output_folder, f'{domain}.pl')
                        save_to_pl_file(entity, output_file, backup_list)
                        if entity not in log_entities:
                            log_entities.append(entity)
                    process_prolog_file(output_file, output_file)
                    with open(f'wiki/transitive/backup/{domain}_backup_list.json', 'w') as file:
                        json.dump(backup_list, file, indent=4)
                    with open(f'wiki/transitive/log/{domain}_log.json', 'w') as file:
                        json.dump(log_entities, file, indent=4)
                    print(f"Transitive Prolog files built and saved to {output_folder}")
                    
                    # wiki_pl_build.py
                    print(f"wiki_pl_build.py Processing topic: {domain}")
                    backup_list = {}
                    log_entities = []
                    output_folder = 'wiki/pl_files'
                    prop_list = get_prop_list('utils/wiki_property_cat_v1.xlsx')
                    wiki_entity_path = f'wiki/{domain}_useful_entities.json'
                    with open(wiki_entity_path, 'r', encoding='utf-8') as file:
                        entities = json.load(file)
                    for entity in tqdm(entities):
                        entity_id = entity['entity'].split('/')[-1]
                        entity_name = entity['entityLabel']
                        related_entities = get_related_entity_list(entity_id, entity_name, prop_list)
                        output_file = os.path.join(output_folder, f'{domain}.pl')
                        save_to_pl_file1(related_entities, output_file, backup_list)
                        for related_entity in related_entities:
                            if related_entity not in log_entities:
                                log_entities.append(related_entity)
                    process_prolog_file1(output_file, output_file)
                    with open(f'wiki/backup/{domain}_backup_list.json', 'w', encoding='utf-8') as file:
                        json.dump(backup_list, file, indent=4)
                    with open(f'wiki/log/{domain}_log.json', 'w', encoding='utf-8') as file:
                        json.dump(log_entities, file, indent=4)
                    print(f"Wiki Prolog files built and saved to {output_folder}")
                    
                    # rule_generation.py
                    print(f"rule_generation.py Processing topic: {domain}")
                    properties_file = 'utils/wiki_property_cat_v1.xlsx'
                    output_folder = 'prolog_rules'
                    rule_generator = RuleGenerator(properties_file, output_folder)
                    rule_generator.generate_rules()
                    rule_generator.write_rules_to_files()
                    print(f"Prolog rules generated and saved to {output_folder}")
                    
                    # prolog_inference.py
                    print(f"prolog_inference.py Processing topic: {domain}")
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    new_fact_list = []

                    prolog_fact_file = os.path.join(current_dir, 'wiki', 'pl_files', f'{domain}.pl')
                    backup_file = os.path.join(current_dir, 'wiki', 'backup', f'{domain}_backup_list.json')
                    log_entities_file = os.path.join(current_dir, 'wiki', 'log', f'{domain}_log.json')

                    prolog_query_file = os.path.join(current_dir, 'prolog_rules', f'{reasoning_type}_rules.pl')
                    problem_file = os.path.join(current_dir, 'prolog_rules', f'{reasoning_type}_problem_dict.json')

                    output_dir = os.path.join(current_dir, 'data', 'derived', reasoning_type)
                    os.makedirs(output_dir, exist_ok=True)

                    if reasoning_type == 'inverse':
                        new_fact_list = inverse_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list)
                    elif reasoning_type == 'composite':
                        new_fact_list = composite_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list)
                    elif reasoning_type == 'negation':
                        new_fact_list = negation_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list)
                    
                    output_file = os.path.join(output_dir, f'{domain}_new_facts.json')
                    with open(output_file, 'w') as f:
                        json.dump(new_fact_list, f, indent=4)
                    print(f"Inferred new facts saved to {output_file}, with a total of {len(new_fact_list)} facts")
                    
                    # question_generation.py
                    print(f"question_generation.py Processing topic: {domain}")
                    rule = reasoning_type
                    new_facts_file = f'data/derived/{rule}/{domain}_new_facts.json'
                    with open(new_facts_file, 'r') as f:
                        new_facts = json.load(f)

                    generated_qa_list = []

                    if rule == 'inverse':
                        inverse_template(new_facts, generated_qa_list, f'prolog_rules/{rule}_problem_dict.json')
                    elif rule == 'negation':
                        negation_template(new_facts, generated_qa_list, f'prolog_rules/{rule}_problem_dict.json')
                    elif rule == 'composite':
                        composite_template(new_facts, generated_qa_list, f'prolog_rules/{rule}_problem_dict.json', f'wiki/backup/{domain}_backup_list.json')
                    
                    
                    output_dir = f'data/generated/{rule}/'
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    output_file = f'{output_dir}/{domain}_qa.json'
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(generated_qa_list, f, indent=4, ensure_ascii=False)
                    print(f"Question-answer pairs generated and saved to {output_file}, with a total of {len(generated_qa_list)} pairs")
                    
                   
                    questions, standard_answers = loadset(reasoning_type, domain)
                    print(f"Loaded {len(questions)} newly generated questions")
                
                
                print(f"choose model: {model_choice}")
                model_map = {
                    "1": "Qwen/Qwen-1_8B",
                    "2": "gpt2-medium",
                    "3": "EleutherAI/gpt-neo-125M"
                }

                model_name = model_map.get(model_choice)
                if model_name:
                    print(f"model chosen: {model_name}")
                    
                    json_result = evaluate_model(model_name, questions, standard_answers)
                    print(f"evaluation results: {json_result}")
                    
                    
                    timestamp = int(time.time())  
                    result_filename = f"{reasoning_type}_{domain}_{model_name.replace('/', '_')}_{timestamp}.json"
                    result_path = os.path.join("results", result_filename)  
                    
                    
                    with open(result_path, "w", encoding="utf-8") as result_file:
                        result_file.write(json_result)  
                    
                    print(f"Results saved to: {result_path}")  
                    
                    return {
                        "status": "complete", 
                        "message": "evaluation completed", 
                        "evaluation_result": json.loads(json_result), 
                        "questions": questions[:5], 
                        "standard_answers": standard_answers[:5],
                        "result_path": result_path  
                    }
                else:
                    print("Invalid model selection")
                    return {"status": "error", "message": "Invalid model selection"}
                
            except Exception as e:
                print(f"Error occurred during evaluation: {str(e)}")
                return {"status": "error", "message": f"Error occurred during evaluation: {str(e)}"}

@app.route('/api/dataset/<rule>/<domain>')
def get_dataset(rule, domain):
    base_dir = 'data/generated'
    rule_dir = os.path.join(base_dir, rule)
    dataset_file = os.path.join(rule_dir, f'{domain}_qa.json')
    
    try:
        
        if not os.path.exists(dataset_file):
            return jsonify({
                "status": "error", 
                "message": f"Dataset file does not exist: {dataset_file}"
            }), 404
        
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        
        if not dataset:
            return jsonify({
                "status": "error", 
                "message": "dataset empty"
            }), 400
        
        
        sample_size = min(5, len(dataset))
        sampled_data = dataset[:sample_size]  
        
        questions = []
        answers = []
        for item in sampled_data:
            full_question = (
                f"Description:\n{item['description']}\n\n"
                f"Question:\n{item['question']}\n\n"
                "Answer me with 'yes' or 'no'. Just 'yes' or 'no', no more other words, no more!"
            )
            questions.append(full_question)
            answers.append(item["answer"])
        
        return jsonify({
            "status": "success",
            "message": f"Loaded dataset samples for {domain} domain under {rule} rule",
            "sample_size": sample_size,
            "total_size": len(dataset),  
            "questions": questions,
            "answers": answers
        })
    
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error retrieving dataset: {str(e)}"
        }), 500
@app.route('/api/logs/<log_file>')
def get_logs(log_file):
    
    full_path = os.path.join("logs", log_file)
    
    if not os.path.exists(full_path):
        return jsonify({"status": "not_found", "message": "Log file does not exist"}), 404
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            logs = f.read()
        return jsonify({"status": "success", "logs": logs})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error reading log file: {str(e)}"}), 500
@app.route('/api/progress/<path:log_file>')
def get_progress(log_file):
    
    full_path = os.path.join("logs", log_file)
    
    if not os.path.exists(full_path):
        return jsonify({"status": "processing", "progress": 0, "message": "Log file does not exist; evaluation may not have started yet"}), 200
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            logs = f.read()
        
        
        progress = 0
        message = "Evaluation in progress..."
        
        if "Selected reasoning rule:" in logs:  
            progress = 10
        if "Selected domain:" in logs:  
            progress = 20
        if "Wikidata entities saved to" in logs:  
            progress = 30
        if "transitive_entity_extract.py" in logs: 
            progress = 40
        if "transitive_pl_build.py" in logs: 
            progress = 50
        if "wiki_pl_build.py" in logs:  
            progress = 60
        if "rule_generation.py" in logs:  
            progress = 70
        if "prolog_inference.py" in logs:  
            progress = 80
        if "question_generation.py" in logs:  
            progress = 90
        if "Results saved to:" in logs: 
            progress = 100
            message = "Evaluation completed" 
        elif "Error:" in logs:  
            progress = 100
            message = "Evaluation error"  
        
        return jsonify({"status": "processing", "progress": progress, "message": message, "log_file": log_file}), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Progress get failure: {str(e)}"}), 500
    
@app.route('/api/results/<log_file>')
def get_results(log_file):
    """Retrieve evaluation results (read from 'results' folder via log file)"""
    log_full_path = os.path.join("logs", log_file)
    
    if not os.path.exists(log_full_path):
        return jsonify({
            "status": "error", 
            "message": f"Log file not found: {log_full_path}"
        }), 404
    
    try:
        result_file_path = None
        with open(log_full_path, 'r', encoding='utf-8') as log_f:
            for line in log_f:
                
                if "Results saved to:" in line:  
                    result_file_path = line.split("Results saved to: ")[1].strip()
                    break
        
        if not result_file_path:
            return jsonify({
                "status": "processing", 
                "message": "Evaluation incomplete. Result file path not found."
            }), 202
        
        if not os.path.exists(result_file_path):
            return jsonify({
                "status": "error", 
                "message": f"Result file not found: {result_file_path}"
            }), 404
        
        with open(result_file_path, 'r', encoding='utf-8') as result_f:
            evaluation_result = json.load(result_f)
        
        return jsonify({
            "status": "success",
            "result_file": result_file_path,
            "evaluation_result": evaluation_result
        })
    
    except json.JSONDecodeError:
        return jsonify({
            "status": "error", 
            "message": f"Invalid result file format: {result_file_path}"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to retrieve results: {str(e)}"
        }), 500


@app.route('/api/evaluate_rag', methods=['POST'])
def evaluate_rag():
    """Handle RAG-enhanced evaluation request"""
    try:
        data = request.json
        rule = data.get('rule')
        domain = data.get('domain')
        model_name = data.get('model_name')
        top_k = data.get('top_k', 3)
        
        if not domain or not model_name:
            return jsonify({"status": "error", "message": "Missing required parameters"}), 400
        
        # Generate unique log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"rag_evaluation_{timestamp}.log"
        log_file = os.path.join("logs", log_filename)
        
        # Start RAG evaluation thread
        thread = threading.Thread(
            target=run_rag_evaluation,
            args=(rule, domain, model_name, top_k, log_file)
        )
        thread.daemon = True
        thread.start()
        
        # Track process status
        process_id = timestamp
        rag_processes[process_id] = {
            "status": "running",
            "start_time": timestamp,
            "log_file": log_filename
        }
        
        return jsonify({
            "status": "processing",
            "message": "RAG evaluation started. Check logs for progress.",
            "process_id": process_id,
            "log_file": log_filename
        }), 200
        
    except Exception as e:
        traceback.print_exc()  # Print detailed error stack
        return jsonify({"status": "error", "message": f"Error processing request: {str(e)}"}), 500

def evaluate_rag_model(model_name, domain, test_questions, top_k):
    """Core logic for RAG evaluation with performance metrics tracking"""
    global model, index, cleaned_abstracts  # Use pre-loaded global variables
    
    # 1. Initialize generative model (select based on model_name)
    try:
        if model_name == "qwen":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B", trust_remote_code=True)
            generate_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-1_8B", 
                trust_remote_code=True,
                # device_map="auto" 
            )
        elif model_name == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
            generate_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        else:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            generate_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    
    except ValueError as e:
        print(f"Failed to load model: {e}")
        return 0.0, 0.0, {}, {}  # 返回空的性能指标
    
    device = get_device()
    generator = pipeline("text-generation", model=generate_model, tokenizer=tokenizer, device=device)
    
    # 2. Initialize counters and total samples
    basic_correct = 0
    rag_correct = 0
    total = len(test_questions)
    print(f"Starting evaluation with {total} samples")  
    
    # 性能指标收集
    basic_metrics = {
        "response_time": [],
        "memory_usage": [],
        "gpu_utilization": []
    }
    
    rag_metrics = {
        "response_time": [],
        "memory_usage": [],
        "gpu_utilization": []
    }
    
    # 3. Iterate through test questions with progress tracking
    for i, question_item in enumerate(test_questions):
        # Print progress (every 10 samples or last sample)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} samples")
        
        # Extract question and reference answer
        question = question_item["question"]
        raw_reference = question_item["answer"].strip().lower()
        # Clean reference answer (remove punctuation, keep letters only)
        reference_answer = re.sub(r'[^a-z]', '', raw_reference)
        
        # 基础模型评估（带性能指标）
        basic_start_time = time.time()
        basic_start_memory = get_memory_usage()
        basic_start_gpu = get_gpu_utilization() if torch.cuda.is_available() else 0
        
        basic_prompt = f"Answer with 'yes' or 'no': {question}"
        basic_output = generator(
            basic_prompt,
            max_new_tokens=100,
            temperature=0.1,
            truncation=True,
            do_sample=True
        )

        basic_generated_text = basic_output[0]["generated_text"].strip().lower()

        if basic_generated_text.startswith(basic_prompt):
            basic_answer = basic_generated_text[len(basic_prompt):].strip()
        else:
            basic_answer = basic_generated_text
        
        # 记录基础模型性能指标
        basic_end_time = time.time()
        basic_end_memory = get_memory_usage()
        basic_end_gpu = get_gpu_utilization() if torch.cuda.is_available() else 0
        
        # 只记录大于0的值
        basic_time = basic_end_time - basic_start_time
        basic_mem = basic_end_memory - basic_start_memory
        basic_gpu = basic_end_gpu - basic_start_gpu
        
        if basic_time > 0:
            basic_metrics["response_time"].append(basic_time)
        if basic_mem > 0:
            basic_metrics["memory_usage"].append(basic_mem)
        if basic_gpu > 0:
            basic_metrics["gpu_utilization"].append(basic_gpu)
        
        # Use classifier to predict basic model answer
        basic_prediction, basic_confidence = answer_classifier.predict(basic_answer)
        # Check correctness
        if basic_prediction == reference_answer:
            basic_correct += 1
        
        # RAG模型评估（带性能指标）
        rag_start_time = time.time()
        rag_start_memory = get_memory_usage()
        rag_start_gpu = get_gpu_utilization() if torch.cuda.is_available() else 0
        
        # Retrieve relevant documents
        query_embedding = model.encode(question, convert_to_tensor=True)
        query_embedding_2d = np.expand_dims(query_embedding.cpu().numpy(), axis=0)
        distances, indices = index.search(query_embedding_2d, top_k)
        retrieved_docs = [cleaned_abstracts[i] for i in indices[0] if i < len(cleaned_abstracts)]
        
        # Build context-aware prompt
        context = "\n".join(retrieved_docs)
        rag_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with 'yes' or 'no':"
        
        # Generate RAG model answer
        rag_output = generator(
            rag_prompt,
            max_new_tokens=150,
            temperature=0.1,
            truncation=True,
            do_sample=True
        )
        rag_generated_text = rag_output[0]["generated_text"].strip().lower()

        target_phrase = "answer with 'yes' or 'no':"
        target_pos = rag_generated_text.find(target_phrase)
        if target_pos != -1:
            rag_answer = rag_generated_text[target_pos + len(target_phrase):].strip()
        elif rag_generated_text.startswith(rag_prompt):
            
            rag_answer = rag_generated_text[len(rag_prompt):].strip()
        else:
            rag_answer = rag_generated_text
        
        # 记录RAG模型性能指标
        rag_end_time = time.time()
        rag_end_memory = get_memory_usage()
        rag_end_gpu = get_gpu_utilization() if torch.cuda.is_available() else 0
        
        # 只记录大于0的值
        rag_time = rag_end_time - rag_start_time
        rag_mem = rag_end_memory - rag_start_memory
        rag_gpu = rag_end_gpu - rag_start_gpu
        
        if rag_time > 0:
            rag_metrics["response_time"].append(rag_time)
        if rag_mem > 0:
            rag_metrics["memory_usage"].append(rag_mem)
        if rag_gpu > 0:
            rag_metrics["gpu_utilization"].append(rag_gpu)
        
        # Use classifier to predict RAG model answer
        rag_prediction, rag_confidence = answer_classifier.predict(rag_answer)
        # Check correctness
        if rag_prediction == reference_answer:
            rag_correct += 1
        
        # 7. Print detailed debug information for first 10 samples
        if i < 10:
            print(f"\n===== Detailed Analysis for Sample {i + 1} =====")
            print(f"Question: {question}")
            print(f"Reference Answer: {raw_reference} → Cleaned: {reference_answer}")
            
            print(f"\nBase Model (No RAG):")
            print(f"  Raw Output: {basic_answer}")
            print(f"  Semantic Prediction: {basic_prediction} (Confidence: {basic_confidence:.4f})")
            print(f"  Correctness: {'Correct' if basic_prediction == reference_answer else 'Incorrect'}")
            print(f"  Performance: Time={basic_time:.4f}s, "
                  f"Memory={basic_mem:.4f}MB, "
                  f"GPU={basic_gpu:.2f}%")
            
            print(f"\nRAG-Enhanced Model:")
            print(f"  Raw Output: {rag_answer}")
            print(f"  Semantic Prediction: {rag_prediction} (Confidence: {rag_confidence:.4f})")
            print(f"  Correctness: {'Correct' if rag_prediction == reference_answer else 'Incorrect'}")
            print(f"  Performance: Time={rag_time:.4f}s, "
                  f"Memory={rag_mem:.4f}MB, "
                  f"GPU={rag_gpu:.2f}%")
            print(f"  Retrieved Documents: {len(retrieved_docs)}")
            print("=" * 70)  # Separator line
    
    # Calculate accuracy
    basic_accuracy = basic_correct / total if total > 0 else 0
    rag_accuracy = rag_correct / total if total > 0 else 0
    
    # 计算性能指标平均值（只考虑大于0的值）
    basic_avg_metrics = {
        "response_time": sum(basic_metrics["response_time"]) / len(basic_metrics["response_time"]) if basic_metrics["response_time"] else 0,
        "memory_usage": sum(basic_metrics["memory_usage"]) / len(basic_metrics["memory_usage"]) if basic_metrics["memory_usage"] else 0,
        "gpu_utilization": sum(basic_metrics["gpu_utilization"]) / len(basic_metrics["gpu_utilization"]) if basic_metrics["gpu_utilization"] else 0
    }
    
    rag_avg_metrics = {
        "response_time": sum(rag_metrics["response_time"]) / len(rag_metrics["response_time"]) if rag_metrics["response_time"] else 0,
        "memory_usage": sum(rag_metrics["memory_usage"]) / len(rag_metrics["memory_usage"]) if rag_metrics["memory_usage"] else 0,
        "gpu_utilization": sum(rag_metrics["gpu_utilization"]) / len(rag_metrics["gpu_utilization"]) if rag_metrics["gpu_utilization"] else 0
    }
    
    # 计算增幅（RAG/基础）
    performance_ratios = {
        "response_time": rag_avg_metrics["response_time"] / basic_avg_metrics["response_time"] if basic_avg_metrics["response_time"] > 0 else 0,
        "memory_usage": rag_avg_metrics["memory_usage"] / basic_avg_metrics["memory_usage"] if basic_avg_metrics["memory_usage"] > 0 else 0,
        "gpu_utilization": rag_avg_metrics["gpu_utilization"] / basic_avg_metrics["gpu_utilization"] if basic_avg_metrics["gpu_utilization"] > 0 else 0
    }
    
    print("\n===== Performance Metrics Summary =====")
    print(f"Base Model:")
    print(f"  Avg Response Time: {basic_avg_metrics['response_time']:.4f}s (based on {len(basic_metrics['response_time'])}/{total} samples)")
    print(f"  Avg Memory Usage: {basic_avg_metrics['memory_usage']:.4f}MB (based on {len(basic_metrics['memory_usage'])}/{total} samples)")
    print(f"  Avg GPU Utilization: {basic_avg_metrics['gpu_utilization']:.2f}% (based on {len(basic_metrics['gpu_utilization'])}/{total} samples)")
    
    print(f"\nRAG Model:")
    print(f"  Avg Response Time: {rag_avg_metrics['response_time']:.4f}s ({performance_ratios['response_time']:.2f}x base)")
    print(f"  Avg Memory Usage: {rag_avg_metrics['memory_usage']:.4f}MB ({performance_ratios['memory_usage']:.2f}x base)")
    print(f"  Avg GPU Utilization: {rag_avg_metrics['gpu_utilization']:.2f}% ({performance_ratios['gpu_utilization']:.2f}x base)")
    
    return basic_accuracy, rag_accuracy, basic_avg_metrics, rag_avg_metrics, performance_ratios

def run_rag_evaluation(rule, domain, model_name, top_k, log_file):
    """Background task for RAG evaluation with performance metrics"""
    global current_rag_log
    current_rag_log = log_file
    
    try:
        # Redirect standard output to log file
        with open(log_file, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print(f"Starting RAG evaluation - Domain: {domain}, Model: {model_name}, Retrieved documents: {top_k}")
                print("=" * 50)
                
                # 1. Preprocess domain data (if not already processed)
                print(f"[1/5] Checking and preprocessing {domain} domain data...")
                if not os.path.exists(f"RAG_cleaned/cleaned_{domain}.txt"):
                    print(f"Data not preprocessed, starting rag_preprocess({domain})...")
                    rag_preprocess(domain)
                else:
                    print(f"Data already preprocessed, skipping this step")

                # 2. Build vector index (if not already built)
                print(f"[2/5] Checking and building {domain} domain vector index...")
                if not os.path.exists(f"knowledge_base/{domain}_index.faiss"):
                    print(f"Index not built, starting embed_faiss({domain})...")
                    build_faiss_index(domain)
                else:
                    print(f"Index already exists, loading...")
                
                # 3. Load evaluation dataset
                print(f"[3/5] Loading {domain} domain evaluation dataset...")
                dataset_path = f"data/generated/{rule}/{domain}_qa.json"  # Select directory based on rule
                if not os.path.exists(dataset_path):
                    print(f"Error: Dataset does not exist - {dataset_path}")
                    return
                
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    test_dataset = json.load(f)
                
                # 4. Initialize model and retriever
                print(f"[4/5] Initializing model and retriever...")
                global model, index, cleaned_abstracts
                
                # Load embedding model
                model = load_embedding_model()
                print(f"Embedding model device: {model.device if hasattr(model, 'device') else 'CPU'}") 

                # Load FAISS index
                index = load_faiss_index(domain)
                
                # Load preprocessed documents
                with open(f"RAG_cleaned/cleaned_{domain}.txt", 'r', encoding='utf-8') as f:
                    cleaned_abstracts = [line.strip() for line in f.readlines() if line.strip()]
                
                # 5. Execute RAG evaluation
                print(f"[5/5] Starting RAG evaluation... ({len(test_dataset)} samples)")
                
                # Call the modified evaluate_rag_model function
                basic_score, rag_score, basic_metrics, rag_metrics, perf_ratios = evaluate_rag_model(
                    model_name=model_name,
                    domain=domain,
                    test_questions=test_dataset,
                    top_k=top_k
                )
                
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                # 6. Save evaluation results
                result_path = f"rag_results/rag_{domain}_{model_name}_{timestamp}.json"
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                
                result_data = {
                    "rule": rule,
                    "domain": domain,
                    "model_name": model_name,
                    "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "metrics": {
                        "basic_accuracy": basic_score,
                        "rag_accuracy": rag_score,
                        "improvement": rag_score - basic_score,
                        "performance": {
                            "basic": basic_metrics,
                            "rag": rag_metrics,
                            "ratios": perf_ratios
                        }
                    },
                    "parameters": {
                        "top_k": top_k,
                        "dataset_size": len(test_dataset)
                    }
                }
                
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                print(f"\nRAG evaluation completed! Results saved to: {result_path}")
                print(f"Basic accuracy: {basic_score:.4f}")
                print(f"RAG-augmented accuracy: {rag_score:.4f}")
                print(f"Performance improvement: {rag_score - basic_score:.4f}")
                
                # Update process status
                process_id = log_file.split('_')[-1].split('.')[0]
                if process_id in rag_processes:
                    rag_processes[process_id]["status"] = "completed"
                    rag_processes[process_id]["result_file"] = result_path
    
    except Exception as e:
        # Log error to file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nError: {str(e)}\n")
            f.write(traceback.format_exc())
        
        # Update process status
        process_id = log_file.split('_')[-1].split('.')[0]
        if process_id in rag_processes:
            rag_processes[process_id]["status"] = "error"
            rag_processes[process_id]["error_message"] = str(e)


@app.route('/api/rag_progress/<log_file>')
def get_rag_progress(log_file):
    process_id = None
    for pid, process in rag_processes.items():
        if process.get("log_file") == log_file:
            process_id = pid
            break
    if not process_id:
        return jsonify({"status": "error", "message": "Evaluation process not found"}), 404
    
    process = rag_processes[process_id]
    log_file = process.get("log_file")
    
    if not log_file:
        return jsonify({"status": "error", "message": "Log file does not exist"}), 404
    
    
    log_path = os.path.join("logs", log_file)
    if not os.path.exists(log_path):
        return jsonify({"status": "error", "message": "Log file does not exist"}), 404
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = f.read()
        
        progress = 0
        message = "RAG evaluation in progress..."
        
        
        if "RAG evaluation completed" in logs:  
            progress = 100
            message = "RAG evaluation completed"
        elif "Error:" in logs:
            progress = 100
            message = f"RAG evaluation error: {process.get('error_message', 'unknown error')}"
        elif "[5/5]" in logs:
            progress = 80
            message = "Executing RAG evaluation..."
        elif "[4/5]" in logs:
            progress = 60
            message = "Initializing model and retriever..."
        elif "[3/5]" in logs:
            progress = 40
            message = "Loading evaluation dataset..."
        elif "[2/5]" in logs:
            progress = 30
            message = "Checking and building vector index..."
        elif "[1/5]" in logs:
            progress = 20
            message = "Checking and preprocessing data..."
        elif "Starting RAG evaluation" in logs:  
            progress = 10
            message = "RAG evaluation started..."
        
        return jsonify({
            "status": process["status"],
            "progress": progress,
            "message": message,
            "log_file": log_file,
            "start_time": process.get("start_time")
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to get progress: {str(e)}"}), 500

@app.route('/api/rag_results/<log_file>')
def get_rag_results(log_file):
    """Retrieve RAG evaluation results"""
    # Construct full path from log file name
    log_path = os.path.join("logs", log_file)
    
    if not os.path.exists(log_path):
        return jsonify({"status": "error", "message": "Log file not found"}), 404
    
    try:
        # Extract result file path from log
        result_file = None
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "RAG evaluation completed! Results saved to:" in line:  
                    result_file = line.split("RAG evaluation completed! Results saved to: ")[1].strip()
                    break
        

        if not result_file:
           
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = f.read()
            
            if "RAG evaluation completed" in logs:  # Match English log output
                return jsonify({"status": "error", "message": "Failed to extract result file path from logs"}), 500
            else:
                # Evaluation in progress, return current status with full progress checks
                progress = 0
                message = "RAG evaluation in progress..."
                
                if "[5/5]" in logs:
                    progress = 80
                    message = "Executing RAG evaluation..."
                elif "[4/5]" in logs:
                    progress = 60
                    message = "Initializing model and retriever..."
                elif "[3/5]" in logs:
                    progress = 40
                    message = "Loading evaluation dataset..."
                elif "[2/5]" in logs:
                    progress = 30
                    message = "Checking and building vector index..."
                elif "[1/5]" in logs:
                    progress = 20
                    message = "Checking and preprocessing data..."
                elif "Starting RAG evaluation" in logs:
                    progress = 10
                    message = "RAG evaluation started..."
                
                return jsonify({
                    "status": "processing",
                    "progress": progress,
                    "message": message,
                    "log_file": log_file
                }), 202
        
        # Check if result file exists
        if not os.path.exists(result_file):
            # Try relative path if absolute fails
            relative_result_file = os.path.join(os.getcwd(), result_file)
            if os.path.exists(relative_result_file):
                result_file = relative_result_file
            else:
                return jsonify({"status": "error", "message": f"Result file not found: {result_file}"}), 404
        
        # Read result file
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Ensure new metrics are present, default to 0 if not
        metrics = results.get('metrics', {})
        metrics.setdefault('base_avg_response_time', 0)
        metrics.setdefault('base_avg_memory_usage', 0)
        metrics.setdefault('base_avg_gpu_utilization', 0)
        metrics.setdefault('rag_avg_response_time', 0)
        metrics.setdefault('rag_avg_memory_usage', 0)
        metrics.setdefault('rag_avg_gpu_utilization', 0)
        metrics.setdefault('rag_avg_response_time_ratio', 0)
        metrics.setdefault('rag_avg_memory_usage_ratio', 0)
        metrics.setdefault('rag_avg_gpu_utilization_ratio', 0)
        results['metrics'] = metrics

        return jsonify({
            "status": "success",
            "results": results,
            "result_file": result_file
        }), 200
        
    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": f"Invalid result file format, cannot parse: {result_file}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to retrieve results: {str(e)}"}), 500


@app.route('/api/get_rag_materials/<domain>', methods=['GET'])
def get_rag_materials(domain):
    try:
        data_file_path = f"RAG_cleaned/cleaned_{domain}.txt"
        if os.path.exists(data_file_path):
            with open(data_file_path, 'r', encoding='utf-8') as f:
                
                all_lines = [line.strip() for line in f.readlines() if line.strip()]
                
                
                sample_size = min(10, len(all_lines))
                random_lines = random.sample(all_lines, sample_size) if all_lines else []
                
                
                data_content = '\n'.join(random_lines) + ('\n...' if len(all_lines) > sample_size else '')
        else:
            data_content = "No preprocessed data found in RAG_cleaned."

        index_file_path = f"knowledge_base/{domain}_index.faiss"
        index_content = "Vector index file cannot be directly displayed. " \
                        f"File exists: {os.path.exists(index_file_path)}"

        return jsonify({
            "status": "success",
            "data_content": data_content,
            "index_content": index_content
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to get RAG materials: {str(e)}"}), 500
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/api/upload_rag_materials', methods=['POST'])
def upload_rag_materials():
    try:
        domain = request.form.get('domain')
        if not domain:
            return jsonify({"status": "error", "message": "Domain is required"}), 400

        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], domain)
        os.makedirs(upload_dir, exist_ok=True)

        if 'pdf_files' not in request.files:
            return jsonify({"status": "error", "message": "No files uploaded"}), 400

        files = request.files.getlist('pdf_files')
        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, filename)
                
                if os.path.exists(file_path):
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_{int(time.time())}{ext}"
                    file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                uploaded_files.append(filename)

        if not uploaded_files:
            return jsonify({"status": "error", "message": "No valid PDF files uploaded"}), 400

        
        rag_preprocess(domain)  
        build_faiss_index(domain)  

        return jsonify({
            "status": "success",
            "message": f"Successfully uploaded {len(uploaded_files)} files. RAG materials processed and stored!"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Upload failed: {str(e)}"}), 500

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists("logs"):
        os.makedirs("logs")
    app.run(debug=True, port=5000)


