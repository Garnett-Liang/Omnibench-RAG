# src/backend/workflow.py
import os
import re
import io
import json
import torch
import requests  
import warnings
from tqdm import tqdm
import psutil
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import time
import threading
import sys
from werkzeug.utils import secure_filename
import random
import traceback
from datetime import datetime
import numpy as np
from datasets import load_dataset, Dataset
from difflib import SequenceMatcher
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
from work_models.data_preprocess import rag_preprocess
from work_models.embed_faiss import build_faiss_index, load_embedding_model, load_faiss_index
from work_models.get_wiki_cat_id import (
    get_category_pages, save_links_to_file, 
    get_category_members, extract_entity_pages, save_entity_links_to_file
)
from work_models.transitive_entity_extract import (
    get_entity_info, get_predicate_labels, replace_predicates_with_labels
)
from work_models.transitive_pl_build import (
    replace_special_characters, save_to_pl_file, process_prolog_file
)
from work_models.wiki_pl_build import (
    get_related_entity_list, get_prop_list, 
    replace_special_characters1, save_to_pl_file1, process_prolog_file1
)
from work_models.rule_generation import RuleGenerator
from work_models.prolog_inference import (
    normalize_path, safe_consult, get_wikipedia_summary, 
    replace_special_characters2, is_q_followed_by_digits, 
    negation_prolog_inference, composite_prolog_inference, inverse_prolog_inference
)
from work_models.question_generation import inverse_template, negation_template, composite_template
from evaluate import loadset, get_wikidata_id_from_wikipedia_url
from dynamic_dataset import generate_dynamic_dataset

# global
current_log_file = None
rag_processes = {}  
current_rag_log = None


class BinaryAnswerClassifier:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.labels = {0: "no", 1: "yes"}  
    
    def predict(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1)
        confidence = probabilities[0][prediction].item()
        return self.labels[prediction], confidence
    
answer_classifier = BinaryAnswerClassifier()


def extract_valid_answer(full_output: str, prompt: str) -> str:
    prompt_end_idx = full_output.find(prompt) + len(prompt)
    valid_answer = full_output[prompt_end_idx:].strip()
    return valid_answer if valid_answer else full_output

def get_memory_usage(samples: int = 3, interval: float = 0.2) -> float:
    process = psutil.Process(os.getpid())
    peak = 0
    for _ in range(samples):
        mem = process.memory_info().rss / (1024 ** 2)
        peak = max(peak, mem)
        time.sleep(interval)
    return round(peak, 4)


def get_gpu_utilization(samples: int = 3, interval: float = 0.2) -> float:

    try:

        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_util = float(result.stdout.strip().split('\n')[0])
            return round(gpu_util, 2)
        else:
            return 0.0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, IndexError) as e:
        return 0.0
    except Exception:
        return 0.0

def get_device():
    """统一设备检测函数"""
    if torch.cuda.is_available():
        return "cuda"  # 返回字符串而不是数字
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def generate_answers(model_name, questions):
    """
    改进版：记录真实峰值内存与稳定GPU利用率，消除负值与波动。
    """
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=False, torch_dtype=torch.float32)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, is_decoder=True)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

    answers, response_times, memory_peaks, gpu_utils = [], [], [], []

    for question in questions:
        start_time = time.time()
        mem_peak_before = get_memory_usage()
        gpu_avg_before = get_gpu_utilization()

        # 生成答案
        response = text_generator(question, max_new_tokens=300, num_return_sequences=1)
        
        # 再次采样峰值
        mem_peak_after = get_memory_usage()
        gpu_avg_after = get_gpu_utilization()

        response_time = time.time() - start_time
        response_times.append(response_time)
        memory_peaks.append(max(mem_peak_before, mem_peak_after))
        gpu_utils.append(max(gpu_avg_before, gpu_avg_after))

        # 解析结果
        generated_text = response[0]['generated_text'].strip()
        answer = generated_text[len(question):].strip() if generated_text.startswith(question) else generated_text
        answers.append(answer if answer else "No answer")

    return answers, response_times, memory_peaks, gpu_utils


def generate_answers_with_api(api_config, questions):

    import requests
    
    answers = []
    response_times = []
    memory_peaks = []  # 改为记录内存峰值而不是增量
    gpu_utilizations = []
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    if api_config.get('api_key'):
        headers['Authorization'] = f'Bearer {api_config["api_key"]}'
    
    # 自动转换 Hugging Face 旧端点为新路由端点
    api_endpoint = api_config['api_endpoint']
    if 'api-inference.huggingface.co/models/' in api_endpoint:
        api_endpoint = 'https://router.huggingface.co/v1/chat/completions'
        print(f"Converted Hugging Face endpoint to: {api_endpoint}")
    
    # 格式化提示模板
    prompt_template = api_config.get('api_prompt_template', 'Answer with "yes" or "no": {question}')
    
    for question in questions:
        start_time = time.time()
        start_memory = get_memory_usage()
        start_gpu = get_gpu_utilization()
        
        # 记录初始内存作为基准
        current_memory_peak = start_memory
        
        # 格式化提示
        prompt = prompt_template.format(question=question)
        
        payload = {
            'model': api_config['api_model_name'],
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that answers questions with only "yes" or "no". Be concise and accurate.'
                },
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': api_config.get('api_max_tokens', 1000),
            'temperature': 0.0
        }
        
        try:
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0]['message']['content'].strip()
            else:
                generated_text = "No answer"
                
        except Exception as e:
            print(f"API call failed: {e}")
            generated_text = "No answer"
        
        # 在API调用后记录内存峰值
        end_time = time.time()
        end_memory = get_memory_usage()
        current_memory_peak = max(current_memory_peak, end_memory)
        end_gpu = get_gpu_utilization()
        
        response_time = end_time - start_time
        response_times.append(response_time)
        memory_peaks.append(current_memory_peak)  # 记录峰值而不是增量
        gpu_utilizations.append((start_gpu + end_gpu) / 2)
        
        answers.append(generated_text)
    
    return answers, response_times, memory_peaks, gpu_utilizations


def evaluate_model_with_api(api_config, questions, standard_answers):

    model_answers, response_times, memory_peaks, gpu_utilizations = generate_answers_with_api(api_config, questions)

    basic_correct = 0
    total = len(questions)
    total_response_time = sum(response_times)
    average_response_time = total_response_time / total if total > 0 else 0

    # 计算内存峰值平均值（不需要过滤，因为峰值都是正值）
    average_memory_peak = sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0
    average_gpu_utilization = sum(gpu_utilizations) / total if total > 0 else 0


    results = {
        "model_name": f"DeepSeek API ({api_config['api_model_name']})",
        "api_config": api_config,
        "questions": [],
        "basic_accuracy": 0.0,
        "average_response_time": average_response_time,
        "average_memory_usage": average_memory_peak,
        "average_gpu_utilization": average_gpu_utilization,
    }

    for i, (question, model_answer, standard_answer, response_time, memory_peak, gpu_utilization) in enumerate(
            zip(questions, model_answers, standard_answers, response_times, memory_peaks, gpu_utilizations)):
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
            "memory_usage": memory_peak,  # 保存峰值而不是增量
            "gpu_utilization": gpu_utilization
        })

    basic_accuracy = (basic_correct / total) * 100 if total > 0 else 0
    results["basic_accuracy"] = basic_accuracy

    return json.dumps(results, indent=4, ensure_ascii=False)


def evaluate_model(model_name, questions, standard_answers):
    device = get_device()
    print(f"Using device: {device}")
    
    model_answers, response_times, memory_peaks, gpu_utilizations = generate_answers(model_name, questions)

    basic_correct = 0
    total = len(questions)
    total_response_time = sum(response_times)
    average_response_time = total_response_time / total if total > 0 else 0

    # 计算内存峰值平均值（不需要过滤，因为峰值都是正值）
    average_memory_peak = sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0
    average_gpu_utilization = sum(gpu_utilizations) / total if total > 0 else 0


    results = {
        "model_name": model_name,
        "questions": [],
        "basic_accuracy": 0.0,
        "average_response_time": average_response_time,
        "average_memory_usage": average_memory_peak,
        "average_gpu_utilization": average_gpu_utilization,
    }

    for i, (question, model_answer, standard_answer, response_time, memory_peak, gpu_utilization) in enumerate(
            zip(questions, model_answers, standard_answers, response_times, memory_peaks, gpu_utilizations)):
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
            "memory_usage": memory_peak,  # 保存峰值而不是增量
            "gpu_utilization": gpu_utilization
        })

    basic_accuracy = (basic_correct / total) * 100 if total > 0 else 0
    results["basic_accuracy"] = basic_accuracy

    return json.dumps(results, indent=4, ensure_ascii=False)


def run_evaluation(rule_choice, domain_choice, model_choice, log_file, dataset_source, custom_api_config=None):
    global current_log_file
    current_log_file = log_file
    
    # 核心：获取当前脚本绝对路径（固定路径基准）
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    results_root = os.path.abspath(os.path.join(current_script_dir, '..', 'experiments', 'results', 'results'))
    os.makedirs(results_root, exist_ok=True)  # 确保结果目录存在
    
    with open(log_file, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):  # 所有打印重定向到日志文件
            try:
                # 1. 设备检测（CPU/GPU）
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Using device: {device}")
                
                # 2. 推理规则与领域映射
                rule_map = {"1": "inverse", "2": "negation", "3": "composite"}
                reasoning_type = rule_map.get(rule_choice, "inverse")
                print(f"Selected reasoning rule: {reasoning_type}")
                
                domains = ["geography", "history", "health", "mathematics", "nature", 
                           "people", "society", "technology", "culture"]
                domain = domains[int(domain_choice) - 1] if (
                    domain_choice.isdigit() and 1 <= int(domain_choice) <= 9
                ) else "geography"
                print(f"Selected domain: {domain}")
                print(f"Processing domain: {domain}")
                
                questions, standard_answers = [], []
                current_file_path = ""  # 关键：初始化路径变量，每个文件操作前更新
                
                # -------------------------- 数据集加载/生成逻辑 --------------------------
                if dataset_source == "existing":
                    print("Using reproducible dataset...")
                    # 加载现有数据集路径
                    dataset_path = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'dataset', 'generated', reasoning_type,
                        f'{domain}_qa.json'
                    ))
                    current_file_path = dataset_path
                    questions, standard_answers = loadset(reasoning_type, domain, current_file_path)
                    print(f"Loaded {len(questions)} questions from {current_file_path}")
                else:  
                    print("Using dynamically generated dataset...")
                    
                    # 检查dynamic文件夹下是否存在数据集
                    dynamic_dataset_path = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'dataset', 'dynamic', reasoning_type,
                        f'{domain}_qa.json'
                    ))
                    
                    if os.path.exists(dynamic_dataset_path):
                        print(f"Found existing dynamic dataset: {dynamic_dataset_path}")
                        current_file_path = dynamic_dataset_path
                        questions, standard_answers = loadset(reasoning_type, domain, current_file_path)
                        print(f"Loaded {len(questions)} questions from existing dynamic dataset")
                    else:
                        print("Dynamic dataset not found, generating new dataset...")
                        

                        qa_file_path = generate_dynamic_dataset(rule_choice, domain_choice, current_script_dir)
                        

                        questions, standard_answers = loadset(reasoning_type, domain, qa_file_path)
                        print(f"Loaded {len(questions)} newly generated questions")

                # -------------------------- 模型评估逻辑 --------------------------
                if not questions or not standard_answers:
                    raise ValueError("No valid questions/answers for evaluation")
                
                # 检查是否使用自定义API
                if model_choice == "api" and custom_api_config:
                    print(f"Using DeepSeek API: {custom_api_config['api_model_name']}")
                    json_result = evaluate_model_with_api(custom_api_config, questions, standard_answers)
                else:
                    # 传统模型映射
                    model_map = {"1": "Qwen/Qwen-1_8B", "2": "gpt2-medium", "3": "EleutherAI/gpt-neo-125M", "5": "facebook/opt-1.3b"}
                    model_name = model_map.get(model_choice)
                    if not model_name:
                        raise ValueError(f"Invalid model choice: {model_choice}")
                    
                    print(f"Evaluating with model: {model_name}")
                    json_result = evaluate_model(model_name, questions, standard_answers)
                
                # 保存评估结果
                timestamp = int(time.time())
                if model_choice == "api":
                    result_filename = f"{reasoning_type}_{domain}_custom_api_{timestamp}.json"
                else:
                    model_map = {"1": "Qwen/Qwen-1_8B", "2": "gpt2-medium", "3": "EleutherAI/gpt-neo-125M", "5": "facebook/opt-1.3b"}
                    model_name = model_map.get(model_choice, "unknown")
                    result_filename = f"{reasoning_type}_{domain}_{model_name.replace('/', '_')}_{timestamp}.json"
                
                result_path = os.path.join(results_root, result_filename)
                
                with open(result_path, "w", encoding="utf-8") as f_result:
                    f_result.write(json_result)
                
                print(f"Results saved to: {result_path}")  
                return {
                    "status": "complete", 
                    "message": "Evaluation successful", 
                    "evaluation_result": json.loads(json_result), 
                    "sample_questions": questions[:5],
                    "sample_answers": standard_answers[:5],
                    "result_path": result_path  
                }

            # -------------------------- 异常捕获 --------------------------
            except json.JSONDecodeError as e:
                error_msg = f"JSON解析错误\n问题文件: {current_file_path}\n错误详情: {str(e)}"
                print(error_msg)
                return {"status": "error", "message": error_msg}
            
            except FileNotFoundError as e:
                error_msg = f"文件未找到\n目标文件: {current_file_path}\n错误详情: {str(e)}"
                print(error_msg)
                return {"status": "error", "message": error_msg}
            
            except Exception as e:
                error_msg = f"评估失败\n当前处理文件: {current_file_path}\n错误详情: {str(e)}"
                print(error_msg)
                return {"status": "error", "message": error_msg}


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
        elif model_name == "gptneo":
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            generate_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
        elif model_name == "opt":
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b",trust_remote_code=True)
            generate_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

    except ValueError as e:
        print(f"Failed to load model: {e}")
        return 0.0, 0.0, {}, {}  # 返回空的性能指标
    
    device = get_device()
    generator = pipeline("text-generation", model=generate_model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
    
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
        "memory_usage": [],  # 存储检索环节开销 + 最终增量的总和
        "retrieval_memory": [],  # 单独记录检索环节内存开销
        "final_increment": [],   # 单独记录最终增量
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
        
        # 基础模型评估（保持原有计算逻辑）
        basic_start_time = time.time()
        basic_start_memory = get_memory_usage()
        basic_start_gpu = get_gpu_utilization() if get_device() == "cuda" else 0
        
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
        basic_end_gpu = get_gpu_utilization() if get_device() == "cuda" else 0
        
        basic_time = basic_end_time - basic_start_time
        basic_mem = basic_end_memory - basic_start_memory  # 仅计算最终增量
        basic_gpu = basic_end_gpu - basic_start_gpu
        
        if basic_time > 0:
            basic_metrics["response_time"].append(basic_time)
        # 基础模型保留原有过滤逻辑（0-5MB，排除异常值）
        if 0 <= basic_mem <= 5:  
            basic_metrics["memory_usage"].append(basic_mem)  
        # GPU利用率仅统计大于0的值（排除无效0值）
        if basic_gpu > 0:  
            basic_metrics["gpu_utilization"].append(basic_gpu)  
        
        # Use classifier to predict basic model answer
        basic_prediction, basic_confidence = answer_classifier.predict(basic_answer)
        # Check correctness
        if basic_prediction == reference_answer:
            basic_correct += 1
        
        # RAG模型评估（修改内存计算逻辑）
        rag_start_time = time.time()
        rag_start_memory = get_memory_usage()  # RAG流程开始时的内存
        rag_start_gpu = get_gpu_utilization() if get_device() == "cuda" else 0
        
        # 记录检索前的内存（用于计算检索环节开销）
        pre_retrieval_memory = get_memory_usage()
        
        # Retrieve relevant documents（检索环节）
        query_embedding = model.encode(question, convert_to_tensor=True)
        query_embedding_2d = np.expand_dims(query_embedding.cpu().numpy(), axis=0)
        distances, indices = index.search(query_embedding_2d, top_k)
        retrieved_docs = [cleaned_abstracts[i] for i in indices[0] if i < len(cleaned_abstracts)]
        
        # 计算检索环节的内存开销（检索后 - 检索前）
        post_retrieval_memory = get_memory_usage()
        retrieval_memory = post_retrieval_memory - pre_retrieval_memory
        
        # Build context-aware prompt and generate answer（生成环节）
        context = "\n".join(retrieved_docs)
        rag_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with 'yes' or 'no':"
        
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
        rag_end_gpu = get_gpu_utilization() if get_device() == "cuda" else 0
        
        rag_time = rag_end_time - rag_start_time
        final_increment = rag_end_memory - rag_start_memory  # 最终增量（整个RAG流程）
        # RAG总内存开销 = 检索环节开销 + 最终增量
        rag_total_memory = retrieval_memory + final_increment
        rag_gpu = rag_end_gpu - rag_start_gpu
        
        if rag_time > 0:
            rag_metrics["response_time"].append(rag_time)
        # RAG内存过滤：总内存开销在0-5MB范围内（保留异常值过滤）
        if 0 <= rag_total_memory <= 5:  
            rag_metrics["memory_usage"].append(rag_total_memory)
        # 单独记录检索环节开销和最终增量（用于调试分析）
        rag_metrics["retrieval_memory"].append(retrieval_memory)
        rag_metrics["final_increment"].append(final_increment)
        # GPU利用率仅统计大于0的值（排除无效0值）
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
                  f"Memory={basic_mem:.4f}MB ({'included' if 0 <= basic_mem <= 5 else 'excluded'}), "
                  f"GPU={basic_gpu:.2f}% ({'included' if basic_gpu > 0 else 'excluded'})")
            
            print(f"\nRAG-Enhanced Model:")
            print(f"  Raw Output: {rag_answer}")
            print(f"  Semantic Prediction: {rag_prediction} (Confidence: {rag_confidence:.4f})")
            print(f"  Correctness: {'Correct' if rag_prediction == reference_answer else 'Incorrect'}")
            print(f"  Performance: Time={rag_time:.4f}s, "
                  f"Retrieval Memory={retrieval_memory:.4f}MB, "
                  f"Final Increment={final_increment:.4f}MB, "
                  f"Total Memory={rag_total_memory:.4f}MB ({'included' if 0 <= rag_total_memory <=5 else 'excluded'}), "
                  f"GPU={rag_gpu:.2f}% ({'included' if rag_gpu > 0 else 'excluded'})")
            print(f"  Retrieved Documents: {len(retrieved_docs)}")
            print("=" * 70)  # Separator line
    
    # Calculate accuracy
    basic_accuracy = basic_correct / total if total > 0 else 0
    rag_accuracy = rag_correct / total if total > 0 else 0
    
    # 计算性能指标平均值
    basic_avg_metrics = {
        "response_time": sum(basic_metrics["response_time"]) / len(basic_metrics["response_time"]) if basic_metrics["response_time"] else 0,
        "memory_usage": sum(basic_metrics["memory_usage"]) / len(basic_metrics["memory_usage"]) if basic_metrics["memory_usage"] else 0,
        "gpu_utilization": sum(basic_metrics["gpu_utilization"]) / len(basic_metrics["gpu_utilization"]) if basic_metrics["gpu_utilization"] else 0
    }
    
    rag_avg_metrics = {
        "response_time": sum(rag_metrics["response_time"]) / len(rag_metrics["response_time"]) if rag_metrics["response_time"] else 0,
        "memory_usage": sum(rag_metrics["memory_usage"]) / len(rag_metrics["memory_usage"]) if rag_metrics["memory_usage"] else 0,
        "retrieval_memory_avg": sum(rag_metrics["retrieval_memory"]) / len(rag_metrics["retrieval_memory"]) if rag_metrics["retrieval_memory"] else 0,
        "final_increment_avg": sum(rag_metrics["final_increment"]) / len(rag_metrics["final_increment"]) if rag_metrics["final_increment"] else 0,
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
    print(f"  Avg Memory Usage: {basic_avg_metrics['memory_usage']:.4f}MB (based on {len(basic_metrics['memory_usage'])}/{total} samples, 0-5MB only)")
    print(f"  Avg GPU Utilization: {basic_avg_metrics['gpu_utilization']:.2f}% (based on {len(basic_metrics['gpu_utilization'])}/{total} samples, >0 only)")
    
    print(f"\nRAG Model:")
    print(f"  Avg Response Time: {rag_avg_metrics['response_time']:.4f}s ({performance_ratios['response_time']:.2f}x base)")
    print(f"  Avg Total Memory: {rag_avg_metrics['memory_usage']:.4f}MB ({performance_ratios['memory_usage']:.2f}x base) (based on {len(rag_metrics['memory_usage'])}/{total} samples, 0-5MB only)")
    print(f"  - Breakdown: Retrieval={rag_avg_metrics['retrieval_memory_avg']:.4f}MB, Final Increment={rag_avg_metrics['final_increment_avg']:.4f}MB")
    print(f"  Avg GPU Utilization: {rag_avg_metrics['gpu_utilization']:.2f}% ({performance_ratios['gpu_utilization']:.2f}x base) (based on {len(rag_metrics['gpu_utilization'])}/{total} samples, >0 only)")
    
    return basic_accuracy, rag_accuracy, basic_avg_metrics, rag_avg_metrics, performance_ratios


def run_rag_evaluation(rule, domain, model_name, top_k, log_file, dataset_source='existing', rag_material_source='strong', custom_api_config=None):
    """Background task for RAG evaluation with performance metrics"""
    global current_rag_log
    current_rag_log = log_file
    
    # 核心：获取当前脚本(workflow.py)的绝对路径和目录，所有路径基于此计算
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    
    try:
        # 重定向输出到日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print(f"Starting RAG evaluation - Domain: {domain}, Model: {model_name}, Top-K: {top_k}")
                print(f"Dataset source: {dataset_source}, RAG material source: {rag_material_source}")
                print("=" * 50)
                
                # -------------------------- 1. 检查RAG预处理数据 --------------------------
                print(f"[1/5] Checking and preprocessing {domain} domain data...")
                
                # 根据RAG材料来源选择正确的路径
                if rag_material_source == "strong":
                    print("Using Strong Wiki material...")
                    # RAG预处理文件路径：../../data/RAG_material/Strong_cleaned/Strong_{domain}.txt
                    rag_cleaned_dir = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'RAG_material', 'Strong_cleaned'
                    ))
                    data_file_path = os.path.join(rag_cleaned_dir, f'Strong_{domain}.txt')
                    
                    if not os.path.exists(data_file_path):
                        print(f"Strong data not found, generating from descriptions...")
                        # 调用修改后的description_collector来生成Strong材料
                        from work_models.description_collector import collect_descriptions_to_strong
                        collect_descriptions_to_strong(domain, 'reproducible')
                    else:
                        print(f"Strong data already exists (path: {data_file_path}), skipping generation")
                elif rag_material_source == "dynamic_strong":
                    print("Using Dynamic Strong Wiki material...")
                    # RAG预处理文件路径：../../data/RAG_material/Dynamic_Strong_cleaned/Dynamic_Strong_{domain}.txt
                    rag_cleaned_dir = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'RAG_material', 'Dynamic_Strong_cleaned'
                    ))
                    data_file_path = os.path.join(rag_cleaned_dir, f'Dynamic_Strong_{domain}.txt')
                    
                    if not os.path.exists(data_file_path):
                        print(f"Dynamic Strong data not found, generating from descriptions...")
                        # 调用修改后的description_collector来生成Dynamic Strong材料
                        from work_models.description_collector import collect_descriptions_to_strong
                        collect_descriptions_to_strong(domain, 'dynamic')
                    else:
                        print(f"Dynamic Strong data already exists (path: {data_file_path}), skipping generation")
                else:  # rag_material_source == "flexible"
                    print("Using flexible uploaded material...")
                    # RAG预处理文件路径：../../data/RAG_material/cleaned/cleaned_{domain}.txt
                    rag_cleaned_dir = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'RAG_material', 'cleaned'
                    ))
                    data_file_path = os.path.join(rag_cleaned_dir, f'cleaned_{domain}.txt')
                    
                    if not os.path.exists(data_file_path):
                        print(f"Flexible data not found, starting rag_preprocess({domain})...")
                        rag_preprocess(domain, data_file_path)
                    else:
                        print(f"Flexible data already processed (path: {data_file_path}), skipping")

                # -------------------------- 2. 检查FAISS向量索引 --------------------------
                print(f"[2/5] Checking and building {domain} domain vector index...")
                
                # 根据RAG材料来源选择正确的索引路径
                if rag_material_source == "strong":
                    print("Using Strong Wiki material index...")
                    # 向量索引路径：../../data/RAG_material/Strong_base/{domain}_index.faiss
                    knowledge_base_dir = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'RAG_material', 'Strong_base'
                    ))
                    index_file_path = os.path.join(knowledge_base_dir, f'{domain}_index.faiss')
                    
                    if not os.path.exists(index_file_path):
                        print(f"Strong index not found, generating from descriptions...")
                        # 调用修改后的description_collector来生成Strong索引
                        from work_models.description_collector import collect_descriptions_to_strong
                        collect_descriptions_to_strong(domain, 'reproducible')
                    else:
                        print(f"Strong index already exists (path: {index_file_path}), loading...")
                elif rag_material_source == "dynamic_strong":
                    print("Using Dynamic Strong Wiki material index...")
                    # 向量索引路径：../../data/RAG_material/Dynamic_Strong_base/{domain}_index.faiss
                    knowledge_base_dir = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'RAG_material', 'Dynamic_Strong_base'
                    ))
                    index_file_path = os.path.join(knowledge_base_dir, f'{domain}_index.faiss')
                    
                    if not os.path.exists(index_file_path):
                        print(f"Dynamic Strong index not found, generating from descriptions...")
                        # 调用修改后的description_collector来生成Dynamic Strong索引
                        from work_models.description_collector import collect_descriptions_to_strong
                        collect_descriptions_to_strong(domain, 'dynamic')
                    else:
                        print(f"Dynamic Strong index already exists (path: {index_file_path}), loading...")
                else:  # rag_material_source == "flexible"
                    print("Using flexible uploaded material index...")
                    # 向量索引路径：../../data/RAG_material/knowledge_base/{domain}_index.faiss
                    knowledge_base_dir = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'RAG_material', 'knowledge_base'
                    ))
                    index_file_path = os.path.join(knowledge_base_dir, f'{domain}_index.faiss')
                    
                    if not os.path.exists(index_file_path):
                        print(f"Flexible index not built, starting build_faiss_index({domain})...")
                        build_faiss_index(domain, index_file_path)
                    else:
                        print(f"Flexible index already exists (path: {index_file_path}), loading...")
                
                # -------------------------- 3. 加载评估数据集 --------------------------
                print(f"[3/5] Loading {domain} domain evaluation dataset...")
                
                # 根据数据集来源选择正确的路径
                if dataset_source == "existing":
                    print("Using reproducible dataset...")
                    # 加载现有数据集路径：../../data/dataset/generated/{rule}/{domain}_qa.json
                    dataset_path = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'dataset', 'generated', rule,
                        f'{domain}_qa.json'
                    ))
                else:  # dataset_source == "new"
                    print("Using dynamically generated dataset...")
                    # 使用动态数据集路径：../../data/dataset/dynamic/{rule}/{domain}_qa.json
                    dataset_path = os.path.abspath(os.path.join(
                        current_script_dir, '../../', 'data', 'dataset', 'dynamic', rule,
                        f'{domain}_qa.json'
                    ))
                
                if not os.path.exists(dataset_path):
                    print(f"Error: Dataset not found - {dataset_path}")
                    return
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    test_dataset = json.load(f)

                max_samples = 59
                if len(test_dataset) > max_samples:
                    test_dataset = test_dataset[:max_samples]  
                    print(f"Limited dataset to first {max_samples} samples")

                print(f"Loaded {len(test_dataset)} samples from {dataset_path}")
                # -------------------------- 4. 初始化模型和检索器 --------------------------
                print(f"[4/5] Initializing model and retriever...")
                global model, index, cleaned_abstracts
                
                # 加载嵌入模型
                model = load_embedding_model()
                print(f"Embedding model device: {model.device if hasattr(model, 'device') else 'CPU'}") 

                # 加载FAISS索引（复用步骤2的index_file_path）
                # 注意：需确保load_faiss_index支持传入自定义路径
                index = load_faiss_index(domain, index_file_path)
                
                # 加载预处理文档（复用步骤1的data_file_path）
                with open(data_file_path, 'r', encoding='utf-8') as f:
                    cleaned_abstracts = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded {len(cleaned_abstracts)} preprocessed document chunks")
                
                # -------------------------- 5. 执行RAG评估 --------------------------
                print(f"[5/5] Starting RAG evaluation... ({len(test_dataset)} samples)")
                
                # 检查是否使用自定义API
                if model_name == 'api' and custom_api_config:
                    print(f"Using DeepSeek API for RAG evaluation: {custom_api_config['api_model_name']}")
                    basic_score, rag_score, basic_metrics, rag_metrics, perf_ratios = evaluate_rag_model_with_api(
                        api_config=custom_api_config,
                        domain=domain,
                        test_questions=test_dataset,
                        top_k=top_k
                    )
                else:
                    basic_score, rag_score, basic_metrics, rag_metrics, perf_ratios = evaluate_rag_model(
                        model_name=model_name,
                        domain=domain,
                        test_questions=test_dataset,
                        top_k=top_k
                    )
                
                # -------------------------- 6. 保存RAG评估结果 --------------------------
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                # 结果保存路径：../experiments/results/rag_results/...
                rag_results_dir = os.path.abspath(os.path.join(
                    current_script_dir, '..', 'experiments', 'results', 'rag_results'
                ))
                os.makedirs(rag_results_dir, exist_ok=True)
                result_filename = f"rag_{domain}_{model_name}_{timestamp}.json"
                result_path = os.path.join(rag_results_dir, result_filename)
                
                                # 组织结果数据
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
                
                # 计算Transformation指标
                metrics = result_data["metrics"]
                w_time = 0.4
                w_gpu = 0.3
                w_mem = 0.3
                
                r_time = metrics.get('performance', {}).get('ratios', {}).get('response_time', 0)
                r_gpu = metrics.get('performance', {}).get('ratios', {}).get('gpu_utilization', 0)  
                r_mem = metrics.get('performance', {}).get('ratios', {}).get('memory_usage', 0)
                
                transformation = 0.0
                if r_time != 0:
                    transformation += w_time / r_time
                if r_gpu != 0:
                    transformation += w_gpu / r_gpu
                if r_mem != 0:
                    transformation += w_mem / r_mem
                
                metrics['transformation'] = round(transformation, 4)
                result_data["metrics"] = metrics
                
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                print(f"\nRAG evaluation completed! Results saved to: {result_path}")
                print(f"Basic accuracy: {basic_score:.4f} | RAG accuracy: {rag_score:.4f} | Improvement: {rag_score - basic_score:.4f}")
                
                # 更新进程状态（若需要）
                process_id = log_file.split('_')[-1].split('.')[0]
                if process_id in rag_processes:
                    rag_processes[process_id]["status"] = "completed"
                    rag_processes[process_id]["result_file"] = result_path
    
    except Exception as e:
        # 追加错误信息到日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nError: {str(e)}\n")
            f.write(traceback.format_exc())  # 打印详细错误栈
        
        # 更新进程错误状态
        process_id = log_file.split('_')[-1].split('.')[0]
        if process_id in rag_processes:
            rag_processes[process_id]["status"] = "error"
            rag_processes[process_id]["error_message"] = str(e)


def evaluate_rag_model_with_api(api_config, domain, test_questions, top_k):
    """RAG evaluation with DeepSeek API"""
    global model, index, cleaned_abstracts  # Use pre-loaded global variables
    
    # 1. Initialize API connection
    headers = {
        'Content-Type': 'application/json',
    }
    
    if api_config.get('api_key'):
        headers['Authorization'] = f'Bearer {api_config["api_key"]}'
    
    # 自动转换 Hugging Face 旧端点为新路由端点
    api_endpoint = api_config['api_endpoint']
    if 'api-inference.huggingface.co/models/' in api_endpoint:
        api_endpoint = 'https://router.huggingface.co/v1/chat/completions'
        print(f"Converted Hugging Face endpoint to: {api_endpoint}")
    
    # 格式化提示模板
    prompt_template = api_config.get('api_prompt_template', '') or '''Based on the provided context, answer the following question with ONLY "yes" or "no".

Context:
{context}

Question: {question}

Answer (yes or no only):'''
    # 确保 prompt_template 包含必要的占位符
    if '{context}' not in prompt_template and '{question}' not in prompt_template:
        prompt_template = '''Based on the provided context, answer the following question with ONLY "yes" or "no".

Context:
{context}

Question: {question}

Answer (yes or no only):'''
    elif '{context}' not in prompt_template:
        # 如果没有 context 占位符，添加它
        prompt_template = f'Context:\n{{context}}\n\n{prompt_template}'
    
    # 2. Initialize counters and total samples
    basic_correct = 0
    rag_correct = 0
    basic_failed_count = 0  # 记录基础模型 API 失败次数
    rag_failed_count = 0    # 记录 RAG 模型 API 失败次数
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
        
        # 基础模型评估（使用API）
        basic_start_time = time.time()
        basic_start_memory = get_memory_usage()
        basic_start_gpu = get_gpu_utilization() if get_device() == "cuda" else 0
        
        # 对于基础模型，如果没有 context 占位符，直接使用 question；否则将 context 设为空
        if '{context}' in prompt_template:
            basic_prompt = prompt_template.format(context="", question=question)
        else:
            basic_prompt = prompt_template.format(question=question)
        
        # 调试信息：打印前几个样本的 prompt（仅前3个）
        if i < 3:
            print(f"[DEBUG] Basic prompt for sample {i+1}: {basic_prompt[:200]}...")
        basic_payload = {
            'model': api_config['api_model_name'],
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that answers questions with only "yes" or "no". Be concise and accurate.'
                },
                {'role': 'user', 'content': basic_prompt}
            ],
            'max_tokens': api_config.get('api_max_tokens', 1000),
            'temperature': 0.0
        }
        
        # 调试信息：打印前几个样本的 payload（仅前2个）
        if i < 2:
            print(f"[DEBUG] Basic payload for sample {i+1}: model={basic_payload['model']}, messages={basic_payload['messages']}")
        
        basic_api_success = False  # 标记基础模型 API 是否成功
        try:
            basic_response = requests.post(
                api_endpoint,
                headers=headers,
                json=basic_payload,
                timeout=30
            )
            basic_response.raise_for_status()
            basic_result = basic_response.json()
            basic_answer = basic_result['choices'][0]['message']['content'].strip().lower()
            basic_api_success = True
        except Exception as e:
            print(f"Basic API call failed: {e}")
            basic_answer = "no answer"
            basic_failed_count += 1
        
        # 记录基础模型性能指标
        basic_end_time = time.time()
        basic_end_memory = get_memory_usage()
        basic_end_gpu = get_gpu_utilization() if get_device() == "cuda" else 0
        
        basic_time = basic_end_time - basic_start_time
        basic_peak_memory = max(basic_start_memory, basic_end_memory)
        basic_gpu = basic_end_gpu - basic_start_gpu
        
        if basic_time > 0:
            basic_metrics["response_time"].append(basic_time)
        # 跳过第一次评估的内存统计以避免模型加载开销的影响
        if i > 0:  # i > 0 表示不是第一次评估
            basic_metrics["memory_usage"].append(basic_peak_memory)  # 使用峰值而不是差值
        if basic_start_gpu >= 0 and basic_end_gpu >= 0:  
            basic_metrics["gpu_utilization"].append(abs(basic_gpu))  
        
        # Use classifier to predict basic model answer
        basic_prediction, basic_confidence = answer_classifier.predict(basic_answer)
        # Check correctness
        if basic_prediction == reference_answer:
            basic_correct += 1
        
        # RAG模型评估（使用API）
        rag_start_time = time.time()
        rag_start_memory = get_memory_usage()
        rag_start_gpu = get_gpu_utilization() if get_device() == "cuda" else 0
        
        # Retrieve relevant documents
        query_embedding = model.encode(question, convert_to_tensor=True)
        query_embedding_2d = np.expand_dims(query_embedding.cpu().numpy(), axis=0)
        distances, indices = index.search(query_embedding_2d, top_k)
        retrieved_docs = [cleaned_abstracts[i] for i in indices[0] if i < len(cleaned_abstracts)]
        
        # 记录检索后的内存峰值
        after_retrieval_memory = get_memory_usage()
        rag_peak_memory = max(rag_start_memory, after_retrieval_memory)
        
        # Build context-aware prompt with length limit and better formatting
        context_docs = retrieved_docs[:3]  # Limit to top 5 most relevant docs
        context = "\n\n".join([f"[Document {i+1}] {doc}" for i, doc in enumerate(context_docs)])

        max_context_length = 1000  
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        # 对于 RAG 模型，确保使用完整的 context
        if '{context}' in prompt_template:
            rag_prompt = prompt_template.format(context=context, question=question)
        else:

            rag_prompt = f"Context:\n{context}\n\n{prompt_template.format(question=question)}"

        if i < 3:
            print(f"[DEBUG] RAG prompt for sample {i+1}: {rag_prompt[:300]}...")
        
        rag_payload = {
            'model': api_config['api_model_name'],
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that answers questions with only "yes" or "no" based on the provided context. Be concise and accurate.'
                },
                {'role': 'user', 'content': rag_prompt}
            ],
            'max_tokens': api_config.get('api_max_tokens', 1000),
            'temperature': 0.0
        }
        

        if i < 2:
            print(f"[DEBUG] RAG payload for sample {i+1}: model={rag_payload['model']}, message_length={len(rag_prompt)}")
        
        rag_api_success = False  # 标记 RAG 模型 API 是否成功
        try:
            rag_response = requests.post(
                api_endpoint,
                headers=headers,
                json=rag_payload,
                timeout=30
            )
            rag_response.raise_for_status()
            rag_result = rag_response.json()
            rag_answer = rag_result['choices'][0]['message']['content'].strip().lower()
            rag_api_success = True
        except Exception as e:
            print(f"RAG API call failed: {e}")
            rag_answer = "no answer"
            rag_failed_count += 1
        
        # 记录RAG模型性能指标
        rag_end_time = time.time()
        rag_end_memory = get_memory_usage()
        rag_end_gpu = get_gpu_utilization() if get_device() == "cuda" else 0
        
        rag_time = rag_end_time - rag_start_time
        rag_peak_memory = max(rag_peak_memory, rag_start_memory, rag_end_memory)
        rag_gpu = rag_end_gpu - rag_start_gpu
        
        if rag_time > 0:
            rag_metrics["response_time"].append(rag_time)
        # 跳过第一次评估的内存统计以避免模型加载开销的影响
        if i > 0:  # i > 0 表示不是第一次评估
            rag_metrics["memory_usage"].append(rag_peak_memory)  # 使用峰值而不是差值
        if rag_start_gpu >= 0 and rag_end_gpu >= 0:  
            rag_metrics["gpu_utilization"].append(abs(rag_gpu))  
        
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
            print(f"  API Success: {basic_api_success}")
            if basic_api_success:
                print(f"  Semantic Prediction: {basic_prediction} (Confidence: {basic_confidence:.4f})")
                print(f"  Correctness: {'Correct' if basic_prediction == reference_answer else 'Incorrect'}")
            else:
                print("  Status: API call failed - excluded from statistics")
            print(f"  Performance: Time={basic_time:.4f}s, "
                  f"Memory={basic_peak_memory:.4f}MB, "
                  f"GPU={basic_gpu:.2f}%")

            print(f"\nRAG-Enhanced Model:")
            print(f"  Raw Output: {rag_answer}")
            print(f"  API Success: {rag_api_success}")
            if rag_api_success:
                print(f"  Semantic Prediction: {rag_prediction} (Confidence: {rag_confidence:.4f})")
                print(f"  Correctness: {'Correct' if rag_prediction == reference_answer else 'Incorrect'}")
            else:
                print("  Status: API call failed - excluded from statistics")
            print(f"  Performance: Time={rag_time:.4f}s, "
                  f"Memory={rag_peak_memory:.4f}MB, "
                  f"GPU={rag_gpu:.2f}%")
            print(f"  Retrieved Documents: {len(retrieved_docs)}")
            print("=" * 70)  # Separator line
    
    # Calculate accuracy (excluding failed API calls)
    basic_successful_calls = total - basic_failed_count
    rag_successful_calls = total - rag_failed_count

    basic_accuracy = basic_correct / basic_successful_calls if basic_successful_calls > 0 else 0
    rag_accuracy = rag_correct / rag_successful_calls if rag_successful_calls > 0 else 0

    # 计算性能指标平均值（基于峰值）
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
    print(f"Total Samples: {total}")
    print(f"Base Model API Failures: {basic_failed_count} | Successful Calls: {basic_successful_calls}")
    print(f"RAG Model API Failures: {rag_failed_count} | Successful Calls: {rag_successful_calls}")
    print(f"Base Model:")
    print(f"  Accuracy: {basic_accuracy:.4f} (based on {basic_successful_calls} successful calls)")
    print(f"  Avg Response Time: {basic_avg_metrics['response_time']:.4f}s (based on {len(basic_metrics['response_time'])}/{total} samples)")
    print(f"  Avg Memory Usage: {basic_avg_metrics['memory_usage']:.4f}MB (based on {len(basic_metrics['memory_usage'])}/{total} samples)")
    print(f"  Avg GPU Utilization: {basic_avg_metrics['gpu_utilization']:.2f}% (based on {len(basic_metrics['gpu_utilization'])}/{total} samples)")

    print(f"\nRAG Model:")
    print(f"  Accuracy: {rag_accuracy:.4f} (based on {rag_successful_calls} successful calls)")
    print(f"  Avg Response Time: {rag_avg_metrics['response_time']:.4f}s ({performance_ratios['response_time']:.2f}x base)")
    print(f"  Avg Memory Usage: {rag_avg_metrics['memory_usage']:.4f}MB ({performance_ratios['memory_usage']:.2f}x base)")
    print(f"  Avg GPU Utilization: {rag_avg_metrics['gpu_utilization']:.2f}% ({performance_ratios['gpu_utilization']:.2f}x base)")

    return basic_accuracy, rag_accuracy, basic_avg_metrics, rag_avg_metrics, performance_ratios

