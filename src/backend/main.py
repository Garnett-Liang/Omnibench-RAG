# src/backend/main.py
import os
import time
import json
import random
import threading
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, make_response
from contextlib import redirect_stdout
from pyswip import Prolog, Atom
from dynamic_dataset import generate_dynamic_dataset
from workflow import (
    BinaryAnswerClassifier, extract_valid_answer, 
    generate_answers, evaluate_model, run_evaluation,
    evaluate_rag_model, run_rag_evaluation
)
from work_models.data_preprocess import rag_preprocess
from work_models.embed_faiss import build_faiss_index

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = '../../data/RAG_material/raw'  
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}


answer_classifier = BinaryAnswerClassifier()


current_log_file = None
rag_processes = {}  
current_rag_log = None


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
        
        # Check for custom API configuration
        custom_api_config = None
        if model_choice == 'api':
            custom_api_config = {
                'api_endpoint': data.get('api_endpoint'),
                'api_key': data.get('api_key'),
                'api_model_name': data.get('api_model_name'),
                'api_max_tokens': data.get('api_max_tokens', 1000),
                'api_prompt_template': data.get('api_prompt_template', 'Answer with "yes" or "no": {question}')
            }
            
            if not custom_api_config['api_endpoint'] or not custom_api_config['api_model_name']:
                return jsonify({"status": "error", "message": "API endpoint and model name are required for custom API"}), 400
        
        if not all([rule_choice, domain_choice, model_choice]):
            return jsonify({"status": "error", "message": "Missing required parameters"}), 400
        
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
        timestamp = int(time.time())
        log_filename = f"evaluation_{timestamp}.log"
        logs_dir = os.path.join(project_root, "experiments", "logs")
        log_file = os.path.join(logs_dir, log_filename) 
        
        thread = threading.Thread(
            target=run_evaluation, 
            args=(rule_choice, domain_choice, model_choice, log_file, dataset_source, custom_api_config)
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

@app.route('/api/dataset/<rule>/<domain>')
def get_dataset(rule, domain):

    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    base_dir = os.path.abspath(os.path.join(current_file_dir, '../..', 'data/dataset/generated'))
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
                f"Question:\n{item['question']}\n\n"
                "Answer me with ONE word 'yes' or 'no'."
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
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
    full_path = os.path.join(project_root, "experiments", "logs", log_file)
    
    if not os.path.exists(full_path):
        return jsonify({"status": "not_found", "message": f"Log file does not exist: {full_path}"}), 404
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            logs = f.read()
        return jsonify({"status": "success", "logs": logs})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error reading log file: {str(e)}"}), 500
    

@app.route('/api/progress/<path:log_file>')
def get_progress(log_file):

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
    full_path = os.path.join(project_root, "experiments", "logs", log_file)
    
    if not os.path.exists(full_path):
        return jsonify({
            "status": "processing", 
            "progress": 0, 
            "message": f"Log file does not exist; evaluation may not have started yet (looked in: {full_path})"
        }), 200
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            logs = f.read()
        
        progress = 0
        message = "Evaluation in progress..."
        
        
        if "Selected reasoning rule:" in logs:  
            progress = 10
        if "Selected domain:" in logs:  
            progress = 20
        if "Using reproducible dataset" in logs or "Using dynamically generated dataset" in logs or "Loaded" in logs and "questions" in logs:
            progress = 40
        if "Evaluating with model:" in logs:
            progress = 70
        if "Results saved to:" in logs: 
            progress = 100
            message = "Evaluation completed" 
        elif "Error:" in logs:  
            progress = 100
            message = "Evaluation error"  
            
        return jsonify({
            "status": "processing", 
            "progress": progress, 
            "message": message, 
            "log_file": log_file
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Progress get failure: {str(e)}"
        }), 500
    
@app.route('/api/results/<log_file>')
def get_results(log_file):
    """Retrieve evaluation results (read from 'results' folder via log file)"""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
    log_full_path = os.path.join(project_root, "experiments", "logs", log_file)

    
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
        dataset_source = data.get('dataset_source', 'existing')  # Default to existing dataset
        rag_material_source = data.get('rag_material_source', 'strong')  # Default to strong wiki material
        
        # Check for custom API configuration
        custom_api_config = None
        if model_name == 'api':
            custom_api_config = {
                'api_endpoint': data.get('api_endpoint'),
                'api_key': data.get('api_key'),
                'api_model_name': data.get('api_model_name'),
                'api_max_tokens': data.get('api_max_tokens', 1000),
                'api_prompt_template': data.get('api_prompt_template', 'Answer with "yes" or "no": {question}')
            }
            
            if not custom_api_config['api_endpoint'] or not custom_api_config['api_model_name']:
                return jsonify({"status": "error", "message": "API endpoint and model name are required for custom API"}), 400
        
        if not domain:
            return jsonify({"status": "error", "message": "Missing required parameters"}), 400
        
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"rag_evaluation_{timestamp}.log"
        log_file = os.path.abspath(os.path.join(
            current_script_dir,
            '..',  
            'experiments', 'logs',
            log_filename
        ))
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        thread = threading.Thread(
            target=run_rag_evaluation,
            args=(rule, domain, model_name, top_k, log_file, dataset_source, rag_material_source, custom_api_config)
        )
        thread.daemon = True
        thread.start()
        
  
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
            "log_file": log_filename,
            "debug_log_path": log_file  
        }), 200
        
    except Exception as e:
        traceback.print_exc()  
        return jsonify({"status": "error", "message": f"Error processing request: {str(e)}"}), 500

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
    

    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    

    log_path = os.path.abspath(os.path.join(
        current_script_dir,
        '..',  
        'experiments', 'logs',
        log_file
    ))
    
    if not os.path.exists(log_path):
        return jsonify({
            "status": "error", 
            "message": f"Log file not found at: {log_path}"
        }), 404
    
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
            "start_time": process.get("start_time"),
            "debug_log_path": log_path  
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to get progress: {str(e)}"}), 500

@app.route('/api/rag_results/<log_file>')
def get_rag_results(log_file):
    """Retrieve RAG evaluation results with Transformation metric"""
    try:
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)        
        log_path = os.path.abspath(os.path.join(
            current_script_dir, 
            '..', 
            'experiments', 'logs', 
            log_file
        ))
        
        if not os.path.exists(log_path):
            return jsonify({
                "status": "error", 
                "message": f"Log file not found at: {log_path}"
            }), 404
        

        result_file = None
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "RAG evaluation completed! Results saved to:" in line:  
                    result_file = line.split("RAG evaluation completed! Results saved to: ")[1].strip()
                    break
        
        if not result_file:
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = f.read()
            
            if "RAG evaluation completed" in logs:
                return jsonify({
                    "status": "error", 
                    "message": "Failed to extract result file path from logs"
                }), 500
            else:
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
                    "log_file": log_file,
                    "debug_log_path": log_path 
                }), 202
        

        if not os.path.isabs(result_file):

            log_dir = os.path.dirname(log_path)
            result_file = os.path.abspath(os.path.join(log_dir, result_file))
        
        if not os.path.exists(result_file):
            return jsonify({
                "status": "error", 
                "message": f"Result file not found: {result_file}"
            }), 404
        
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
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
        results['metrics'] = metrics

        return jsonify({
            "status": "success",
            "results": results,
            "result_file": result_file
        }), 200
        
    except json.JSONDecodeError:
        return jsonify({
            "status": "error", 
            "message": f"Invalid result file format, cannot parse: {result_file}"
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Failed to retrieve results: {str(e)}"
        }), 500



@app.route('/api/get_rag_materials/<domain>', methods=['GET'])
def get_rag_materials(domain):
    try:

        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)        
        data_file_path = os.path.abspath(os.path.join(
            current_script_dir, 
            '../..',  
            'data', 'RAG_material', 'cleaned', 
            f'cleaned_{domain}.txt'
        ))
        

        index_file_path = os.path.abspath(os.path.join(
            current_script_dir, 
            '../..',  
            'data', 'RAG_material', 'knowledge_base', 
            f'{domain}_index.faiss'
        ))
        

        if os.path.exists(data_file_path):
            with open(data_file_path, 'r', encoding='utf-8') as f:
                all_lines = [line.strip() for line in f.readlines() if line.strip()]
                sample_size = min(10, len(all_lines))
                random_lines = random.sample(all_lines, sample_size) if all_lines else []
                data_content = '\n'.join(random_lines) + ('\n...' if len(all_lines) > sample_size else '')
        else:
            data_content = f"No preprocessed data found in {data_file_path}"


        index_content = "Vector index file cannot be directly displayed. " \
                        f"File exists: {os.path.exists(index_file_path)}"

        return jsonify({
            "status": "success",
            "data_content": data_content,
            "index_content": index_content,
            "debug_path": data_file_path 
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
    
    
@app.route('/dataset_generation.html')
def dataset_generation():
    return render_template('dataset_generation.html')

@app.route('/rag_material_management.html')
def rag_management():
    return render_template('rag_material_management.html')

@app.route('/api/generate_dataset', methods=['POST'])
def generate_dataset():
    try:
        data = request.json
        rule_choice = data.get('rule_choice')
        domain_choice = data.get('domain_choice')
        
        if not all([rule_choice, domain_choice]):
            return jsonify({"status": "error", "message": "Missing required parameters"}), 400
        
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = int(time.time())
        log_filename = f"dataset_generation_{timestamp}.log"
        logs_dir = os.path.join(current_script_dir, '..', 'experiments', 'logs')
        log_file = os.path.join(logs_dir, log_filename) 
        
        thread = threading.Thread(
            target=run_dataset_generation, 
            args=(rule_choice, domain_choice, log_file)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "status": "processing", 
            "message": "Dataset generation started. Check logs for details.",
            "log_file": log_filename
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing request: {str(e)}"}), 500

@app.route('/api/dataset_progress/<log_file>')
def get_dataset_progress(log_file):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_script_dir, '..', 'experiments', 'logs', log_file)
    
    if not os.path.exists(full_path):
        return jsonify({
            "status": "processing", 
            "progress": 0, 
            "message": f"Log file does not exist; generation may not have started yet"
        }), 200
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            logs = f.read()
        
        progress = 0
        message = "Dataset generation in progress..."
        
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
            message = "Dataset generation completed" 
        elif "Error:" in logs:  
            progress = 100
            message = "Dataset generation error"  
        
        return jsonify({
            "status": "processing", 
            "progress": progress, 
            "message": message, 
            "log_file": log_file
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Progress get failure: {str(e)}"
        }), 500

@app.route('/api/dataset_logs/<log_file>')
def get_dataset_logs(log_file):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_script_dir, '..', 'experiments', 'logs', log_file)
    
    if not os.path.exists(full_path):
        return jsonify({"status": "not_found", "message": f"Log file does not exist: {full_path}"}), 404
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            logs = f.read()
        return jsonify({"status": "success", "logs": logs})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error reading log file: {str(e)}"}), 500

@app.route('/api/dataset_results/<log_file>')
def get_dataset_results(log_file):
    """Retrieve dataset generation results"""
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    log_full_path = os.path.join(current_script_dir, '..', 'experiments', 'logs', log_file)
    
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
                "message": "Generation incomplete. Result file path not found."
            }), 202
        
        if not os.path.exists(result_file_path):
            return jsonify({
                "status": "error", 
                "message": f"Result file not found: {result_file_path}"
            }), 404
        
        with open(result_file_path, 'r', encoding='utf-8') as result_f:
            dataset = json.load(result_f)
        
        return jsonify({
            "status": "success",
            "result_file": result_file_path,
            "total_questions": len(dataset),
            "dataset_size": f"{os.path.getsize(result_file_path)} bytes",
            "file_path": result_file_path
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


@app.route('/api/dynamic_dataset/<rule>/<domain>')
def get_dynamic_dataset(rule, domain):
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    base_dir = os.path.abspath(os.path.join(current_file_dir, '../..', 'data/dataset/dynamic'))
    rule_dir = os.path.join(base_dir, rule)
    dataset_file = os.path.join(rule_dir, f'{domain}_qa.json')
    
    try:
        if not os.path.exists(dataset_file):
            return jsonify({
                "status": "error", 
                "message": f"Dynamic dataset file does not exist: {dataset_file}"
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
                "Answer me with 'yes' or 'no'.No more other words"
            )
            questions.append(full_question)
            answers.append(item["answer"])
        
        return jsonify({
            "status": "success",
            "message": f"Loaded dynamic dataset samples for {domain} domain under {rule} rule",
            "sample_size": sample_size,
            "total_size": len(dataset),  
            "questions": questions,
            "answers": answers
        })
    
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error retrieving dynamic dataset: {str(e)}"
        }), 500


def run_dataset_generation(rule_choice, domain_choice, log_file):
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        result_file_path = generate_dynamic_dataset(rule_choice, domain_choice, current_script_dir, log_file)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Results saved to: {result_file_path}\n")
            
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Error: {str(e)}\n")


@app.route('/api/generate_strong_material_sync', methods=['POST'])
def generate_strong_material_sync():
    """Generate Strong Wiki materials from datasets (synchronous)"""
    try:
        data = request.json
        domain = data.get('domain')
        dataset_source = data.get('dataset_source', 'reproducible')

        if not domain:
            return jsonify({"status": "error", "message": "Domain is required"}), 400

        from work_models.description_collector import collect_descriptions_to_strong
        collect_descriptions_to_strong(domain, dataset_source)

        return jsonify({
            "status": "success",
            "message": "Strong material generated successfully"
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error generating material: {str(e)}"}), 500

@app.route('/api/get_rag_materials/<domain>/<material_source>')
def get_rag_materials_with_source(domain, material_source):
    """Get RAG materials for a specific domain and material source"""
    try:
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)

        # Determine paths based on material source
        if material_source == "flexible":
            data_file_path = os.path.abspath(os.path.join(
                current_script_dir,
                '../..',
                'data', 'RAG_material', 'cleaned',
                f'cleaned_{domain}.txt'
            ))
            index_file_path = os.path.abspath(os.path.join(
                current_script_dir,
                '../..',
                'data', 'RAG_material', 'knowledge_base',
                f'{domain}_index.faiss'
            ))
        elif material_source == "strong":
            data_file_path = os.path.abspath(os.path.join(
                current_script_dir,
                '../..',
                'data', 'RAG_material', 'Strong_cleaned',
                f'Strong_{domain}.txt'
            ))
            index_file_path = os.path.abspath(os.path.join(
                current_script_dir,
                '../..',
                'data', 'RAG_material', 'Strong_base',
                f'{domain}_index.faiss'
            ))
        elif material_source == "dynamic_strong":
            data_file_path = os.path.abspath(os.path.join(
                current_script_dir,
                '../..',
                'data', 'RAG_material', 'Dynamic_Strong_cleaned',
                f'Dynamic_Strong_{domain}.txt'
            ))
            index_file_path = os.path.abspath(os.path.join(
                current_script_dir,
                '../..',
                'data', 'RAG_material', 'Dynamic_Strong_base',
                f'{domain}_index.faiss'
            ))
        else:
            return jsonify({"status": "error", "message": "Invalid material source"}), 400

        if os.path.exists(data_file_path):
            with open(data_file_path, 'r', encoding='utf-8') as f:
                all_lines = [line.strip() for line in f.readlines() if line.strip()]
                sample_size = min(10, len(all_lines))
                random_lines = random.sample(all_lines, sample_size) if all_lines else []
                data_content = '\n'.join(random_lines) + ('\n...' if len(all_lines) > sample_size else '')
        else:
            data_content = f"No preprocessed data found in {data_file_path}"

        index_content = "Vector index file cannot be directly displayed. " \
                        f"File exists: {os.path.exists(index_file_path)}"

        return jsonify({
            "status": "success",
            "data_content": data_content,
            "index_content": index_content,
            "debug_path": data_file_path
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to get RAG materials: {str(e)}"}), 500

@app.route('/api/statistics/<model_id>')
def get_model_statistics(model_id):
    """获取指定模型的统计数据"""
    try:
        # 导入统计功能
        import importlib.util
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        
        # statistics_utils.py现在在work_models文件夹下
        statistics_utils_file = os.path.abspath(os.path.join(current_script_dir, 'work_models', 'statistics_utils.py'))
        # rag_results文件夹路径
        rag_results_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'experiments', 'results', 'rag_results'))
        
        # 使用importlib导入指定路径的模块
        spec = importlib.util.spec_from_file_location("statistics_utils", statistics_utils_file)
        statistics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(statistics_module)
        
        # 加载统计数据
        results = statistics_module.load_rag_results(rag_results_dir)
        statistics = statistics_module.get_model_statistics(results, model_id)
        
        return jsonify({
            "status": "success",
            "statistics": statistics
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to get statistics: {str(e)}"}), 500

@app.route('/api/radar_chart/<model_id>')
def get_radar_chart(model_id):
    """获取指定模型的雷达图"""
    try:
        import importlib.util
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        
        # statistics_utils.py现在在work_models文件夹下
        statistics_utils_file = os.path.abspath(os.path.join(current_script_dir, 'work_models', 'statistics_utils.py'))
        
        # 使用importlib导入指定路径的模块
        spec = importlib.util.spec_from_file_location("statistics_utils", statistics_utils_file)
        statistics_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(statistics_module)
        
        # 生成雷达图的base64数据
        chart_base64 = statistics_module.generate_model_radar_chart(model_id, "base64")
        
        if chart_base64:
            return jsonify({
                "status": "success",
                "chart_base64": chart_base64,
                "model_id": model_id
            }), 200
        else:
            return jsonify({"status": "error", "message": "Failed to generate radar chart"}), 500
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to get radar chart: {str(e)}"}), 500



if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, port=5000)