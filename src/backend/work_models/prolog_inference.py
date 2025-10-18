import os
import json
import random
import time
from pathlib import Path
from pyswip import Prolog, Atom
from tqdm import tqdm
import re
import requests
import os
import json
import random
import time
from pathlib import Path
from pyswip import Prolog, Atom
from tqdm import tqdm
import re
import requests
import argparse

# 用户代理列表，模仿真实浏览器
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

def get_random_user_agent():
    """获取随机用户代理"""
    return random.choice(USER_AGENTS)

def get_realistic_headers():
    """生成更真实的浏览器头信息"""
    return {
        'User-Agent': get_random_user_agent(),
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://www.wikidata.org/',
        'DNT': '1'
    }

# 修改：统一获取错误日志路径，现在接收log_file参数
def get_error_log_path(log_file=None):
    """获取统一的错误日志路径"""
    if log_file:
        return log_file
    else:
        # 1. 获取当前脚本的绝对路径和所在目录
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        
        # 2. 构建目标日志目录：../../experiments/logs（向上两级目录）
        target_log_dir = os.path.abspath(
            os.path.join(current_script_dir, "../../experiments/logs")
        )
        
        # 3. 确保日志目录存在（不存在则创建）
        os.makedirs(target_log_dir, exist_ok=True)
        
        # 4. 返回完整日志文件路径（统一文件名：prolog_inference_error.log）
        return os.path.join(target_log_dir, "prolog_inference_error.log")

def log_message(message, log_file=None):
    """统一的日志写入函数"""
    log_path = get_error_log_path(log_file)
    
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")
        print(message)  # 如果日志写入失败，至少打印到控制台
        
# -------------------------- 原有工具函数 --------------------------
def normalize_path(path):
    return str(path).replace('\\', '/')

def safe_consult(prolog, path):
    abs_path = os.path.abspath(path)
    abs_path = normalize_path(abs_path)
    query = f'consult("{abs_path}")'
    try:
        list(prolog.query(query))
        print(f"success: {abs_path}")
    except Exception as e:
        # 调用统一日志路径，记录consult失败错误
        error_log_path = get_error_log_path()
        with open(error_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"[SafeConsult Error] File: {abs_path}, Error: {str(e)}\n")
        print(f"fail: {abs_path}")
        print(f"error: {str(e)}")
        if not os.path.exists(abs_path):
            print(f"File not exist: {abs_path}")
        raise

def get_wikipedia_summary(wikidata_id, language='en', log_file=None):
    wikidata_url = f'https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json'
    
    headers = get_realistic_headers()
    
    for attempt in range(3):  # 重试3次
        try:
            # 添加随机延时，模拟人类行为
            if attempt > 0:
                time.sleep(random.uniform(2, 5))
            
            response = requests.get(wikidata_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                entities = data.get('entities', {})
                entity = entities.get(wikidata_id, {})
                sitelinks = entity.get('sitelinks', {})
                wikipedia_key = f'{language}wiki'
                sitelink = sitelinks.get(wikipedia_key, {})
                page_title = sitelink.get('title', None)

                if not page_title:
                    log_message(f"Cannot find Wikipedia page for Wikidata ID {wikidata_id} in language '{language}'.", log_file)
                    return None

                wikipedia_api_url = f'https://{language}.wikipedia.org/api/rest_v1/page/summary/{page_title}'
                
                # 获取Wikipedia摘要
                summary_response = requests.get(wikipedia_api_url, headers=headers, timeout=10)
                if summary_response.status_code == 200:
                    summary_data = summary_response.json()
                    summary = summary_data.get('extract', None)
                    log_message(f"Successfully get summary for Wikipedia page '{page_title}'.", log_file)
                    return summary
                else:
                    log_message(f"Cannot get summary for Wikipedia page '{page_title}' (status: {summary_response.status_code}).", log_file)
                    return None
                    
            elif response.status_code == 403:
                log_message(f"403 Forbidden for Wikidata ID {wikidata_id}. This may be due to rate limiting or bot protection.", log_file)
                if attempt < 2:  # 最多重试2次
                    wait_time = 10 + random.uniform(5, 10)
                    log_message(f"Waiting {wait_time:.1f} seconds before retry...", log_file)
                    time.sleep(wait_time)
                    continue
                return None
            else:
                log_message(f"HTTP {response.status_code} error for Wikidata ID {wikidata_id}", log_file)
                return None
                
        except requests.exceptions.RequestException as e:
            # 调用统一日志路径，记录请求错误
            log_message(f"[WikiData Request Error] ID: {wikidata_id}, Error: {str(e)}", log_file)
            if attempt < 2:  # 最多重试2次
                time.sleep(random.uniform(3, 7))
                continue
            return None
        except json.JSONDecodeError as e:
            log_message(f"JSON decode error for Wikidata ID {wikidata_id}: {e}", log_file)
            return None
    
    return None

def replace_special_characters2(s):
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    return s

def is_q_followed_by_digits(s): 
    pattern = r'^q\d+$'  
    return (bool(re.match(pattern, s)) and s != '_')

# -------------------------- 否定推理（修改日志路径） --------------------------
def negation_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list):
    prolog = Prolog()
    safe_consult(prolog, prolog_fact_file)
    safe_consult(prolog, prolog_query_file)
    
    with open(problem_file, 'r') as f:
        problems = json.load(f)
    with open(backup_file, 'r') as f:
        backup_dict = json.load(f)
    original_name_list = list(backup_dict.keys())
    modified_name_list = list(backup_dict.values())
    modified_name_list = [name.lower() for name in modified_name_list]
    with open(prolog_fact_file, 'r') as f:
        fact_content = f.read()
        fact_lines = fact_content.split('\n')
    with open(prolog_query_file, 'r') as f:
        query_content = f.read()
    key_list = list(problems.keys())
    query_list = list(problems.values())
    with open(log_entities_file, 'r') as f:
        log_entities = json.load(f)
    all_cnt = 0
    for idx, problem in tqdm(enumerate(query_list)):
        predicate = key_list[idx]
        useful_entitya = [line.split('(')[1].split(',')[0] for line in fact_lines if predicate in line]
        useful_entityb = [line.split(',')[1].split(')')[0].strip() for line in fact_lines if predicate in line]
        key = key_list[idx]+'(Entity_A,'
        if (key not in query_content) or ('\n'+key_list[idx]+'(' not in fact_content):
            continue
        for entitya in useful_entitya:
            cnt = 0
            if entitya.strip("'") not in modified_name_list or is_q_followed_by_digits(entitya):
                continue
            subject = original_name_list[modified_name_list.index(entitya.strip("'"))]
            # 查找subject_id（增加异常捕获）
            subject_id = None
            for entity in log_entities:
                if entity['entityLabel'] == subject :
                    subject_id = entity['entity'].split('/')[-1]
                    break
                elif entity['valueLabel'] == subject:
                    subject_id = entity['value'].split('/')[-1]
                    break
            if not subject_id:
                error_log_path = get_error_log_path()
                with open(error_log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[Negation Inference Error] Subject: {subject}, No Wikidata ID found\n")
                continue
            subject_description = get_wikipedia_summary(subject_id)
            if subject_description == None:
                continue
            for entityb in useful_entityb:
                new_problem = problem.replace('Entity_A', entitya).replace('Entity_B', entityb)
                try:
                    result = list(prolog.query(new_problem))
                    if len(result) == 0:
                        continue
                    if result:
                        # 查找object_（增加异常捕获）
                        if entityb.strip("'").lower() not in modified_name_list:
                            error_log_path = get_error_log_path()
                            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                                log_file.write(f"[Negation Inference Error] EntityB: {entityb}, Not in modified name list\n")
                            continue
                        object_ = original_name_list[modified_name_list.index(entityb.strip("'").lower())]
                        object_id = None
                        for entity in log_entities:
                            if entity['entityLabel'] == object_:
                                object_id = entity['entity'].split('/')[-1]
                            elif entity['valueLabel'] == object_:
                                object_id = entity['value'].split('/')[-1]
                        if not object_id:
                            error_log_path = get_error_log_path()
                            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                                log_file.write(f"[Negation Inference Error] Object: {object_}, No Wikidata ID found\n")
                            continue
                        object_description = get_wikipedia_summary(object_id)
                        if object_description == None:
                            continue
                        else:
                            description = subject_description + '\n' + object_description
                        evidence = []
                        for entity in log_entities:
                            if (entity['entityLabel'] == object_) or (entity['entityLabel'] == subject):
                                if replace_special_characters2(entity['propertyLabel']) == predicate:
                                    evidence.append((entity['entityLabel'], entity['propertyLabel'], entity['valueLabel']))
                        new_fact = {
                        "category": domain,
                        "reasoning": "Negation Inference",
                        "description": description,
                        "subject": subject,
                        "predicate": problem.split('(')[0],
                        "object": object_,
                        "evidence": evidence
                        }
                        new_fact_list.append(new_fact)
                        cnt += 1
                        all_cnt += 1
                except Exception as e:
                    # 替换为统一日志路径，增加推理类型标识
                    error_log_path = get_error_log_path()
                    with open(error_log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"[Negation Inference Error] EntityA: {entitya}, EntityB: {entityb}, Error: {str(e)}\n")
                    print(f"Error occurred while processing (Negation): {str(e)}")
                if cnt > 3:
                    break
        if all_cnt > 300:
            break
    return new_fact_list

# -------------------------- 复合推理（修改日志路径） --------------------------
def composite_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list):
    prolog = Prolog()
    safe_consult(prolog, prolog_fact_file)
    safe_consult(prolog, prolog_query_file)
    
    with open(problem_file, 'r') as f:
        problems = json.load(f)
    with open(backup_file, 'r') as f:
        backup_dict = json.load(f)
    original_name_list = list(backup_dict.keys())
    modified_name_list = list(backup_dict.values())
    modified_name_list = [name.lower() for name in modified_name_list]
    with open(prolog_fact_file, 'r') as f:
        fact_content = f.read()
    with open(prolog_query_file, 'r') as f:
        query_content = f.read()
    key_list = list(problems.keys())
    query_list = list(problems.values())
    with open(log_entities_file, 'r') as f:
        log_entities = json.load(f)
    for idx, problem in enumerate(query_list):
        key = key_list[idx]+'(Entity_A,'
        if (key not in query_content) or ('\n'+key_list[idx]+'(' not in fact_content):
            continue
        cnt = 0
        for soln in prolog.query(problem):
            if soln:
                try:
                    # 增加参数合法性检查
                    if 'Entity_A' not in soln or 'Entity_B' not in soln:
                        error_log_path = get_error_log_path()
                        with open(error_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[Composite Inference Error] Invalid solution: {soln}, Missing Entity_A/Entity_B\n")
                        continue
                    entity_a = soln['Entity_A'].strip("'")
                    entity_b = soln['Entity_B'].strip("'")
                    if entity_a not in modified_name_list or entity_b not in modified_name_list:
                        error_log_path = get_error_log_path()
                        with open(error_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[Composite Inference Error] EntityA: {entity_a}, EntityB: {entity_b}, Not in modified list\n")
                        continue
                    if is_q_followed_by_digits(entity_a) or is_q_followed_by_digits(entity_b):
                        continue
                    subject = original_name_list[modified_name_list.index(entity_a)]
                    subject_id = None
                    for entity in log_entities:
                        if entity['entityLabel'] == subject :
                            subject_id = entity['entity'].split('/')[-1]
                        elif entity['valueLabel'] == subject:
                            subject_id = entity['value'].split('/')[-1]
                    if not subject_id:
                        error_log_path = get_error_log_path()
                        with open(error_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[Composite Inference Error] Subject: {subject}, No Wikidata ID found\n")
                        break
                    subject_description = get_wikipedia_summary(subject_id)
                    if subject_description == None:
                        break
                    object_ = original_name_list[modified_name_list.index(entity_b.lower())]
                    object_id = None
                    for entity in log_entities:
                        if entity['entityLabel'] == object_:
                            object_id = entity['entity'].split('/')[-1]
                        elif entity['valueLabel'] == object_:
                            object_id = entity['value'].split('/')[-1]
                    if not object_id:
                        error_log_path = get_error_log_path()
                        with open(error_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[Composite Inference Error] Object: {object_}, No Wikidata ID found\n")
                        continue
                    object_description = get_wikipedia_summary(object_id)
                    if object_description == None:
                        continue
                    else:
                        description = subject_description + '\n' + object_description
                    evidence = []
                    for entity in log_entities:
                        if entity['entityLabel'] == object_ or entity['entityLabel'] == subject:
                            if replace_special_characters2(entity['propertyLabel']) == key_list[idx]:
                                evidence.append((entity['entityLabel'], entity['propertyLabel'], entity['valueLabel']))
                    new_fact = {
                        "category": domain,
                        "reasoning": "Composite Inference",
                        "description": description,
                        "subject": subject,
                        "predicate": problem.split('(')[0],
                        "object": object_,
                        "evidence": evidence
                    }
                    print(new_fact)
                    new_fact_list.append(new_fact)
                    cnt += 1
                except Exception as e:
                    # 替换为统一日志路径，增加推理类型标识
                    error_log_path = get_error_log_path()
                    with open(error_log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"[Composite Inference Error] Solution: {soln}, Error: {str(e)}\n")
                    print(f"Error occurred while processing (Composite): {str(e)}")
            if cnt > 10:
                break
    return new_fact_list

# -------------------------- 逆推理（修改日志路径） --------------------------
def inverse_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list):
    prolog = Prolog()
    prolog.query('abolish(all)')
    safe_consult(prolog, prolog_fact_file)
    safe_consult(prolog, prolog_query_file)
    
    with open(problem_file, 'r') as f:
        problems = json.load(f)
    with open(backup_file, 'r') as f:
        backup_dict = json.load(f)
    original_name_list = list(backup_dict.keys())
    modified_name_list = list(backup_dict.values())
    modified_name_list = [name.lower() for name in modified_name_list]
    with open(prolog_fact_file, 'r') as f:
        fact_content = f.read()
    with open(prolog_query_file, 'r') as f:
        query_content = f.read()
    key_list = list(problems.keys())
    query_list = list(problems.values())
    with open(log_entities_file, 'r') as f:
        log_entities = json.load(f)
    all_cnt = 0
    for idx, problem in enumerate(query_list):
        print(len(query_list))
        key = key_list[idx]+'(Entity_B,'
        if (key not in query_content) or ('\n'+key_list[idx]+'(' not in fact_content):
            continue
        cnt = 0
        try:
            result = list(prolog.query(problem))
        except Exception as e:
            error_log_path = get_error_log_path()
            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"[Inverse Inference Query Error] Problem: {problem}, Error: {str(e)}\n")
            print(f"Query error (Inverse): {str(e)}")
            continue
        for soln in result:
            if soln:
                try:
                    # 增加参数合法性检查
                    if 'Entity_A' not in soln or 'Entity_B' not in soln:
                        error_log_path = get_error_log_path()
                        with open(error_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[Inverse Inference Error] Invalid solution: {soln}, Missing Entity_A/Entity_B\n")
                        continue
                    entity_a = soln['Entity_A'].strip("'")
                    entity_b = soln['Entity_B'].strip("'")
                    if entity_a not in modified_name_list:
                        error_log_path = get_error_log_path()
                        with open(error_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[Inverse Inference Error] EntityA: {entity_a}, Not in modified list\n")
                        continue
                    subject = original_name_list[modified_name_list.index(entity_a)]
                    subject_id = None
                    for entity in log_entities:
                        if entity['entityLabel'] == subject :
                            subject_id = entity['entity'].split('/')[-1]
                        elif entity['valueLabel'] == subject:
                            subject_id = entity['value'].split('/')[-1]
                    if not subject_id:
                        error_log_path = get_error_log_path()
                        with open(error_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[Inverse Inference Error] Subject: {subject}, No Wikidata ID found\n")
                        continue
                    subject_description = get_wikipedia_summary(subject_id)
                    if subject_description == None:
                        continue
                    if entity_b.lower() not in modified_name_list:
                        error_log_path = get_error_log_path()
                        with open(error_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[Inverse Inference Error] EntityB: {entity_b}, Not in modified list\n")
                        continue
                    object_ = original_name_list[modified_name_list.index(entity_b.lower())]
                    object_id = None
                    for entity in log_entities:
                        if entity['entityLabel'] == object_:
                            object_id = entity['entity'].split('/')[-1]
                        elif entity['valueLabel'] == object_:
                            object_id = entity['value'].split('/')[-1]
                    if not object_id:
                        error_log_path = get_error_log_path()
                        with open(error_log_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f"[Inverse Inference Error] Object: {object_}, No Wikidata ID found\n")
                        continue
                    object_description = get_wikipedia_summary(object_id)
                    if object_description == None:
                        continue
                    else:
                        description = subject_description + '\n' + object_description
                    evidence = []
                    for entity in log_entities:
                        if entity['entityLabel'] == object_ and replace_special_characters2(entity['propertyLabel']) == key_list[idx]:
                            evidence.append((entity['entityLabel'], entity['propertyLabel'], entity['valueLabel']))
                    new_fact = {
                        "category": domain,
                        "reasoning": "Inverse Function Inference",
                        "description": description,
                        "subject": subject,
                        "predicate": problem.split('(')[0],
                        "object": object_,
                        "evidence": evidence
                    }
                    new_fact_list.append(new_fact)
                    cnt += 1
                    all_cnt += 1
                except Exception as e:
                    # 替换为统一日志路径，增加推理类型标识
                    error_log_path = get_error_log_path()
                    with open(error_log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"[Inverse Inference Error] Solution: {soln}, Error: {str(e)}\n")
                    print(f"Error occurred while processing (Inverse): {str(e)}")
        if all_cnt > 200:
            break
    return new_fact_list

# -------------------------- 传递推理（修改日志路径） --------------------------
def transitive_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list):
    prolog = Prolog()
    safe_consult(prolog, prolog_fact_file)
    safe_consult(prolog, prolog_query_file)
    
    with open(problem_file, 'r') as f:
        problems = json.load(f)
    with open(backup_file, 'r') as f:
        backup_dict = json.load(f)
    original_name_list = list(backup_dict.keys())
    modified_name_list = list(backup_dict.values())
    modified_name_list = [name.lower() for name in modified_name_list]
    with open(prolog_fact_file, 'r') as f:
        fact_content = f.read()
    with open(prolog_query_file, 'r') as f:
        query_content = f.read()
    key_list = list(problems.keys())
    query_list = list(problems.values())
    with open(log_entities_file, 'r') as f:
        log_entities = json.load(f)
    
    unique_facts = set()
    cnt = 0
    soln_set = set()
    for idx, problem in enumerate(query_list):
        key = key_list[idx]+'(Entity_A,'
        if (key not in query_content) or ('\n'+key_list[idx]+'(' not in fact_content):
            continue
        per_cnt = 0
        try:
            # 捕获Prolog查询异常
            prolog_results = list(prolog.query(problem))
        except Exception as e:
            error_log_path = get_error_log_path()
            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"[Transitive Inference Query Error] Problem: {problem}, Error: {str(e)}\n")
            print(f"Query error (Transitive): {str(e)}")
            continue
        for soln in prolog_results:  
            if soln and 'Entity_A' in soln and 'Entity_B' in soln:  
                soln_set.add((soln['Entity_A'], problem.split('(')[0], soln['Entity_B']))
    print(f"Total {len(soln_set)} facts found.")
    for soln_a, predicate, solnb in soln_set:
        try:
            entity_a = soln_a.strip("'")
            entity_b = solnb.strip("'")
            if entity_a not in modified_name_list or entity_b not in modified_name_list:
                error_log_path = get_error_log_path()
                with open(error_log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[Transitive Inference Error] EntityA: {entity_a}, EntityB: {entity_b}, Not in modified list\n")
                continue
            if is_q_followed_by_digits(entity_a) or is_q_followed_by_digits(entity_b):
                continue
            subject = original_name_list[modified_name_list.index(entity_a)]
            subject_id = None
            for entity in log_entities:
                if entity['subjectValue'] == subject :
                    subject_id = entity['subject']
            if not subject_id:
                error_log_path = get_error_log_path()
                with open(error_log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[Transitive Inference Error] Subject: {subject}, No Wikidata ID found\n")
                continue
            subject_description = get_wikipedia_summary(subject_id)
            if subject_description is None:
                break
            if entity_b.lower() not in modified_name_list:
                error_log_path = get_error_log_path()
                with open(error_log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[Transitive Inference Error] EntityB: {entity_b}, Not in modified list\n")
                continue
            object_ = original_name_list[modified_name_list.index(entity_b.lower())]
            object_id = None
            for entity in log_entities:
                if entity['anotherObjectValue'] == object_:
                    object_id = entity['anotherObject'].split('/')[-1]
            if not object_id:
                error_log_path = get_error_log_path()
                with open(error_log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[Transitive Inference Error] Object: {object_}, No Wikidata ID found\n")
                continue
            object_description = get_wikipedia_summary(object_id)
            if object_description is None:
                continue
            else:
                description = subject_description + '\n' + object_description
            evidence = []
            for entity in log_entities:
                if entity['subjectValue'] == subject or entity['anotherObjectValue'] == object_:
                    if replace_special_characters2(entity['predicateValue']) == predicate.split('trans_')[1]:
                        evidence.append((entity['subjectValue'], entity['predicateValue'], entity['objectValue']))
                        evidence.append((entity['objectValue'], entity['predicateValue'], entity['anotherObjectValue']))
            
            fact_tuple = (subject, predicate, object_)
            if fact_tuple in unique_facts:
                continue  
            unique_facts.add(fact_tuple) 

            new_fact = {
                "category": domain,
                "reasoning": "Transitive Inference",
                "description": description,
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "evidence": evidence
            }
            new_fact_list.append(new_fact)
            cnt += 1
            per_cnt += 1
            print(f"Processed {cnt} facts.")
        except Exception as e:
            # 替换为统一日志路径，增加推理类型标识
            error_log_path = get_error_log_path()
            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"[Transitive Inference Error] EntityA: {soln_a}, EntityB: {solnb}, Error: {str(e)}\n")
            print(f"Error occurred while processing (Transitive): {str(e)}")
        if cnt > 200:
            break
    return new_fact_list