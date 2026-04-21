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

prolog = Prolog()

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
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        target_log_dir = os.path.abspath(
            os.path.join(current_script_dir, "../../experiments/logs")
        )
        os.makedirs(target_log_dir, exist_ok=True)
        return os.path.join(target_log_dir, "prolog_inference_error.log")

def log_message(message, log_file=None):
    """统一的日志写入函数"""
    log_path = get_error_log_path(log_file)
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")
        print(message)

# -------------------------- 工具函数 --------------------------
def normalize_path(path):
    return str(path).replace('\\', '/')

def safe_consult(prolog, path):
    abs_path = os.path.abspath(path)
    abs_path = normalize_path(abs_path)
    query = f'consult("{abs_path}")'
    try:
        prolog.query(query)
        print(f"success: {abs_path}")
    except Exception as e:
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
    
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(random.uniform(2, 5))
            response = requests.get(wikidata_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                entity = data.get('entities', {}).get(wikidata_id, {})
                sitelinks = entity.get('sitelinks', {})
                page_title = sitelinks.get(f'{language}wiki', {}).get('title', None)

                if not page_title:
                    log_message(f"Cannot find Wikipedia page for {wikidata_id}", log_file)
                    return None

                summary_url = f'https://{language}.wikipedia.org/api/rest_v1/page/summary/{page_title}'
                summary_resp = requests.get(summary_url, headers=headers, timeout=10)
                
                if summary_resp.status_code == 200:
                    return summary_resp.json().get('extract')
            elif response.status_code == 403 and attempt < 2:
                time.sleep(10 + random.uniform(5, 10))
                continue
        except Exception as e:
            log_message(f"[WikiData Error] {wikidata_id}: {str(e)}", log_file)
            if attempt < 2:
                time.sleep(random.uniform(3, 7))
                continue
    return None

def replace_special_characters2(s):
    return re.sub(r'[^a-zA-Z0-9]', '_', s)

def is_q_followed_by_digits(s):
    return bool(re.match(r'^q\d+$', s)) and s != '_'

# -------------------------- 否定推理 --------------------------
def negation_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list):
    prolog.query("abolish(_)")
    safe_consult(prolog, prolog_fact_file)
    safe_consult(prolog, prolog_query_file)
    
    with open(problem_file, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    with open(backup_file, 'r', encoding='utf-8') as f:
        backup_dict = json.load(f)
    original_name_list = list(backup_dict.keys())
    modified_name_list = [v.lower() for v in backup_dict.values()]
    
    with open(prolog_fact_file, 'r', encoding='utf-8') as f:
        fact_lines = f.read().split('\n')
    with open(prolog_query_file, 'r', encoding='utf-8') as f:
        query_content = f.read()
    with open(log_entities_file, 'r', encoding='utf-8') as f:
        log_entities = json.load(f)

    key_list = list(problems.keys())
    query_list = list(problems.values())
    all_cnt = 0

    for idx, problem in tqdm(enumerate(query_list)):
        predicate = key_list[idx]
        useful_entitya = [line.split('(')[1].split(',')[0] for line in fact_lines if predicate in line]
        useful_entityb = [line.split(',')[1].split(')')[0].strip() for line in fact_lines if predicate in line]
        key = f"{key_list[idx]}(Entity_A,"
        if key not in query_content or f"\n{key_list[idx]}(" not in fact_content:
            continue

        for entitya in useful_entitya:
            cnt = 0
            ea_stripped = entitya.strip("'")
            if ea_stripped not in modified_name_list or is_q_followed_by_digits(ea_stripped):
                continue

            subject = original_name_list[modified_name_list.index(ea_stripped)]
            subject_id = None
            for ent in log_entities:
                if ent.get('entityLabel') == subject:
                    subject_id = ent['entity'].split('/')[-1]
                    break
                elif ent.get('valueLabel') == subject:
                    subject_id = ent['value'].split('/')[-1]
                    break
            if not subject_id:
                continue
            subj_desc = get_wikipedia_summary(subject_id)
            if not subj_desc:
                continue

            for entityb in useful_entityb:
                new_problem = problem.replace('Entity_A', entitya).replace('Entity_B', entityb)
                try:
                    res = list(prolog.query(new_problem))
                    if not res:
                        continue

                    eb_stripped = entityb.strip("'").lower()
                    if eb_stripped not in modified_name_list:
                        continue
                    obj = original_name_list[modified_name_list.index(eb_stripped)]
                    obj_id = None
                    for ent in log_entities:
                        if ent.get('entityLabel') == obj:
                            obj_id = ent['entity'].split('/')[-1]
                            break
                        elif ent.get('valueLabel') == obj:
                            obj_id = ent['value'].split('/')[-1]
                            break
                    if not obj_id:
                        continue
                    obj_desc = get_wikipedia_summary(obj_id)
                    if not obj_desc:
                        continue

                    evidence = []
                    for ent in log_entities:
                        if ent.get('entityLabel') in (obj, subject):
                            if replace_special_characters2(ent.get('propertyLabel', '')) == predicate:
                                evidence.append((ent['entityLabel'], ent['propertyLabel'], ent['valueLabel']))

                    new_fact = {
                        "category": domain,
                        "reasoning": "Negation Inference",
                        "description": f"{subj_desc}\n{obj_desc}",
                        "subject": subject,
                        "predicate": problem.split('(')[0],
                        "object": obj,
                        "evidence": evidence
                    }
                    new_fact_list.append(new_fact)
                    cnt += 1
                    all_cnt += 1
                    if cnt > 3:
                        break
                except Exception:
                    continue
            if all_cnt > 300:
                break
    return new_fact_list

# -------------------------- 复合推理 --------------------------
def composite_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list):
    prolog.query("abolish(_)")
    safe_consult(prolog, prolog_fact_file)
    safe_consult(prolog, prolog_query_file)
    
    with open(problem_file, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    with open(backup_file, 'r', encoding='utf-8') as f:
        backup_dict = json.load(f)
    original_names = list(backup_dict.keys())
    modified_names = [v.lower() for v in backup_dict.values()]
    
    with open(prolog_fact_file, 'r', encoding='utf-8') as f:
        fact_content = f.read()
    with open(prolog_query_file, 'r', encoding='utf-8') as f:
        query_content = f.read()
    with open(log_entities_file, 'r', encoding='utf-8') as f:
        log_entities = json.load(f)

    for idx, problem in enumerate(problems.values()):
        key = f"{list(problems.keys())[idx]}(Entity_A,"
        if key not in query_content or f"\n{list(problems.keys())[idx]}(" not in fact_content:
            continue
        cnt = 0
        for soln in prolog.query(problem):
            if 'Entity_A' not in soln or 'Entity_B' not in soln:
                continue
            ea = soln['Entity_A'].strip("'")
            eb = soln['Entity_B'].strip("'")
            if ea not in modified_names or eb not in modified_names:
                continue
            if is_q_followed_by_digits(ea) or is_q_followed_by_digits(eb):
                continue

            subj = original_names[modified_names.index(ea)]
            obj = original_names[modified_names.index(eb.lower())]
            new_fact_list.append({
                "category": domain,
                "reasoning": "Composite Inference",
                "subject": subj,
                "predicate": problem.split('(')[0],
                "object": obj,
                "description": "",
                "evidence": []
            })
            cnt += 1
            if cnt >= 10:
                break
    return new_fact_list

# -------------------------- 逆推理（✅ 最终正确版） --------------------------
def inverse_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list):
    prolog.query("abolish(_)")  # 🔥 修复：统一正确清空
    safe_consult(prolog, prolog_fact_file)
    safe_consult(prolog, prolog_query_file)
    
    with open(problem_file, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    with open(backup_file, 'r', encoding='utf-8') as f:
        backup_dict = json.load(f)
    original_names = list(backup_dict.keys())
    modified_names = [v.lower() for v in backup_dict.values()]
    
    with open(prolog_fact_file, 'r', encoding='utf-8') as f:
        fact_content = f.read()
    with open(prolog_query_file, 'r', encoding='utf-8') as f:
        query_content = f.read()
    with open(log_entities_file, 'r', encoding='utf-8') as f:
        log_entities = json.load(f)

    for problem in problems.values():
        try:
            for soln in prolog.query(problem):
                ea = soln.get('Entity_A', '').strip("'")
                eb = soln.get('Entity_B', '').strip("'")
                if ea not in modified_names:
                    continue
                subj = original_names[modified_names.index(ea)]
                new_fact_list.append({
                    "category": domain,
                    "reasoning": "Inverse Function Inference",
                    "subject": subj,
                    "predicate": problem.split('(')[0],
                    "object": original_names[modified_names.index(eb.lower())] if eb.lower() in modified_names else eb,
                    "description": "",
                    "evidence": []
                })
        except Exception:
            continue
    return new_fact_list

# -------------------------- 传递推理 --------------------------
def transitive_prolog_inference(prolog_fact_file, prolog_query_file, problem_file, backup_file, log_entities_file, domain, new_fact_list):
    prolog.query("abolish(_)")
    safe_consult(prolog, prolog_fact_file)
    safe_consult(prolog, prolog_query_file)
    
    with open(problem_file, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    with open(backup_file, 'r', encoding='utf-8') as f:
        backup_dict = json.load(f)
    with open(log_entities_file, 'r', encoding='utf-8') as f:
        log_entities = json.load(f)

    soln_set = set()
    for problem in problems.values():
        try:
            for soln in prolog.query(problem):
                if 'Entity_A' in soln and 'Entity_B' in soln:
                    soln_set.add((soln['Entity_A'], problem.split('(')[0], soln['Entity_B']))
        except Exception:
            continue

    for a, pred, b in soln_set:
        new_fact_list.append({
            "category": domain,
            "reasoning": "Transitive Inference",
            "subject": a.strip("'"),
            "predicate": pred,
            "object": b.strip("'"),
            "description": "",
            "evidence": []
        })
    return new_fact_list