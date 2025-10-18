# src/backend/dynamic_dataset.py
import os
import json
import random
import time
import requests
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.parse import unquote, quote
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
from work_models.prolog_inference import (
    normalize_path, safe_consult, get_wikipedia_summary, 
    replace_special_characters2, is_q_followed_by_digits, 
    negation_prolog_inference, composite_prolog_inference, inverse_prolog_inference
)
from work_models.question_generation import inverse_template, negation_template, composite_template
from evaluate import get_wikidata_id_from_wikipedia_url

# ==================== 改进的浏览器模拟 ====================

# 更完整的用户代理列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0"
]

def get_random_user_agent():
    """获取随机用户代理"""
    return random.choice(USER_AGENTS)

def create_realistic_session():
    """创建带完整浏览器特征的会话"""
    session = requests.Session()
    session.headers.update({
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://en.wikipedia.org/',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    })
    return session

def setup_requests_session():
    """设置带用户代理的requests会话"""
    session = create_realistic_session()
    return session

def setup_sparql_wrapper():
    """设置SPARQLWrapper的用户代理"""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader('User-Agent', get_random_user_agent())
    return sparql

def log_message(message, log_file=None):
    """统一的日志写入函数"""
    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}")
            print(message)  # 如果日志写入失败，至少打印到控制台
    else:
        print(message)

# ==================== 改进的API请求函数 ====================

def make_wikipedia_request(url, params, max_retries=3, log_file=None):
    """统一的Wikipedia请求处理函数，带智能重试"""
    
    for attempt in range(max_retries):
        try:
            # 随机延时，模拟人类行为
            delay = random.uniform(2.0, 4.0) + (attempt * 2)  # 递增延时
            time.sleep(delay)
            
            headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }
            
            # 创建会话，保持连接
            session = requests.Session()
            session.headers.update(headers)
            
            response = session.get(url, params=params, timeout=15)
            
            # 处理不同状态码
            if response.status_code == 200:
                if response.headers.get('content-type', '').startswith('application/json'):
                    return response.json()
                else:
                    log_message(f"Non-JSON response: {response.text[:200]}", log_file)
                    return None
            elif response.status_code == 403:
                log_message(f"403 Forbidden - likely anti-bot protection. Attempt {attempt + 1}/{max_retries}", log_file)
                if attempt < max_retries - 1:
                    # 等待更长时间后重试
                    time.sleep(10 + random.uniform(5, 10))
                    continue
                return None
            elif response.status_code == 429:
                wait_time = 30 + (attempt * 30)  # 递增等待时间
                log_message(f"Rate limited. Waiting {wait_time} seconds...", log_file)
                time.sleep(wait_time)
                continue
            else:
                log_message(f"HTTP {response.status_code} error", log_file)
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
                
        except requests.exceptions.RequestException as e:
            log_message(f"Request error: {e}", log_file)
            if attempt < max_retries - 1:
                time.sleep(5 + random.uniform(2, 5))
                continue
            return None
        except json.JSONDecodeError as e:
            log_message(f"JSON decode error: {e}", log_file)
            return None
    
    return None

# ==================== 改进的实体获取函数 ====================

def get_wikidata_id_from_wikipedia_url_improved(wikipedia_url, max_retries=2, log_file=None):
    """从Wikipedia URL获取Wikidata ID，带更好的错误处理"""
    title = wikipedia_url.split("/")[-1]
    
        
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "format": "json"
    }
    
    data = make_wikipedia_request(url, params, log_file=log_file)
    if not data:
        return None
    
    pages = data.get("query", {}).get("pages", {})
    
    for page_id, page_info in pages.items():
        if "pageprops" in page_info and "wikibase_item" in page_info["pageprops"]:
            return {
                "entity": f"http://www.wikidata.org/entity/{page_info['pageprops']['wikibase_item']}",
                "entityLabel": title.replace("_", " ")
            }
    
    return None

def generate_dynamic_dataset(rule_choice, domain_choice, current_script_dir, log_file=None):
    """动态生成数据集的主函数"""
    
    # 设置全局会话和SPARQLWrapper
    global_session = setup_requests_session()
    global_sparql = setup_sparql_wrapper()
    
    # 推理规则与领域映射
    rule_map = {"1": "inverse", "2": "negation", "3": "composite"}
    reasoning_type = rule_map.get(rule_choice, "inverse")
    log_message(f"Selected reasoning rule: {reasoning_type}", log_file)
    
    domains = ["geography", "history", "health", "mathematics", "nature", 
               "people", "society", "technology", "culture"]
    domain = domains[int(domain_choice) - 1] if (
        domain_choice.isdigit() and 1 <= int(domain_choice) <= 9
    ) else "geography"
    log_message(f"Selected domain: {domain}", log_file)
    log_message(f"Processing domain: {domain}", log_file)
    
    questions, standard_answers = [], []
    current_file_path = ""
    
    try:
        # -------------------------- 步骤1：保存分类链接 --------------------------
        log_message("Step 1: Saving category links...", log_file)
        category_dir = os.path.abspath(os.path.join(
            current_script_dir, '../../', 'data', 'dataset', 'category'
        ))
        os.makedirs(category_dir, exist_ok=True)
        current_file_path = os.path.join(category_dir, f'{domain.lower()}_links.txt')
        
        save_links_to_file(category=domain, filename=current_file_path, log_file=log_file)
        if not os.path.exists(current_file_path) or os.path.getsize(current_file_path) == 0:
            raise FileNotFoundError(f"Category file empty/missing: {current_file_path}")
        log_message(f"Category links saved to {current_file_path}", log_file)

        # -------------------------- 步骤2：生成selected_categories --------------------------
        selected_categories = []
        with open(current_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line and 'Category:' in line:
                    # 提取分类名并进行URL解码
                    encoded_category = line.split('/Category:')[-1].strip()
                    # URL解码分类名
                    category_name = unquote(encoded_category)
                    selected_categories.append(category_name)
        if not selected_categories:
            selected_categories.append(domain)
            log_message(f"Warning: Use default category '{domain}'", log_file)

        # -------------------------- 步骤3：保存实体链接 --------------------------
        log_message("Step 3: Saving entity links...", log_file)
        selected_dir = os.path.join(category_dir, 'selected')
        os.makedirs(selected_dir, exist_ok=True)
        current_file_path = os.path.join(selected_dir, f'{domain.lower()}_wiki_links.txt')
        
        max_pages_limit = 50  
        wikipedia_urls = save_entity_links_to_file(
            categories=selected_categories, 
            filename=current_file_path, 
            max_pages=max_pages_limit,
            log_file=log_file
        )
        if not os.path.exists(current_file_path) or os.path.getsize(current_file_path) == 0:
            raise FileNotFoundError(f"Entity links file empty/missing: {current_file_path}")
        log_message(f"Entity links saved to {current_file_path} (total: {len(wikipedia_urls)} links)", log_file)

        # -------------------------- 步骤4：生成Wikidata实体JSON --------------------------
        log_message("Step 4: Generating Wikidata entities...", log_file)
        wiki_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'wiki'))
        os.makedirs(wiki_dir, exist_ok=True)
        current_file_path = os.path.join(wiki_dir, f'{domain}_useful_entities.json')
        
        wikidata_entities = []
        successful_count = 0
        
        for url in tqdm(wikipedia_urls, desc=f"Getting Wikidata IDs for {domain}"):
            result = get_wikidata_id_from_wikipedia_url_improved(url, log_file=log_file)
            if result:
                wikidata_entities.append(result)
                successful_count += 1
            
            # 减少请求间隔，但保持随机性
            time.sleep(random.uniform(1.5, 2.5))
        
        log_message(f"Successfully retrieved {successful_count}/{len(wikipedia_urls)} Wikidata entities", log_file)
        
        with open(current_file_path, 'w', encoding='utf-8') as f_json:
            json.dump(wikidata_entities if wikidata_entities else [], f_json, indent=4)
        if os.path.getsize(current_file_path) == 0:
            raise ValueError(f"Wikidata entity file empty: {current_file_path}")
        log_message(f"Wikidata entities saved to {current_file_path} (total: {len(wikidata_entities)} entities)", log_file)

        # 如果没有成功获取实体，直接报错
        if len(wikidata_entities) == 0:
            raise ValueError(f"No Wikidata entities retrieved for {domain} domain")


        # -------------------------- 步骤5：处理transitive实体 --------------------------
        log_message("Step 5: Processing transitive entities...", log_file)
        transitive_wiki_dir = os.path.join(wiki_dir, 'transitive')
        os.makedirs(transitive_wiki_dir, exist_ok=True)
        current_file_path = os.path.join(transitive_wiki_dir, f'{domain}_useful_entities.json')
        
        try:
            wikidata_file = os.path.join(wiki_dir, f'{domain}_useful_entities.json')
            with open(wikidata_file, 'r', encoding='utf-8') as f_json:
                data = json.load(f_json)
            
            class_list = [(item['entity'].split('/')[-1], item['entityLabel']) for item in data]
            useful_list = []
            for entity_id, entity_name in tqdm(class_list, desc="Processing transitive entities"):
                useful_entities = get_entity_info(entity_id, entity_name)
                if useful_entities:
                    useful_list.extend(useful_entities)
            
            predicate_set = {ent["predicate"] for ent in useful_list}
            predicate_label_map = get_predicate_labels(predicate_set)
            updated_entities = replace_predicates_with_labels(useful_list, predicate_label_map)
            
            with open(current_file_path, 'w', encoding='utf-8') as f_json:
                json.dump(updated_entities if updated_entities else [], f_json, indent=4)
            log_message(f"Transitive entities saved to {current_file_path}", log_file)
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Parse Wikidata file failed: {str(e)}")

        # -------------------------- 步骤6：构建transitive Prolog --------------------------
        log_message("Step 6: Building transitive Prolog...", log_file)
        transitive_pl_dir = os.path.join(transitive_wiki_dir, 'pl_files')
        os.makedirs(transitive_pl_dir, exist_ok=True)
        current_file_path = os.path.join(transitive_wiki_dir, f'{domain}_useful_entities.json')
        
        backup_list = {}
        log_entities = []
        pl_file = os.path.join(transitive_pl_dir, f'{domain}.pl')  # 在循环前定义
        
        with open(current_file_path, 'r', encoding='utf-8') as f_json:
            entities = json.load(f_json)
            
        for entity in tqdm(entities, desc="Generating transitive Prolog"):
            save_to_pl_file(entity, pl_file, backup_list)
            if entity not in log_entities:
                log_entities.append(entity)
        
        if entities:  
            process_prolog_file(pl_file, pl_file)
            log_message(f"Transitive Prolog saved to {transitive_pl_dir}", log_file)
        else:
            log_message(f"No entities found for {domain}, skipping Prolog generation", log_file)

        # -------------------------- 步骤7：构建普通Wiki Prolog --------------------------
        log_message("Step 7: Building Wiki Prolog...", log_file)
        wiki_pl_dir = os.path.join(wiki_dir, 'pl_files')
        os.makedirs(wiki_pl_dir, exist_ok=True)
        current_file_path = os.path.join(wiki_dir, f'{domain}_useful_entities.json')
        
        backup_list = {}
        log_entities = []
        pl_file = os.path.join(wiki_pl_dir, f'{domain}.pl')  # 在循环前定义
        
        utils_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'utils'))
        properties_file = os.path.join(utils_dir, 'wiki_property_cat_v1.xlsx')
        prop_list = get_prop_list(properties_file)
        
        with open(current_file_path, 'r', encoding='utf-8') as f_json:
            entities = json.load(f_json)
        for entity in tqdm(entities, desc="Generating Wiki Prolog"):
            entity_id = entity['entity'].split('/')[-1]
            entity_name = entity['entityLabel']
            related_entities = get_related_entity_list(entity_id, entity_name, prop_list)
            save_to_pl_file1(related_entities, pl_file, backup_list)
            for rel_ent in related_entities:
                if rel_ent not in log_entities:
                    log_entities.append(rel_ent)
        
        if entities:  
            process_prolog_file1(pl_file, pl_file)
            log_message(f"Wiki Prolog saved to {wiki_pl_dir}", log_file)
        else:
            log_message(f"No entities found for {domain}, skipping Wiki Prolog generation", log_file)

        # -------------------------- 步骤8：Prolog推理 --------------------------
        log_message(f"Step 8: Prolog inference for {reasoning_type} rule...", log_file)
        new_fact_list = []  
        prolog_fact_file = os.path.join(wiki_pl_dir, f'{domain}.pl')
        backup_file = os.path.join(wiki_dir, 'backup', f'{domain}_backup_list.json')
        log_entities_file = os.path.join(wiki_dir, 'log', f'{domain}_log.json')
        prolog_query_file = os.path.join(current_script_dir, '..', 'prolog_rules', f'{reasoning_type}_rules.pl')
        problem_file = os.path.join(current_script_dir, '..', 'prolog_rules', f'{reasoning_type}_problem_dict.json')
        
        # 新事实保存路径
        derived_dir = os.path.abspath(os.path.join(
            current_script_dir, '../../', 'data', 'dataset', 'derived', reasoning_type
        ))
        os.makedirs(derived_dir, exist_ok=True)
        current_file_path = os.path.join(derived_dir, f'{domain}_new_facts.json')
        
        # 调用推理函数
        if reasoning_type == 'inverse':
            new_fact_list = inverse_prolog_inference(
                prolog_fact_file, prolog_query_file, problem_file, 
                backup_file, log_entities_file, domain, new_fact_list
            )
        elif reasoning_type == 'composite':
            new_fact_list = composite_prolog_inference(
                prolog_fact_file, prolog_query_file, problem_file, 
                backup_file, log_entities_file, domain, new_fact_list
            )
        elif reasoning_type == 'negation':
            new_fact_list = negation_prolog_inference(
                prolog_fact_file, prolog_query_file, problem_file, 
                backup_file, log_entities_file, domain, new_fact_list
            )
        
        # 保存新事实
        with open(current_file_path, 'w', encoding='utf-8') as f_json:
            json.dump(new_fact_list if new_fact_list else [], f_json, indent=4)
        if os.path.getsize(current_file_path) == 0:
            raise ValueError(f"New facts file empty: {current_file_path}")
        log_message(f"New facts saved to {current_file_path} (total: {len(new_fact_list)} facts)", log_file)

        # -------------------------- 步骤9：生成QA对 --------------------------
        log_message(f"Step 9: Generating QA pairs for {domain}...", log_file)
        generated_qa_list = []  
        new_facts_file = current_file_path
        current_file_path = new_facts_file
        
        # 读取新事实
        with open(current_file_path, 'r', encoding='utf-8') as f_json:
            new_facts = json.load(f_json)
        
        # 生成QA对
        prolog_rules_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'prolog_rules'))
        problem_dict_path = os.path.join(prolog_rules_dir, f'{reasoning_type}_problem_dict.json')
        if reasoning_type == 'inverse':
            inverse_template(new_facts, generated_qa_list, problem_dict_path)
        elif reasoning_type == 'negation':
            negation_template(new_facts, generated_qa_list, problem_dict_path)
        elif reasoning_type == 'composite':
            backup_dict_path = os.path.join(wiki_dir, 'backup', f'{domain}_backup_list.json')
            composite_template(new_facts, generated_qa_list, problem_dict_path, backup_dict_path)
        
        # 保存QA对
        qa_output_dir = os.path.abspath(os.path.join(
            current_script_dir, '../../', 'data', 'dataset', 'dynamic', reasoning_type
        ))
        os.makedirs(qa_output_dir, exist_ok=True)
        current_file_path = os.path.join(qa_output_dir, f'{domain}_qa.json')
        
        with open(current_file_path, 'w', encoding='utf-8') as f_json:
            json.dump(generated_qa_list if generated_qa_list else [], f_json, indent=4, ensure_ascii=False)
        if os.path.getsize(current_file_path) == 0:
            raise ValueError(f"QA file empty: {current_file_path}")
        log_message(f"QA pairs saved to {current_file_path} (total: {len(generated_qa_list)} pairs)", log_file)
        log_message("Dataset generation completed", log_file)
        
        return current_file_path

    except Exception as e:
        log_message(f"Error in dynamic dataset generation: {str(e)}", log_file)
        # 直接抛出异常，不要使用备用方案
        raise




# 全局会话和SPARQLWrapper实例
requests_session = setup_requests_session()
sparql_wrapper = setup_sparql_wrapper()