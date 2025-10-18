# -*- coding: utf-8 -*-

import requests
import time
import random
import json
import os
from urllib.parse import quote, unquote
from tqdm import tqdm

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
    }

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
        
def make_wikipedia_request(url, params, max_retries=3, log_file=None):
    """统一的Wikipedia请求处理函数"""
    
    for attempt in range(max_retries):
        try:
            # 随机延时，模拟人类行为
            delay = random.uniform(0 , 0.5) 
            time.sleep(delay)
            
            headers = get_realistic_headers()
            
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

def get_category_pages(category, continue_token=None, log_file=None):
    """获取分类页面 - 重写版本"""
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "max",
        "format": "json",
        "cmcontinue": continue_token
    }

    data = make_wikipedia_request(base_url, params, log_file=log_file)
    if not data:
        return [], None
    
    pages = data.get("query", {}).get("categorymembers", [])
    next_continue_token = data.get("continue", {}).get("cmcontinue", None)
    
    return pages, next_continue_token


def save_links_to_file(category, filename, log_file=None):
    """保存分类链接到文件 - 修复版本"""
    all_links = []
    continue_token = None
    
    # 限制请求数量，避免过度请求
    max_iterations = 20
    iteration = 0
    
    while iteration < max_iterations:
        pages, continue_token = get_category_pages(category, continue_token, log_file)
        
        if not pages:
            log_message(f"No pages found for category {category}", log_file)
            break
            
        for page in pages:
            page_title = page["title"]
            # 只保存分类页面（以"Category:"开头的页面）
            if page_title.startswith("Category:"):
                category_name = page_title[9:]  # 去掉 "Category:" 前缀
                page_link = f"https://en.wikipedia.org/wiki/Category:{quote(category_name)}"
                all_links.append(page_link)

        if not continue_token or len(all_links) > 100:  # 限制分类数量
            break

        iteration += 1
        log_message(f"Iteration {iteration}: collected {len(all_links)} category links", log_file)
    
    with open(filename, "w", encoding='utf-8') as file:
        for link in all_links:
            file.write(link + "\n")

    log_message(f"Saved {len(all_links)} category links to {filename}", log_file)

def get_category_members(category, continue_token=None, log_file=None):
    """获取分类成员 - 重写版本"""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "max"
    }
    if continue_token:
        params["cmcontinue"] = continue_token

    data = make_wikipedia_request(url, params, log_file=log_file)
    if not data:
        return [], None
    
    pages = data.get('query', {}).get('categorymembers', [])
    continue_token = data.get('continue', {}).get('cmcontinue')
    
    return pages, continue_token


def extract_entity_pages(category, max_pages, current_page_count, log_file=None):
    """提取实体页面 - 重写版本"""
    entity_pages = []
    subcategories = []
    continue_token = None
    failed_requests = 0
    max_failures = 3  # 最多允许3次连续失败

    while current_page_count < max_pages and failed_requests < max_failures:
        pages, continue_token = get_category_members(category, continue_token, log_file)

        if not pages:
            failed_requests += 1
            log_message(f"Failed to get pages for category {category}, attempt {failed_requests}", log_file)
            continue

        for page in pages:
            if current_page_count >= max_pages:
                return entity_pages, current_page_count, True  

            # 检查是否为分类页面
            if page['title'].startswith("Category:"):
                subcategories.append(page['title'][9:])
            else:
                # 实体页面
                page_title = page["title"].replace(" ", "_")
                if page_title.startswith("List_of") or page_title.startswith("Template"):
                    continue
                
                page_link = f"https://en.wikipedia.org/wiki/{quote(page_title)}"
                entity_pages.append(page_link)
                current_page_count += 1  
                failed_requests = 0  # 重置失败计数

        if not continue_token or current_page_count >= max_pages:  
            break
    
    return entity_pages, current_page_count, failed_requests >= max_failures


def save_entity_links_to_file(categories, filename, max_pages=200, log_file=None):
    """保存实体链接到文件 - 重写版本"""
    all_links = []
    current_page_count = 0
    stop_flag = False

    for category in categories:
        if stop_flag or current_page_count >= max_pages:
            break
            
        log_message(f"Processing category: {category}", log_file)
        new_links, current_page_count, stop_flag = extract_entity_pages(
            category, max_pages - current_page_count, current_page_count, log_file
        )
        all_links.extend(new_links)
        
        if stop_flag:
            log_message(f"Stopping due to failures in category {category}", log_file)
            break
            
        # 分类间更长的延时
        if len(categories) > 1:
            time.sleep(random.uniform(3.0, 5.0))
    
    with open(filename, "w", encoding='utf-8') as file:
        for link in all_links:
            file.write(link + "\n")

    log_message(f"Saved {len(all_links)} links to {filename}. Total pages: {current_page_count}", log_file)
    return all_links

def get_wikidata_id_from_wikipedia_url(wikipedia_url):
    """从Wikipedia URL获取Wikidata ID - 重写版本"""
    title = wikipedia_url.split("/")[-1]
    
    # 跳过可能有问题的页面
    if any(keyword in title.lower() for keyword in ['list_of', 'template', 'category']):
        return None
        
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageprops",
        "format": "json"
    }
    
    data = make_wikipedia_request(url, params)
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