from SPARQLWrapper import SPARQLWrapper, JSON
import re
import pandas as pd
import json
import os
from tqdm import tqdm
from collections import defaultdict
import argparse

def get_related_entity_list(entity_id, entity_name, prop_list):
    """Get the related entities of the given entity"""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = """
    SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {
        wd:%s ?p ?statement .
        ?statement ?ps ?value .
        ?property wikibase:claim ?p.
        ?property wikibase:statementProperty ?ps.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 50
    """ % entity_id
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(30)
    try:
        results = sparql.query().convert()
        entities = []
        if len(results["results"]["bindings"]) == 0:
            return entities
        for result in results["results"]["bindings"]:
            prop = result["property"]["value"].split('/')[-1]
            if prop not in prop_list:
                continue
            entities.append({
                "entity": entity_id,
                "entityLabel": entity_name,
                "property": result["property"]["value"].split('/')[-1],
                "propertyLabel": result["propertyLabel"]["value"],
                "value": result["value"]["value"].split('/')[-1],
                "valueLabel": result["valueLabel"]["value"]
            })
        return entities
    except Exception as e:
        # 1. 获取当前函数所在脚本的绝对路径和目录
        current_script_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_script_path)
        
        # 2. 向上跳2级 → 定位到 experiments/logs 目录
        experiments_log_dir = os.path.abspath(os.path.join(
            current_script_dir, "../../experiments/logs"
        ))
        # 3. 确保 experiments/logs 目录存在（避免写入失败）
        os.makedirs(experiments_log_dir, exist_ok=True)
        
        # 4. 拼接错误日志路径（experiments/logs/error_log.txt）
        error_log_path = os.path.join(experiments_log_dir, "error_log.txt")
        
        # 5. 写入错误日志（路径替换为 error_log_path）
        with open(error_log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Error occurred while processing entity '{entity_id}': {str(e)}\n")
        print(f"Error occurred while processing property '{entity_id}': {str(e)}")
    return []

def get_prop_list(xlsx_path):
    """Get the list of properties from the Excel file"""
    df = pd.read_excel(xlsx_path)
    prop_list = df['ID'].tolist()
    return prop_list

def replace_special_characters1(s):
    """Replace special characters in the string"""
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    return s

def save_to_pl_file1(data, filename, backup_list):
    """Save the data to a Prolog file"""
    with open(filename, 'a', encoding='utf-8') as file:
        for item in data:
            if item['propertyLabel'] == 'length':
                item['propertyLabel'] = 'the_length_of'
            # Assuming the data needs to be in the form of Prolog facts
            fact = "{}('{}', '{}').\n".format(
                replace_special_characters1(item['propertyLabel'].lower()),
                replace_special_characters1(item['entityLabel'].lower()),
                replace_special_characters1(item['valueLabel'].lower())
            )
            file.write(fact)
            backup_list[item['propertyLabel']] = replace_special_characters1(item['propertyLabel'])
            backup_list[item['entityLabel']] = replace_special_characters1(item['entityLabel'])
            backup_list[item['valueLabel']] = replace_special_characters1(item['valueLabel'])

def process_prolog_file1(input_file, output_file):
    """Process the Prolog file to remove duplicates and sort the facts"""
    predicates = defaultdict(list)
    
    predicate_pattern = re.compile(r'^(\w+)\((.*)\)\.\s*$')

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):  
            continue
        match = predicate_pattern.match(line)
        if match:
            predicate_name = match.group(1)
            predicates[predicate_name].append(line)
        else:
            print(f"Skipping line: {line}")  

    with open(output_file, 'w', encoding='utf-8') as file:
        for predicate, facts in predicates.items():
            for fact in facts:
                file.write(fact + '\n')
