import os
import json
from pathlib import Path
os.environ['SWI_HOME_DIR'] = 'C:\\ZTCMyfile\\CODE\\swipl'  # 
from pyswip import Prolog, Atom
from tqdm import tqdm
import re
import requests
import argparse

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
        print(f"fail: {abs_path}")
        print(f"error: {str(e)}")
        
        if not os.path.exists(abs_path):
            print(f"File not exist: {abs_path}")
        raise

def get_wikipedia_summary(wikidata_id, language='en'):
    
    wikidata_url = f'https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json'
    response = requests.get(wikidata_url)
    if response.status_code != 200:
        print(f"Cannot get entity data for Wikidata ID {wikidata_id}.")
        return None

    data = response.json()
    entities = data.get('entities', {})
    entity = entities.get(wikidata_id, {})
    sitelinks = entity.get('sitelinks', {})
    wikipedia_key = f'{language}wiki'
    sitelink = sitelinks.get(wikipedia_key, {})
    page_title = sitelink.get('title', None)

    if not page_title:
        print(f"Cannot find Wikipedia page for Wikidata ID {wikidata_id} in language '{language}'.")
        return None

    
    wikipedia_api_url = f'https://{language}.wikipedia.org/api/rest_v1/page/summary/{page_title}'
    response = requests.get(wikipedia_api_url)
    if response.status_code != 200:
        print(f"Cannot get summary for Wikipedia page '{page_title}'.")
        return None

    summary_data = response.json()
    summary = summary_data.get('extract', None)
    print(f"Successfully get summary for Wikipedia page '{page_title}'.")
    return summary

def replace_special_characters2(s):
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    return s

def is_q_followed_by_digits(s): 
    pattern = r'^q\d+$'  
    return (bool(re.match(pattern, s)) and s != '_')

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
        # print(len(useful_entitya))
        # print(useful_entityb)
        key = key_list[idx]+'(Entity_A,'
        if (key not in query_content) or ('\n'+key_list[idx]+'(' not in fact_content):
            continue
        for entitya in useful_entitya:
            cnt = 0
            if entitya.strip("'") not in modified_name_list or is_q_followed_by_digits(entitya):
                continue
            subject = original_name_list[modified_name_list.index(entitya.strip("'"))]
            for entity in log_entities:
                if entity['entityLabel'] == subject :
                    subject_id = entity['entity'].split('/')[-1]
                    break
                elif entity['valueLabel'] == subject:
                    subject_id = entity['value'].split('/')[-1]
                    break
            subject_description = get_wikipedia_summary(subject_id)
            if subject_description == None:
                continue
            for entityb in useful_entityb:
                new_problem = problem.replace('Entity_A', entitya).replace('Entity_B', entityb)
                # print(new_problem)
                try:
                    result = list(prolog.query(new_problem))
                    if len(result) == 0:
                        continue
                    if result:
                        object_ = original_name_list[modified_name_list.index(entityb.strip("'").lower())]
                        for entity in log_entities:
                            if entity['entityLabel'] == object_:
                                object_id = entity['entity'].split('/')[-1]
                            elif entity['valueLabel'] == object_:
                                object_id = entity['value'].split('/')[-1]
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
                        # print(new_fact)
                        new_fact_list.append(new_fact)
                        cnt += 1
                        all_cnt += 1
                        # print(all_cnt)
                except Exception as e:
                    with open(f'error_log.txt', 'a') as log_file:
                        log_file.write(f"Error occurred while processing: {str(e)}\n")
                        print(f"Error occurred while processing property: {str(e)}")
                # print(cnt)
                if cnt > 3:
                    break
        if all_cnt > 300:
            break
    return new_fact_list


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
                    if soln['Entity_A'].strip("'") not in modified_name_list or soln['Entity_B'] not in modified_name_list:
                        continue
                    if is_q_followed_by_digits(soln['Entity_A']) or is_q_followed_by_digits(soln['Entity_B']):
                        continue
                    subject = original_name_list[modified_name_list.index(soln['Entity_A'].strip("'"))]
                    for entity in log_entities:
                        if entity['entityLabel'] == subject :
                            subject_id = entity['entity'].split('/')[-1]
                        elif entity['valueLabel'] == subject:
                            subject_id = entity['value'].split('/')[-1]
                    subject_description = get_wikipedia_summary(subject_id)
                    if subject_description == None:
                        break
                    object_ = original_name_list[modified_name_list.index(soln['Entity_B'].strip("'").lower())]
                    for entity in log_entities:
                        if entity['entityLabel'] == object_:
                            object_id = entity['entity'].split('/')[-1]
                        elif entity['valueLabel'] == object_:
                            object_id = entity['value'].split('/')[-1]
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
                    with open(f'error_log.txt', 'a') as log_file:
                        log_file.write(f"Error occurred while processing: {str(e)}\n")
                        print(f"Error occurred while processing property: {str(e)}")
            if cnt > 10:
                break
    return new_fact_list


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
        # if cnt > 50:
        #     continue
        result = list(prolog.query(problem))
        for soln in result:
            if soln:
                try:
                    subject = original_name_list[modified_name_list.index(soln['Entity_A'].strip("'"))]
                    for entity in log_entities:
                        if entity['entityLabel'] == subject :
                            subject_id = entity['entity'].split('/')[-1]
                        elif entity['valueLabel'] == subject:
                            subject_id = entity['value'].split('/')[-1]
                    subject_description = get_wikipedia_summary(subject_id)
                    if subject_description == None:
                        continue
                    object_ = original_name_list[modified_name_list.index(soln['Entity_B'].strip("'").lower())]
                    for entity in log_entities:
                        if entity['entityLabel'] == object_:
                            object_id = entity['entity'].split('/')[-1]
                        elif entity['valueLabel'] == object_:
                            object_id = entity['value'].split('/')[-1]
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
                    # print(new_fact)
                    new_fact_list.append(new_fact)
                    cnt += 1
                    all_cnt += 1
                    # print(all_cnt)
                except Exception as e:
                    with open(f'error_log.txt', 'a') as log_file:
                        log_file.write(f"Error occurred while processing: {str(e)}\n")
                        print(f"Error occurred while processing property: {str(e)}")
        if all_cnt > 200:
            break
    return new_fact_list

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
        for soln in list(prolog.query(problem)):  
            if soln:  
                soln_set.add((soln['Entity_A'], problem.split('(')[0], soln['Entity_B']))
    print(f"Total {len(soln_set)} facts found.")
    for soln_a, predicate, solnb in soln_set:
        if soln:
            try:
                if soln_a.strip("'") not in modified_name_list or solnb not in modified_name_list:
                    continue
                if is_q_followed_by_digits(soln_a) or is_q_followed_by_digits(solnb):
                    continue
                subject = original_name_list[modified_name_list.index(soln_a.strip("'"))]
                subject_id = None
                for entity in log_entities:
                    if entity['subjectValue'] == subject :
                        subject_id = entity['subject']
                if subject_id is None:
                    continue
                subject_description = get_wikipedia_summary(subject_id)
                if subject_description is None:
                    break
                object_ = original_name_list[modified_name_list.index(solnb.strip("'").lower())]
                object_id = None
                for entity in log_entities:
                    if entity['anotherObjectValue'] == object_:
                        object_id = entity['anotherObject'].split('/')[-1]
                if object_id is None:
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
                # print(new_fact)
                new_fact_list.append(new_fact)
                cnt += 1
                per_cnt += 1
                print(f"Processed {cnt} facts.")
            except Exception as e:
                with open(f'error_log.txt', 'a') as log_file:
                    log_file.write(f"Error occurred while processing: {str(e)}\n")
                    print(f"Error occurred while processing property: {str(e)}")
        #     if per_cnt > 5:
        #         break
        if cnt > 200:
            break
    return new_fact_list



