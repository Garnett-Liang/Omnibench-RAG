import os
import json
import spacy
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load spaCy English tokenizer
nlp = spacy.load("en_core_web_sm")

# Chunk size in words (same as rag_preprocess)
CHUNK_SIZE = 100

def collect_descriptions_to_strong(domain, dataset_source='reproducible'):
    """
    Collect descriptions from datasets and generate Strong materials

    Args:
        domain: Domain name (e.g., 'geography', 'history')
        dataset_source: 'reproducible' or 'dynamic'
    """
    # Get absolute path of the current script
    current_script_path = Path(__file__).resolve()
    # Get project root directory
    project_root = current_script_path.parent.parent.parent.parent

    print(f"Processing domain: {domain} with dataset source: {dataset_source}")

    # Define domains and reasoning types
    domains = [domain]  # Process only the specified domain
    reasoning_types = ["inverse", "negation", "composite"]

    # Create Strong directories (based on project root)
    strong_dir = project_root / "data/RAG_material/Strong"
    strong_cleaned_dir = project_root / "data/RAG_material/Strong_cleaned"
    strong_base_dir = project_root / "data/RAG_material/Strong_base"

    # Create directories for dynamic strong materials if needed
    if dataset_source == 'dynamic':
        dynamic_strong_dir = project_root / "data/RAG_material/Dynamic_Strong"
        dynamic_strong_cleaned_dir = project_root / "data/RAG_material/Dynamic_Strong_cleaned"
        dynamic_strong_base_dir = project_root / "data/RAG_material/Dynamic_Strong_base"

        dynamic_strong_dir.mkdir(exist_ok=True)
        dynamic_strong_cleaned_dir.mkdir(exist_ok=True)
        dynamic_strong_base_dir.mkdir(exist_ok=True)

    strong_dir.mkdir(exist_ok=True)
    strong_cleaned_dir.mkdir(exist_ok=True)
    strong_base_dir.mkdir(exist_ok=True)

    for domain in domains:
        print(f"\nProcessing domain: {domain}")
        all_descriptions = []

        for reasoning_type in reasoning_types:
            # Construct file path (based on project root)
            if dataset_source == 'reproducible':
                generated_path = project_root / f"data/dataset/generated/{reasoning_type}/{domain}_qa.json"
            else:  # dataset_source == 'dynamic'
                generated_path = project_root / f"data/dataset/dynamic/{reasoning_type}/{domain}_qa.json"

            if not generated_path.exists():
                print(f"Warning: {generated_path} not found, skipping")
                continue

            try:
                # Read JSON file
                with open(generated_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract all description fields
                for item in data:
                    if 'description' in item and item['description']:
                        # Add description if it's not already in the list to avoid duplicates
                        if item['description'] not in all_descriptions:
                            all_descriptions.append(item['description'])

                print(f"Processed {len(data)} items from {domain}_{reasoning_type}")

            except Exception as e:
                print(f"Error processing {generated_path}: {e}")
                continue

        if all_descriptions:
            # Step 1: Save raw descriptions to appropriate directory
            if dataset_source == 'reproducible':
                strong_file = strong_dir / f"Strong_{domain}.txt"
            else:  # dataset_source == 'dynamic'
                strong_file = dynamic_strong_dir / f"Dynamic_Strong_{domain}.txt"

            with open(strong_file, 'w', encoding='utf-8') as f:
                f.write(f"Descriptions for {domain} domain ({dataset_source})\n")
                f.write("=" * 50 + "\n")
                for description in all_descriptions:
                    f.write(description + "\n\n")  # Add extra newline between descriptions

            print(f"Saved {len(all_descriptions)} descriptions to {strong_file}")

            # Step 2: Process descriptions like RAG preprocessing
            print(f"Preprocessing {len(all_descriptions)} descriptions for {domain}...")
            chunk_list = []
            chunk_count = 0

            for description in all_descriptions:
                if description.strip():
                    doc = nlp(description)
                    # 过滤停用词、空格，只保留字母
                    words = [token.text for token in doc if not token.is_stop and not token.is_space and token.is_alpha]

                    # 按CHUNK_SIZE切分文本
                    chunk = []
                    for word in words:
                        chunk.append(word)
                        if len(chunk) >= CHUNK_SIZE:
                            chunk_list.append(" ".join(chunk))
                            chunk = []
                            chunk_count += 1
                    # 处理最后剩余的不足CHUNK_SIZE的片段
                    if chunk:
                        chunk_list.append(" ".join(chunk))
                        chunk_count += 1

            # Step 3: Save processed chunks to appropriate directory
            if dataset_source == 'reproducible':
                cleaned_file = strong_cleaned_dir / f"Strong_{domain}.txt"
            else:  # dataset_source == 'dynamic'
                cleaned_file = dynamic_strong_cleaned_dir / f"Dynamic_Strong_{domain}.txt"

            with open(cleaned_file, "w", encoding="utf-8") as f:
                f.write("\n".join(chunk_list))

            print(f"Saved {len(chunk_list)} processed chunks to {cleaned_file}")

            # Step 4: Build FAISS index like embed_faiss.py
            print(f"Building FAISS index for {domain}...")

            # Load embedding model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Encode chunks into embeddings
            embeddings = model.encode(chunk_list)

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

            # Save FAISS index to appropriate directory
            if dataset_source == 'reproducible':
                index_file = strong_base_dir / f"{domain}_index.faiss"
            else:  # dataset_source == 'dynamic'
                index_file = dynamic_strong_base_dir / f"{domain}_index.faiss"

            faiss.write_index(index, str(index_file))

            print(f"FAISS index created for domain: {domain}")
            print(f"Saved to: {index_file}")
            print(f"Number of vectors in FAISS index: {index.ntotal}")

        else:
            print(f"No descriptions found for {domain}")

def collect_descriptions_to_strong_all():
    """Legacy function for backward compatibility"""
    collect_descriptions_to_strong()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        domain = sys.argv[1]
        dataset_source = sys.argv[2] if len(sys.argv) > 2 else 'reproducible'
        collect_descriptions_to_strong(domain, dataset_source)
    else:
        collect_descriptions_to_strong()