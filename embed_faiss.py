from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def build_faiss_index(domain: str):
    
    domain = domain.strip().lower()
    input_path = f"RAG_cleaned/cleaned_{domain}.txt"

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Loading cleaned abstracts from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        cleaned_abstracts = [line.strip() for line in f if line.strip()]

    print("Encoding abstracts into embeddings...")
    embeddings = model.encode(cleaned_abstracts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("knowledge_base", exist_ok=True)
    index_path = f"knowledge_base/{domain}_index.faiss"
    faiss.write_index(index, index_path)

    print(f"Index created for domain: {domain}")
    print(f"Saved to: {index_path}")
    print(f"Number of vectors in FAISS index: {index.ntotal}")
    
    return model, index


def load_embedding_model():
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def load_faiss_index(domain: str):
    domain = domain.strip().lower()
    index_path = f"knowledge_base/{domain}_index.faiss"
    
    if not os.path.exists(index_path):
        print(f"Index file not found: {index_path}")
        return None
        
    print(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)
    print(f"Number of vectors in FAISS index: {index.ntotal}")
    return index


if __name__ == "__main__":
    domain_input = input("Enter the domain to index : ").strip()
    faiss(domain_input)