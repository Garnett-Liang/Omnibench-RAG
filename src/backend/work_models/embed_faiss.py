from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def build_faiss_index(domain: str):
    # 获取当前函数所在文件的路径，用于计算绝对路径
    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    
    # 向上跳3级到项目根目录，再进入data/RAG_material
    rag_material_root = os.path.abspath(os.path.join(
        current_script_dir, "../../../data/RAG_material"
    ))
    
    domain = domain.strip().lower()
    
    # 输入文件路径：data/RAG_material/cleaned/cleaned_{domain}.txt
    input_path = os.path.join(rag_material_root, "cleaned", f"cleaned_{domain}.txt")

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

    # 索引保存路径：data/RAG_material/knowledge_base/{domain}_index.faiss
    knowledge_base_dir = os.path.join(rag_material_root, "knowledge_base")
    os.makedirs(knowledge_base_dir, exist_ok=True)
    index_path = os.path.join(knowledge_base_dir, f"{domain}_index.faiss")
    faiss.write_index(index, index_path)

    print(f"Index created for domain: {domain}")
    print(f"Saved to: {index_path}")
    print(f"Number of vectors in FAISS index: {index.ntotal}")
    
    return model, index


def load_embedding_model():
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def load_faiss_index(domain, index_file_path=None):
    """
    加载FAISS索引，支持两种模式：
    1. 当传入index_file_path时，直接从指定路径加载
    2. 当未传入index_file_path时，使用默认路径
    
    参数:
        domain: 领域名称（如geography）
        index_file_path: 可选参数，FAISS索引文件的绝对路径
    返回:
        index: 加载后的FAISS索引对象
    """
    try:
        import faiss
        
        # 模式1：使用指定的索引文件路径
        if index_file_path and os.path.exists(index_file_path):
            index = faiss.read_index(index_file_path)
            print(f"Successfully loaded FAISS index from: {index_file_path}")
            return index
        
        # 模式2：使用默认路径（保持原逻辑兼容）
        default_path = os.path.join("knowledge_base", f"{domain}_index.faiss")
        if os.path.exists(default_path):
            index = faiss.read_index(default_path)
            print(f"Successfully loaded FAISS index from default path: {default_path}")
            return index
        
        # 若两种模式都加载失败
        print(f"Error: FAISS index not found for domain '{domain}'")
        return None
        
    except ImportError:
        print("Error: faiss library not installed. Please install faiss with 'pip install faiss-cpu' or 'faiss-gpu'")
        return None
    except Exception as e:
        print(f"Error loading FAISS index: {str(e)}")
        return None