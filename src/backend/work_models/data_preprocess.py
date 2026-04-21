import os
import spacy
import fitz  # PyMuPDF

# 修复：不依赖外部模型，原生加载英文分词
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        print(f"⚠️ Failed to read {pdf_path}: {e}")
        return ""

# Chunk size in words
CHUNK_SIZE = 100


def rag_preprocess(domain):
    # -------------------------- 核心：基于原文件路径计算所有目录 --------------------------
    # 1. 获取当前函数（rag_preprocess）所在文件的绝对路径（如：src/backend/work_models/data_preprocess.py）
    current_script_path = os.path.abspath(__file__)
    # 2. 获取该文件所在的目录（如：src/backend/work_models）
    current_script_dir = os.path.dirname(current_script_path)
    # 3. 向上跳 2 级到项目根目录的父级？不——按你的结构，从 work_models 向上跳 2 级：
    # work_models → backend → src → 项目根目录（DEMO）？不对，重新算：
    # 正确层级：DEMO/src/backend/work_models/data_preprocess.py
    # 从 data_preprocess.py 向上跳 3 级到 DEMO（项目根目录）：
    project_root = os.path.abspath(os.path.join(current_script_dir, "..", "..", ".."))
    # （解释：current_script_dir 是 work_models → 跳 1 级到 backend → 跳 2 级到 src → 跳 3 级到 DEMO）

    # 4. 定义目标目录（基于项目根目录）
    # PDF源目录：DEMO/data/RAG_material/raw/{domain}（替代原 "RAG_raw/{domain}"）
    pdf_root = os.path.join(project_root, "data", "RAG_material", "raw")
    pdf_folder = os.path.join(pdf_root, domain.strip().lower())  # 最终PDF目录

    # 预处理输出目录：DEMO/data/RAG_material/cleaned（替代原 "RAG_cleaned"）
    output_dir = os.path.join(project_root, "data", "RAG_material", "cleaned")
    # -----------------------------------------------------------------------------------

    domain = domain.strip().lower()

    # 检查PDF目录是否存在
    if not os.path.isdir(pdf_folder):
        print(f"❌ Folder not found: {pdf_folder}")
        return

    chunk_list = []
    file_count = 0
    chunk_count = 0

    # 遍历PDF文件
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            raw_text = extract_text_from_pdf(pdf_path)

            if raw_text.strip():
                doc = nlp(raw_text)
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

                file_count += 1

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 输出文件路径：DEMO/data/RAG_material/cleaned/cleaned_{domain}.txt
    output_path = os.path.join(output_dir, f"cleaned_{domain}.txt")

    # 保存预处理结果
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chunk_list))

    # 打印日志（含完整路径，方便调试）
    print(f"\n✅ Preprocessing complete for domain: {domain}")
    print(f"📄 Processed papers: {file_count}")
    print(f"🧩 Total chunks generated: {chunk_count}")
    print(f"💾 Saved to: {output_path}")


if __name__ == "__main__":
    domain_input = input("Enter the domain to preprocess: ").strip()
    rag_preprocess(domain_input)