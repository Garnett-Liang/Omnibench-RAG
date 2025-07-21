import os
import spacy
import fitz  # PyMuPDF

# Load spaCy English tokenizer
nlp = spacy.load("en_core_web_sm")

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
    domain = domain.strip().lower()
    pdf_folder = os.path.join("RAG_raw", domain)

    if not os.path.isdir(pdf_folder):
        print(f" Folder not found: {pdf_folder}")
        return

    chunk_list = []
    file_count = 0
    chunk_count = 0

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            raw_text = extract_text_from_pdf(pdf_path)

            if raw_text.strip():
                doc = nlp(raw_text)
                words = [token.text for token in doc if not token.is_stop and not token.is_space and token.is_alpha]

                chunk = []
                for word in words:
                    chunk.append(word)
                    if len(chunk) >= CHUNK_SIZE:
                        chunk_list.append(" ".join(chunk))
                        chunk = []
                        chunk_count += 1
                if chunk:
                    chunk_list.append(" ".join(chunk))
                    chunk_count += 1

                file_count += 1

    # Output path
    output_dir = "RAG_cleaned"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cleaned_{domain}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chunk_list))

    print(f"\nPreprocessing complete for domain: {domain}")
    print(f"Processed papers: {file_count}")
    print(f"Total chunks generated: {chunk_count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    domain_input = input("Enter the domain to preprocess: ").strip()
    rag_preprocess(domain_input)
