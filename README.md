# OmniBench-RAG

*A comprehensive RAG (Retrieval-Augmented Generation) evaluation platform for Large Language Models (LLMs), enabling multi-domain performance analysis, custom document integration, and detailed metric tracking.*

---

## Core Features

- **Multi-Domain Evaluation**  
  Assess LLM performance across domains (geography, history, health, technology, etc.).

- **RAG Enhancement**  
  Test models with retrieval-augmented generation using built-in or external documents.

- **Dual Metrics**  
  Track both accuracy (answer correctness) and efficiency (response time, resource usage).

- **PDF Workflow**  
  Upload PDFs to `RAG_raw/<domain>` and auto-process into RAG-ready chunks (`RAG_cleaned/`).

- **FAISS Indexing**  
  Automatically build or load vector indexes (stored in `knowledge_base/`) for fast retrieval.

- **Interactive UI**  
  Web interface for configuring evaluations, viewing logs, and analyzing results.

---

## Installation

### Prerequisites

- Python 3.8+
- `pip` (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Garnett-Liang/Omnibench-RAG.git
cd omnibench-rag
```

### Step 2: Install Dependencies

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```


---

## Quick Start

### 1. Launch the Application

```bash
python main.py
```

The server runs at: [http://localhost:5000](http://localhost:5000)

### 2. Access the UI

Open your browser and navigate to:

```
http://localhost:5000
```

### 3. Workflow Example (RAG as example)

1. Select a Domain (e.g., geography).  
2. Choose an LLM (e.g., Qwen-1.8B, GPT-2).  
3. Set Top-K (number of retrieved documents, 1–10).  
4. *(Optional)* Upload PDFs via "Upload RAG Materials" to customize RAG sources.  
5. Click **Start RAG Evaluation** to run assessments.  
6. View real-time logs and results in the UI.

---

## Project Structure

```plaintext
omnibench-rag/  
├── category/               # Domain classification configs (geography, history, etc.)  
├── data/                   # Folders for datasets of various domains generated dynamically 
├── knowledge_base/         # FAISS vector index files (per-domain)  
├── logs/                   # Evaluation logs (includes auto-generated error_log.txt)  
├── progress/               # Task progress trackers  
├── prolog_rules/           # Prolog inference rules (domain-specific logic)  
├── RAG_cleaned/            # Preprocessed RAG text chunks (per-domain .txt files)  
├── rag_progress/           # RAG task-specific progress records  
├── RAG_raw/                # Raw uploaded PDFs (organized into domain subfolders)  
├── rag_results/            # RAG evaluation results (JSON/CSV outputs)  
├── results/                # General LLM evaluation results  
├── static/                 # Frontend assets (CSS, JS, images)  
├── templates/              # Frontend templates (if using server-side rendering)  
├── utils/                  # Utility functions (text processing, IO helpers)  
├── wiki/                   # Wiki data modules (category IDs, entity extraction)  
├── data_preprocess.py      # PDF text extraction + chunking  
├── download.py             # Paper/data download utilities (domain-filtered)  
├── embed_faiss.py          # FAISS index building, loading, and retrieval  
├── error_log.txt           # System error log (auto-appended)  
├── evaluate.py             # Core LLM evaluation logic (accuracy, efficiency)  
├── get_wiki_cat_id.py      # Wiki category ID lookup tool  
├── main.py                 # Flask backend: API routes + workflow orchestration  
├── prolog_inference.py     # Prolog inference engine integration  
├── question_generation.py  # Domain-specific question generator for RAG  
├── Readme.md               # Project documentation (this file)  
├── rule_generation.py      # Automated domain rule generator  
├── transitive_entity_extract.py  # Transitive entity relation extraction  
├── transitive_pl_build.py  # Transitive Prolog rule builder  
├── wiki_pl_build.py        # Wiki data-to-Prolog rule converter  
└── requirements.txt        # Python dependency list (version-locked)  
```

---

## Key Modules

| File / Module           | Purpose                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| `main.py`               | Handles API endpoints, evaluation logic, and file uploads.              |
| `rag_evaluation.html`   | Frontend interface for configuration, visualization, and interaction.   |

---

##  License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.


---

## Contact

- Open a [GitHub Issue](https://github.com/Garnett-Liang/Omnibench-RAG/issues)
- Email: liangjx@hust.edu.cn 
# Omnibench-RAG
A comprehensive platform for evaluating large language models across domains—assess accuracy, efficiency, and RAG-enhanced performance using existing or uploaded data.
