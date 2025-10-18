# OmniBench-RAG

## A Comprehensive RAG Evaluation Platform for Large Language Models (LLMs)
- This is a dedicated RAG (Retrieval-Augmented Generation) evaluation platform tailored for Large Language Models (LLMs), featuring a suite of core capabilities to support in-depth model assessment and research:
- It supports multi-domain analysis and evaluation, allowing for performance testing across diverse knowledge fields. The platform enables dynamic dataset generation, eliminating reliance on fixed, pre-existing datasets for more flexible assessment scenarios. It also provides multi-dimensional evaluation that covers two key metrics: accuracy (to measure answer correctness) and efficiency (to track resource consumption and response speed).
- A core objective of the platform is to facilitate exploration: it empowers users to independently upload custom RAG materials, or utilize materials with high domain relevance, to conduct comparative exploration—enabling in-depth analysis of how different RAG data sources impact model performance. Additionally, the platform reserves dedicated modules specifically designed for research reproducibility, ensuring that evaluation results can be easily replicated to validate findings.

---

##  Table of Contents

- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Modules](#key-modules)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

##  Core Features


#### Multi-Domain Evaluation
- **9 Specialized Domains**: Geography, History, Health, Technology, Mathematics, Nature, People, Society, Culture
- **Domain-Specific Knowledge Bases**: Custom-built knowledge graphs for each domain using Wikidata
- **Cross-Domain Benchmarking**: Compare model performance across different knowledge areas


#### Intelligent Dataset Generation
- **Dynamic Wikidata-Driven Dataset Creation**: Automatically extract entities & relationships from Wikidata, generate domain-specific inference rules, and build evaluation datasets based on the extracted knowledge.
- **Mitigate Dataset Leakage Bias**: Address unfair evaluation caused by potential dataset leakage in existing benchmarks.
- **Reproducibility-Friendly**: Retain datasets used in evaluations to ensure the reproducibility of research work.


#### Comprehensive Evaluation Metrics
- **Accuracy Assessment**: Use fine-tuned models to perform binary classification of answer correctness.
- **Efficiency Tracking**: Monitor memory usage, response time, and GPU utilization.
- **Multi-Type Question Support**: Evaluate models on inverse reasoning, negation reasoning, and composite reasoning questions.


#### Advanced RAG Capabilities
- **User-Centric Document Integration**: Support users to independently upload custom RAG materials and process PDF/documents with intelligent text chunking.
- **Strong RAG Material Comparison**: Enable users to utilize highly relevant *Strong RAG materials* (generated alongside dynamic dataset creation) for performance comparison.
- **Autonomous Exploration**: Facilitate users in conducting self-directed exploration and experimentation with RAG workflows.

---

##  System Architecture

```plaintext
OmniBench-RAG/
├── src/
│   ├── backend/                    # Flask-based API server
│   │   ├── main.py                # Main application entry point
│   │   ├── workflow.py            # Core evaluation workflow
│   │   ├── dynamic_dataset.py     # Dynamic dataset generation
│   │   └── work_models/           # Specialized processing modules
│   │       ├── data_preprocess.py    # PDF processing & text chunking
│   │       ├── embed_faiss.py        # Vector index management
│   │       ├── get_wiki_cat_id.py    # Wikidata category processing
│   │       ├── prolog_inference.py   # Prolog reasoning engine
│   │       ├── rule_generation.py    # Automated rule creation
│   │       ├── question_generation.py # Question template engine
│   │       ├── transitive_entity_extract.py  # Transitive relation extraction
│   │       ├── transitive_pl_build.py    # Prolog rule building
│   │       └── wiki_pl_build.py     # Wiki-to-Prolog conversion
│   ├── experiments/               # Experiment management
│   │   ├── logs/                  # Evaluation logs
│   │   ├── results/               # Evaluation results
│   │   └── progress/              # Progress tracking
│   └── frontend/                  # Web interface
│       ├── static/               # CSS, JavaScript, images
│       └── templates/            # HTML templates
├── data/                         # Data storage and processing
│   ├── RAG_material/            # RAG document management
│   │   ├── raw/                 # Uploaded PDF documents
│   │   ├── cleaned/             # Processed text chunks
│   │   ├── knowledge_base/      # FAISS vector indexes
│   │   ├── Strong/              # Enhanced knowledge files
│   │   └── Dynamic_Strong/      # Runtime knowledge bases
│   ├── dataset/                 # Generated evaluation datasets
│   │   ├── generated/           # Auto-generated Q&A pairs
│   │   ├── derived/             # Derived datasets
│   │   └── dynamic/             # Runtime datasets
│   └── wiki/                    # Wikidata processing results
├── docs/                        # Documentation
│   ├── module_details.md       # Detailed module descriptions
│   └── quick_start.md          # Installation and deployment guide
└── requirements.txt             # Python dependencies
```

---

##  Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package manager
- **Git**: Version control system
- **GPU Support (Optional)**: Requires manual configuration for NVIDIA GPUs
  - Install CUDA-enabled PyTorch: `pip install torch==2.0.1+cu118` (must match your CUDA version)
  - Install GPU version of FAISS: `pip install faiss-gpu` (replace `faiss-cpu`)
  - Ensure CUDA 11.0+ and a compatible NVIDIA GPU

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Garnett-Liang/Omnibench-RAG.git
cd omnibench-rag
```

### Step 2: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Download Language Models

```bash
# Download spaCy language model
python -m spacy download en_core_web_sm

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt')"
```

---

##  Quick Start

### 1. Launch the Application

Start the Flask development server:

```bash
cd src/backend
python main.py
```

The server will start at: **http://localhost:5000**

### 2. Access the Web Interface

Open your browser and navigate to:
```bash
**http://localhost:5000**
```

### 3. Basic Evaluation Workflow

1. **Select Domain**: Choose from 9 available domains (geography, history, health, etc.)
2. **Choose Model**: Select from supported LLMs (Qwen-1.8B, GPT-2, etc.)
3. **Configure RAG**: Set retrieval parameters (top-k, similarity threshold)
4. **Upload Documents** (Optional): Add custom PDF documents for domain-specific knowledge
5. **Start Evaluation**: Run comprehensive performance assessment
6. **View Results**: Monitor real-time progress and analyze results

### 4. Advanced Usage

For detailed installation and deployment instructions, see [docs/quick_start.md](docs/quickstart.md)


---

##  Key Modules

| Module | Description |
|--------|-------------|
| **main.py** | Flask API server with evaluation endpoints and file upload handling |
| **workflow.py** | Core evaluation orchestration and model management |
| **dynamic_dataset.py** | Automated dataset generation from knowledge bases |
| **data_preprocess.py** | PDF text extraction and intelligent chunking |
| **embed_faiss.py** | FAISS vector index creation and management |
| **prolog_inference.py** | Prolog-based logical reasoning engine |
| **rule_generation.py** | Automated inference rule generation |
| **question_generation.py** | Multi-type question template system |


---
For comprehensive information about each module and component, see [docs/module_details.md](docs/module_details.md)

---
##  API Documentation

### Core Endpoints

#### 1. POST /api/evaluate
Start basic evaluation process.

**Request Body**:
```json
{
  "rule_choice": "inverse|negation|composite|transitive",
  "domain_choice": "geography|history|health|technology|mathematics|nature|people|society|sports|culture",
  "model_choice": "qwen-1.8b|gpt2|...",
  "dataset_source": "existing|dynamic"
}
```

#### 2. GET /api/results/{log_file}
Retrieve evaluation results with detailed performance metrics.

#### 3. POST /api/evaluate_rag
Start RAG-enhanced evaluation with customizable retrieval parameters.

**Request Body**:
```json
{
  "rule": "inverse|negation|composite|transitive",
  "domain": "geography|history|...",
  "model_name": "qwen-1.8b|gpt2|...",
  "top_k": 3,
  "dataset_source": "existing|dynamic",
  "rag_material_source": "strong|flexible|dynamic_strong"
}
```

#### 4. POST /api/upload_rag_materials
Upload custom PDF documents for RAG processing.

**Request Parameters**:
- `domain`: Target domain (geography, history, health, etc.)
- `pdf_files`: List of PDF files to upload

#### 5. POST /api/generate_dataset
Generate dynamic datasets based on Wikidata knowledge extraction.

**Request Body**:
```json
{
  "rule_choice": "inverse|negation|composite|transitive",
  "domain_choice": "geography|history|health|..."
}
```


---

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Add tests for new features
- Use meaningful commit messages

---

##  License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

##  Contact

- **Maintainer**: Garnett Liang
- **Email**: liangjx@hust.edu.cn
- **GitHub Issues**: [Open an Issue](https://github.com/Garnett-Liang/Omnibench-RAG/issues)
- **Documentation**: [Project Wiki](https://github.com/Garnett-Liang/Omnibench-RAG/wiki)

---

##  Acknowledgments

- **Wikidata**: Knowledge base for entity extraction
- **FAISS**: Vector similarity search
- **Hugging Face**: Transformer models and datasets
- **Flask**: Web framework
- **Prolog**: Logic programming for inference

---
