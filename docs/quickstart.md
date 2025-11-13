#  Quick Start Guide

##  Operation Guide

### 1. Launch the Application

1. **Start the Server**:
   ```bash
   cd src/backend
   python main.py
   ```

2. **Access the Interface**:
   Open your browser and navigate to: `http://localhost:5000`

3. **Verify Startup**:
   - Server startup success shows: `Running on http://127.0.0.1:5000/`
   - Browser should display the OmniBench-RAG homepage

---

### 2. Generate Dynamic Dataset (Optional)

Dynamic dataset generation is based on Wikidata knowledge graph, including complete knowledge extraction pipeline:

#### Operation Steps:
1. **Select Reasoning Rule**:
   - `inverse` (Inverse Reasoning): e.g., "If A causes B, then what causes A?"
   - `negation` (Negation Reasoning): e.g., "If A is not a type of B, then..."
   - `composite` (Composite Reasoning): Multi-condition combination reasoning

2. **Select Domain**:
   - 9 specialized domains: geography, history, health, technology, mathematics, nature, people, society, culture

3. **API Call Example**:
   ```bash
   curl -X POST http://localhost:5000/api/generate_dataset \
     -H "Content-Type: application/json" \
     -d '{
       "rule_choice": "inverse",
       "domain_choice": "geography"
     }'
   ```

#### Generation Process:
- **Step 1**: Save domain category links to `data/dataset/category/`
- **Step 2**: Extract entity links and save to `data/dataset/category/selected/`
- **Step 3**: Get Wikidata entity IDs from Wikipedia URLs
- **Step 4**: Build transitive entity relationships and generate JSON files
- **Step 5**: Convert entity relationships to Prolog fact files
- **Step 6**: Perform Prolog inference based on reasoning rules to generate new facts
- **Step 7**: Generate question-answer pairs based on new facts

#### Notes:
- Generation process may take 10-30 minutes depending on entity count
- Detailed error information logged to log files on failure

---

### 3. Upload RAG Materials (Optional)

Support uploading custom PDF documents as RAG knowledge sources:

#### Operation Steps:
1. **Prepare PDF Documents**:
   - Support PDF format files
   - Recommended file size under 10MB each
   - Avoid special characters in filenames

2. **Select Target Domain**:
   - Choose corresponding domain based on document content (geography, history, etc.)

3. **API Call Example**:
   ```bash
   curl -X POST http://localhost:5000/api/upload_rag_materials \
     -F "domain=geography" \
     -F "pdf_files=@document1.pdf" \
     -F "pdf_files=@document2.pdf"
   ```

#### Automatic Processing:
- **Text Extraction**: Extract PDF text content using PyMuPDF
- **Intelligent Chunking**: Split documents into ~100-word text chunks
- **Vector Indexing**: Build FAISS vector index
- **Storage Management**: Save to `data/RAG_material/` directory structure

#### Supported Material Sources:
- **flexible**: User-uploaded custom materials
- **strong**: System pre-built high-quality materials
- **dynamic_strong**: Dynamically generated high-quality materials

---

### 4. Basic Evaluation

Standard large language model performance evaluation with multiple configurations:

#### Operation Steps:
1. **Select Reasoning Rule**:
   - Same rule selection as dynamic dataset generation

2. **Select Evaluation Domain**:
   - Choose from 9 specialized domains

3. **Select Evaluation Model**:
   - **Qwen-1.8B**: Tongyi Qianwen 1.8B parameter model
   - **GPT-2**: OpenAI's GPT-2 model
   - **facebook/opt-1.3b**: Facebook OPT 1.3B parameter model
   - **Custom API**: Use external API-based models (OpenAI, Anthropic, etc.)
     - Requires `api_endpoint`, `api_model_name`, and optionally `api_key`
     - Supports custom prompt templates via `api_prompt_template`
     - Configurable `api_max_tokens` (default: 1000)

4. **Select Dataset Source**:
   - **existing**: Use pre-built standard datasets (for reproduction)
   - **dynamic**: Use dynamically generated datasets

5. **API Call Example**:
   ```bash
   # Using built-in model
   curl -X POST http://localhost:5000/api/evaluate \
     -H "Content-Type: application/json" \
     -d '{
       "rule_choice": "inverse",
       "domain_choice": "geography",
       "model_choice": "qwen-1.8b",
       "dataset_source": "existing"
     }'
   
   # Using custom API
   curl -X POST http://localhost:5000/api/evaluate \
     -H "Content-Type: application/json" \
     -d '{
       "rule_choice": "inverse",
       "domain_choice": "geography",
       "model_choice": "api",
       "dataset_source": "existing",
       "api_endpoint": "https://api.openai.com/v1/chat/completions",
       "api_key": "your-api-key",
       "api_model_name": "gpt-4",
       "api_max_tokens": 1000,
       "api_prompt_template": "Answer with \"yes\" or \"no\": {question}"
     }'
   ```

#### Evaluation Metrics:
- **Accuracy**: Answer correctness assessment based on binary classification model
- **Response Time**: Average time for model to generate answers
- **Memory Usage**: Memory consumption during model inference
- **GPU Utilization**: GPU resource usage (if available)

#### Monitoring & Results:
- Real-time progress: `GET /api/progress/{log_file}`
- Detailed logs: `GET /api/logs/{log_file}`
- Evaluation results: `GET /api/results/{log_file}`

---

### 5. RAG-Enhanced Evaluation

Retrieval-augmented generation evaluation combining external knowledge sources to improve model performance:

#### Operation Steps:
1. **Select Reasoning Rule and Domain**:
   - Same rule and domain selection as basic evaluation

2. **Select Evaluation Model**:
   - **Qwen-1.8B**: Tongyi Qianwen 1.8B parameter model
   - **GPT-2**: OpenAI's GPT-2 model
   - **facebook/opt-1.3b**: Facebook OPT 1.3B parameter model
   - **Custom API**: Use external API-based models (OpenAI, Anthropic, etc.)
     - Requires `api_endpoint`, `api_model_name`, and optionally `api_key`
     - Supports custom prompt templates via `api_prompt_template`
     - For RAG evaluation, prompt template should include `{context}` and `{question}` placeholders
     - Configurable `api_max_tokens` (default: 1000)

3. **Configure Retrieval Parameters**:
   - **top_k**: Number of retrieved documents (1-10, recommended 3-5)

4. **Select Dataset Source**:
   - **existing**: Standard test datasets
   - **dynamic**: Dynamically generated datasets

5. **Select RAG Material Source**:
   - **strong**: High-quality pre-built materials (recommended)
   - **flexible**: User-uploaded custom materials
   - **dynamic_strong**: Dynamically generated high-quality materials

6. **API Call Example**:
   ```bash
   # Using built-in model
   curl -X POST http://localhost:5000/api/evaluate_rag \
     -H "Content-Type: application/json" \
     -d '{
       "rule": "inverse",
       "domain": "geography",
       "model_name": "qwen-1.8b",
       "top_k": 3,
       "dataset_source": "existing",
       "rag_material_source": "strong"
     }'
   
   # Using custom API
   curl -X POST http://localhost:5000/api/evaluate_rag \
     -H "Content-Type: application/json" \
     -d '{
       "rule": "inverse",
       "domain": "geography",
       "model_name": "api",
       "top_k": 3,
       "dataset_source": "existing",
       "rag_material_source": "strong",
       "api_endpoint": "https://api.openai.com/v1/chat/completions",
       "api_key": "your-api-key",
       "api_model_name": "gpt-4",
       "api_max_tokens": 1000,
       "api_prompt_template": "Context:\n{context}\n\nQuestion: {question}\nAnswer with \"yes\" or \"no\":"
     }'
   ```

#### RAG Evaluation Features:
- **Knowledge Retrieval**: Automatically retrieve relevant document chunks
- **Context Enhancement**: Integrate retrieved content into question context
- **Comparative Analysis**: Compare base model vs RAG-enhanced model performance
- **Transformation Metric**: Comprehensive assessment of RAG performance improvement

#### Advanced Monitoring:
- RAG progress: `GET /api/rag_progress/{log_file}`
- RAG results: `GET /api/rag_results/{log_file}`
- RAG materials: `GET /api/get_rag_materials/{domain}/{material_source}`

---

### 6. Results Visualization and Analysis

The platform provides comprehensive result visualization and statistical analysis capabilities:

#### Operation Steps:
1. **Access Statistics Dashboard**:
   - Navigate to the homepage and select a model from the dropdown menu
   - Click "Load Statistics" to view comprehensive performance metrics

2. **View Model Statistics**:
   - **API Call Example**:
     ```bash
     # Get statistics for a specific model
     curl -X GET http://localhost:5000/api/statistics/1
     ```
   - Returns comprehensive statistics including:
     - Basic accuracy and RAG accuracy for each domain
     - Improvement rates across domains
     - Transformation metrics
     - Dataset sizes

3. **Generate Radar Chart**:
   - **API Call Example**:
     ```bash
     # Get radar chart for a specific model
     curl -X GET http://localhost:5000/api/radar_chart/1
     ```
   - Returns base64-encoded PNG image of radar chart
   - Visualizes Basic vs RAG accuracy across all 9 domains
   - Includes performance summary statistics

#### Visualization Features:
- **Multi-Domain Comparison**: Radar charts showing performance across all 9 knowledge domains
- **Performance Metrics Table**: Detailed statistics table with accuracy, improvement, and transformation metrics
- **Average Performance Summary**: Aggregated statistics across all domains
- **Export Capabilities**: High-resolution charts for research publication

#### Supported Model IDs:
- **"1"**: Qwen/Qwen-1_8B
- **"2"**: gpt2-medium
- **"3"**: EleutherAI/gpt-neo-125M
- **"5"**: facebook/opt-1.3b

---

## Usage Recommendations

- **RAG Experiments**: Upload relevant domain PDF documents for better RAG results
- **Evaluation Time**: Evaluation processes may take considerable time (ranging from several minutes to over an hour depending on dataset size and model complexity). Please be patient and monitor progress through the log files and progress APIs.
- **Model Selection**: For optimal RAG performance, consider model-specific top-k parameter tuning (e.g., smaller top-k for GPT-2-medium, larger top-k for Qwen-1.8B)

