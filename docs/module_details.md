# 📋 Module Details

This document provides detailed descriptions of each module and component in the OmniBench-RAG system.

## Core Backend Modules

### main.py - Application Entry Point

**Location**: `src/backend/main.py`

**Purpose**:
- Flask web server initialization and configuration
- API endpoint definitions for evaluation processes
- File upload handling for RAG documents
- Real-time progress tracking and logging
- Multi-threaded evaluation process management

**Key Features**:
- RESTful API endpoints for evaluations, datasets, and RAG materials
- Asynchronous processing with threading
- File upload validation and management
- Progress monitoring and result retrieval

### workflow.py - Evaluation Workflow Engine

**Location**: `src/backend/workflow.py`

**Purpose**:
- Core evaluation orchestration logic
- Model loading and management
- Performance metrics calculation
- RAG-enhanced evaluation workflows
- Progress tracking and result aggregation

**Key Classes**:
- `BinaryAnswerClassifier`: Fine-tuned model for answer correctness assessment
- Evaluation workflow managers for different question types

**Key Functions**:
- `run_evaluation()`: Main evaluation orchestration
- `run_rag_evaluation()`: RAG-specific evaluation logic
- `evaluate_model()`: Individual model performance assessment
- `evaluate_rag_model()`: RAG-enhanced model evaluation

### dynamic_dataset.py - Dynamic Dataset Generation

**Location**: `src/backend/dynamic_dataset.py`

**Purpose**:
- Generate evaluation datasets dynamically from knowledge bases
- Create diverse question types across domains
- Balance dataset difficulty and coverage
- Real-time dataset generation for custom domains

**Key Features**:
- **Multi-Type Questions**: Generate inverse, negation, composite, transitive questions
- **Domain Coverage**: Ensure comprehensive domain representation
- **Quality Control**: Filter and validate generated questions
- **Scalability**: Handle large-scale dataset generation

**Main Function**:
- `generate_dynamic_dataset()`: Complete pipeline for dynamic dataset creation

### evaluate.py - Evaluation Engine

**Location**: `src/backend/evaluate.py`

**Purpose**:
- Core evaluation logic for model performance assessment
- Dataset loading and processing
- Answer validation and scoring
- Integration with external evaluation metrics

**Key Features**:
- **Dataset Loading**: Support multiple dataset formats and sources
- **Answer Validation**: Binary classification of answer correctness
- **Metric Calculation**: Accuracy, precision, and performance metrics
- **Flexible Integration**: Easy integration with different models

**Main Function**:
- `loadset()`: Load and preprocess evaluation datasets

## Work Models - Specialized Processing Modules

### data_preprocess.py - Document Processing

**Location**: `src/backend/work_models/data_preprocess.py`

**Purpose**:
- PDF document text extraction using PyMuPDF
- Intelligent text chunking for RAG processing
- Domain-specific preprocessing pipelines
- Text cleaning and normalization

**Key Features**:
- **PDF Processing**: Extract text content from uploaded PDF files
- **Chunking Strategy**: Split documents into optimal chunks (100 words default)
- **Domain Organization**: Automatic folder structure management
- **Error Handling**: Robust error handling for corrupted files

**Functions**:
- `extract_text_from_pdf()`: Core PDF text extraction
- `rag_preprocess()`: Complete preprocessing pipeline

### description_collector.py - Metadata Collection

**Location**: `src/backend/work_models/description_collector.py`

**Purpose**:
- Collect and process entity descriptions
- Generate comprehensive entity metadata
- Support multiple description sources
- Maintain description quality and relevance

**Key Features**:
- **Multi-Source Integration**: Collect descriptions from various sources
- **Quality Enhancement**: Generate Strong and Dynamic_Strong materials
- **FAISS Integration**: Build vector indexes for collected descriptions
- **Preprocessing Pipeline**: Text chunking and cleaning

**Main Function**:
- `collect_descriptions_to_strong()`: Generate enhanced description materials



### embed_faiss.py - Vector Index Management

**Location**: `src/backend/work_models/embed_faiss.py`

**Purpose**:
- FAISS vector index creation and management
- Sentence embedding generation using SentenceTransformers
- Similarity search implementation
- GPU acceleration support

**Key Features**:
- **Multiple Models**: Support for various embedding models
- **Index Persistence**: Save/load FAISS indexes to disk
- **Batch Processing**: Efficient embedding generation
- **Domain-Specific Indexes**: Separate indexes per domain

**Functions**:
- `build_faiss_index()`: Create FAISS index from processed text
- `load_embedding_model()`: Load pre-trained embedding models
- `load_faiss_index()`: Load existing FAISS indexes

### get_wiki_cat_id.py - Wikidata Integration

**Location**: `src/backend/work_models/get_wiki_cat_id.py`

**Purpose**:
- Wikidata category and entity extraction
- SPARQL query generation and execution
- Entity relationship mapping
- Category hierarchy processing

**Key Features**:
- **SPARQL Integration**: Query Wikidata knowledge graph
- **Category Processing**: Extract domain-specific categories
- **Entity Extraction**: Identify relevant entities per domain
- **Link Generation**: Create entity-to-entity relationship maps

**Functions**:
- `get_category_pages()`: Retrieve category information
- `get_category_members()`: Extract category members
- `extract_entity_pages()`: Process entity pages
- `save_entity_links_to_file()`: Save entity links to files

### prolog_inference.py - Logic Reasoning Engine

**Location**: `src/backend/work_models/prolog_inference.py`

**Purpose**:
- Prolog-based logical inference implementation
- Complex reasoning question evaluation
- Rule-based answer validation
- Multi-hop reasoning support

**Key Features**:
- **Prolog Integration**: PySwip Prolog engine integration
- **Multiple Reasoning Types**: Inverse, negation, composite, transitive
- **Rule Management**: Dynamic rule loading and execution
- **Answer Validation**: Logic-based answer verification

**Key Functions**:
- `inverse_prolog_inference()`: Handle inverse reasoning questions
- `negation_prolog_inference()`: Process negation questions
- `composite_prolog_inference()`: Complex multi-condition reasoning
- `transitive_prolog_inference()`: Transitive relationship inference

### question_generation.py - Question Template Engine

**Location**: `src/backend/work_models/question_generation.py`

**Purpose**:
- Generate diverse question types from knowledge bases
- Template-based question creation
- Multi-lingual question support
- Question difficulty balancing

**Key Features**:
- **Template System**: Pre-defined question templates for each reasoning type
- **Entity Substitution**: Dynamic entity replacement in templates
- **Difficulty Control**: Generate questions of varying complexity
- **Answer Key Generation**: Automatic answer key creation

**Functions**:
- `inverse_template()`: Generate inverse reasoning questions
- `negation_template()`: Generate negation reasoning questions
- `composite_template()`: Generate composite reasoning questions

### rule_generation.py - Automated Rule Creation

**Location**: `src/backend/work_models/rule_generation.py`

**Purpose**:
- Automated generation of Prolog inference rules
- Domain-specific rule extraction from knowledge bases
- Rule optimization and validation
- Dynamic rule set management

**Key Features**:
- **Template-Based Generation**: Automated rule template creation
- **Property Mapping**: Extract properties from Wikidata
- **Rule Optimization**: Remove redundant and conflicting rules
- **Multi-Domain Support**: Domain-specific rule generation

**Key Classes**:
- `RuleGenerator`: Main rule generation engine

### transitive_entity_extract.py - Transitive Relation Extraction

**Location**: `src/backend/work_models/transitive_entity_extract.py`

**Purpose**:
- Extract transitive relationships from Wikidata
- Build transitive closure for inference
- Multi-hop relationship discovery
- Transitive rule generation

**Key Features**:
- **SPARQL Queries**: Complex queries for transitive relationships
- **Path Discovery**: Find indirect relationships between entities
- **Rule Generation**: Create Prolog rules for transitive inference
- **Performance Optimization**: Efficient processing of large knowledge graphs

**Functions**:
- `get_entity_info()`: Extract entity relationship information
- `get_predicate_labels()`: Process predicate labels

### transitive_pl_build.py - Transitive Prolog Rule Builder

**Location**: `src/backend/work_models/transitive_pl_build.py`

**Purpose**:
- Convert transitive relationships to Prolog rules
- Optimize rule sets for inference efficiency
- Handle rule conflicts and redundancies
- Generate transitive reasoning datasets

**Key Features**:
- **Rule Conversion**: Transform Wikidata relationships to Prolog format
- **Rule Optimization**: Remove circular and redundant rules
- **Dataset Generation**: Create question-answer pairs for transitive reasoning
- **Validation**: Ensure rule correctness and consistency

### wiki_pl_build.py - Wiki-to-Prolog Converter

**Location**: `src/backend/work_models/wiki_pl_build.py`

**Purpose**:
- Convert Wikidata entities to Prolog facts
- Generate comprehensive knowledge base in Prolog format
- Handle property normalization and standardization
- Create domain-specific Prolog files

**Key Features**:
- **Entity Processing**: Convert Wikidata entities to Prolog predicates
- **Property Normalization**: Standardize property names and formats
- **Batch Processing**: Efficient processing of large entity sets
- **Error Handling**: Robust error handling for malformed data

## Supporting Modules

### evaluate.py (Core) - Evaluation Engine

**Purpose**:
- Core evaluation logic for model performance assessment
- Dataset loading and processing
- Answer validation and scoring

### BinaryAnswerClassifier - Answer Validation

**Purpose**:
- Fine-tuned model for binary answer classification
- Determine if model answers are correct (yes/no)


