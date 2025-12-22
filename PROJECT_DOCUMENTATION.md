# DocuMind AI - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Features](#core-features)
3. [Architecture](#architecture)
4. [Technical Stack](#technical-stack)
5. [System Components](#system-components)
6. [Workflow & Process Flow](#workflow--process-flow)
7. [Agent System](#agent-system)
8. [Installation & Setup](#installation--setup)
9. [Configuration](#configuration)
10. [Usage Guide](#usage-guide)
11. [File Structure](#file-structure)
12. [API Reference](#api-reference)
13. [Monitoring & Logging](#monitoring--logging)
14. [Performance & Optimization](#performance--optimization)
15. [Troubleshooting](#troubleshooting)

---

## Project Overview

**DocuMind AI** is an intelligent document analysis platform that leverages local Large Language Models (LLMs) via Ollama to provide conversational question-answering over PDF documents. The system uses Retrieval-Augmented Generation (RAG) with vector embeddings to deliver accurate, citation-backed responses.

### Key Highlights
- **Framework**: Built on LangChain and LangGraph for orchestration
- **LLM Backend**: Uses Ollama for local, privacy-preserving inference
- **Vector Database**: ChromaDB for document embeddings
- **UI**: Interactive Streamlit chat interface
- **Architecture**: Multi-agent system with intelligent routing

---

## Core Features

### 1. **Document Processing**
- Multi-PDF upload support
- Automatic text extraction with PyMuPDF
- Intelligent text chunking with overlap for context preservation
- Metadata tracking (page numbers, sources)
- Persistent vector storage

### 2. **Intelligent Query Routing**
- LLM-based query classification
- Automatic detection of conversational vs. document queries
- Keyword-based fallback for robustness
- Prevents unnecessary document retrieval

### 3. **Multi-Agent Workflow**
- Modular agent architecture
- Conditional routing based on query type
- Progressive audit logging at each step
- Error handling and recovery

### 4. **Retrieval-Augmented Generation**
- Semantic similarity search (top-5 documents)
- Context-aware response generation
- Source citations with page numbers
- Support for structured data extraction

### 5. **Chat Interface**
- Conversational UI with message history
- Real-time processing indicators
- Source document preview
- Audit trail inspection
- System reset capability

### 6. **Audit & Monitoring**
- Step-by-step execution logging
- Query trail persistence (JSONL format)
- Performance tracking
- Error tracking and debugging

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                       │
│  (Chat Interface, Document Upload, Audit Viewer)            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Orchestrator Layer                          │
│              (LangGraph State Machine)                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐│
│  │  Router   │→ │ Retrieval │→ │Extraction │→ │ Analysis ││
│  │  Agent    │  │   Agent   │  │   Agent   │  │  Agent   ││
│  └─────┬─────┘  └───────────┘  └───────────┘  └──────────┘│
│        │                                                     │
│        ▼                                                     │
│  ┌───────────┐                                              │
│  │ Rejection │                                              │
│  │ Handler   │                                              │
│  └───────────┘                                              │
└─────────────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│               Infrastructure Layer                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  ChromaDB   │  │    Ollama    │  │ Audit Logger │      │
│  │  (Vectors)  │  │  (LLM API)   │  │   (JSONL)    │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Design Patterns
- **State Machine**: LangGraph for workflow orchestration
- **Agent Pattern**: Modular, single-responsibility agents
- **Repository Pattern**: Vector store abstraction
- **Strategy Pattern**: Conditional routing logic
- **Observer Pattern**: Progressive audit logging

---

## Technical Stack

### Core Technologies
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **LLM Framework** | LangChain | Latest | Orchestration & chains |
| **Workflow Engine** | LangGraph | Latest | State machine graphs |
| **LLM Backend** | Ollama | Latest | Local LLM inference |
| **Vector DB** | ChromaDB | Latest | Embedding storage |
| **Embeddings** | SentenceTransformers | Latest | all-MiniLM-L6-v2 |
| **PDF Processing** | PyMuPDF | Latest | Text extraction |
| **UI Framework** | Streamlit | Latest | Web interface |
| **Configuration** | python-dotenv | Latest | Environment management |

### Python Dependencies
```
streamlit
langchain
langchain-community
langchain-core
langchain-text-splitters
langgraph
chromadb
sentence-transformers
pymupdf
requests
pydantic
python-dotenv
```

---

## System Components

### 1. **Core Module** (`core/`)

#### `config.py`
- Centralized configuration management
- Environment variable loading
- Default values for all settings
```python
class Config:
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_DB_DIR", "data/chroma")
    MODEL_NAME = os.getenv("LLM_MODEL", "qwen2.5:3b")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

#### `state.py`
- Defines `AgentState` TypedDict
- Manages workflow state across agents
- Includes: query, route, docs, extracted data, response, audit log

#### `orchestrator.py`
- Builds LangGraph state machine
- Manages agent initialization
- Handles conditional routing
- Executes workflow and returns results

### 2. **Agent Module** (`agents/`)

#### `router.py`
- **RouterAgent**: LLM-based query classification
- **reject_query()**: Handles out-of-scope queries
- Fallback to keyword-based classification
- Routes: "document" or "conversational"

#### `retrieval.py`
- **RetrievalAgent**: Semantic search over vector store
- Top-k similarity search (k=5)
- Metadata preservation
- Progressive logging

#### `extraction.py`
- **ExtractionAgent**: Structured data extraction
- Conditional execution (only if query requests extraction)
- Table/entity extraction support
- LLM-based extraction with timeout

#### `analysis.py`
- **AnalysisAgent**: Final response generation
- Context aggregation from retrieved docs
- Citation generation with page numbers
- Answer synthesis with LLM

### 3. **Document Processing** (`document_processing/`)

#### `loader.py`
- PDF text extraction with PyMuPDF
- Page-level metadata tracking
- Error handling for corrupted files

#### `chunking.py`
- RecursiveCharacterTextSplitter
- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Document ID assignment

### 4. **Vector Store** (`vectorstore/`)

#### `chroma.py`
- ChromaDB wrapper
- SentenceTransformer embeddings
- Persistent storage to disk
- Similarity search interface

### 5. **Audit System** (`audit/`)

#### `logger.py`
- **AuditLogger**: Query and step logging
- JSONL format for logs
- Progressive step logging (real-time)
- Full query trail logging (post-execution)
- Two log files:
  - `audit_trail.jsonl`: Full query results
  - `audit_trail_steps.jsonl`: Individual agent steps

---

## Workflow & Process Flow

### Document Ingestion Flow
```
User uploads PDF(s)
    ↓
PDFLoader extracts text by page
    ↓
DocumentChunker splits into chunks (500 chars, 50 overlap)
    ↓
VectorStoreManager creates embeddings (all-MiniLM-L6-v2)
    ↓
ChromaDB persists vectors to disk
```

### Query Processing Flow

#### Path 1: Document Query
```
User Query: "What is the revenue?"
    ↓
RouterAgent (LLM classification)
    ↓
route = "document"
    ↓
RetrievalAgent (similarity search, k=5)
    ↓
ExtractionAgent (conditional, skips if not extracting)
    ↓
AnalysisAgent (aggregates context + generates answer)
    ↓
Response with citations returned
```

#### Path 2: Conversational Query
```
User Query: "Hello"
    ↓
RouterAgent (LLM classification)
    ↓
route = "conversational"
    ↓
reject_query() (static response)
    ↓
"I can only answer questions about documents..." returned
    ↓
END (no retrieval)
```

### State Machine Diagram
```
        START
          │
          ▼
     ┌─────────┐
     │ Router  │
     └────┬────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌─────────┐
│Rejection│  │Retrieval│
└────────┘  └────┬────┘
    │            │
    │            ▼
    │       ┌──────────┐
    │       │Extraction│
    │       └────┬─────┘
    │            │
    │            ▼
    │       ┌─────────┐
    │       │Analysis │
    │       └────┬────┘
    │            │
    └─────┬──────┘
          │
          ▼
         END
```

---

## Agent System

### Agent Responsibilities

| Agent | Input | Output | Purpose |
|-------|-------|--------|---------|
| **RouterAgent** | Query text | Route ("document"/"conversational") | Query classification |
| **RetrievalAgent** | Query | Top-5 documents | Semantic search |
| **ExtractionAgent** | Query + Docs | Structured data | Extract tables/entities |
| **AnalysisAgent** | Query + Docs + Extracted | Final answer | Response synthesis |

### Agent Communication
- Agents communicate via shared `AgentState`
- Immutable state updates (TypedDict)
- Each agent reads from state, writes new fields
- Audit log aggregates across agents

### Error Handling
- Try-catch in each agent's `invoke()` method
- Progressive logging captures errors
- Graceful degradation (fallbacks)
- User-facing error messages

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Ollama installed and running
- GPU recommended (but not required)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Documind
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install & Start Ollama
```bash
# Install Ollama (platform-specific)
# Visit: https://ollama.ai

# Pull model
ollama pull qwen2.5:3b

# Verify running
ollama list
```

### Step 4: Configure Environment
```bash
cp .env.example .env

# Edit .env with your settings:
# OLLAMA_BASE_URL=http://localhost:11434
# LLM_MODEL=qwen2.5:3b
# CHROMA_DB_DIR=data/chroma
```

### Step 5: Run Application
```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `LLM_MODEL` | `qwen2.5:3b` | Model name for inference |
| `CHROMA_DB_DIR` | `data/chroma` | Vector DB storage path |

### Model Selection
Supported Ollama models:
- `qwen2.5:3b` - Fast, recommended for CPU
- `llama3` - Balanced performance
- `mistral` - Good accuracy
- `phi3` - Lightweight

### Performance Tuning
```python
# In agents/analysis.py
ChatOllama(
    model=model_name,
    temperature=0.2,  # Lower = more deterministic
    timeout=30        # Adjust based on hardware
)
```

---

## Usage Guide

### Uploading Documents
1. Click "Browse files" in sidebar
2. Select one or more PDFs
3. Click "Process Documents"
4. Wait for embedding completion

### Asking Questions
**Document Queries:**
- "What is the revenue in Q4?"
- "Summarize the executive summary"
- "Extract sales figures from page 5"

**System Response:**
- Answer with citations
- Source page numbers
- Expandable audit trail

**Out-of-Scope Queries:**
- "Hello" → Rejected
- "How are you?" → Rejected
- "What can you do?" → Rejected

### Viewing Audit Logs
1. Switch to "📊 Audit Logs" tab
2. Click "Refresh Logs"
3. View execution trace in table format

### Resetting System
- Click "🔄 Reset System" to:
  - Clear chat history
  - Reload components
  - Apply code changes

---

## File Structure

```
Documind/
├── agents/
│   ├── __init__.py
│   ├── router.py          # Query classification & rejection
│   ├── retrieval.py       # Vector search
│   ├── extraction.py      # Structured extraction
│   └── analysis.py        # Response generation
├── audit/
│   ├── __init__.py
│   └── logger.py          # Audit trail logging
├── core/
│   ├── __init__.py
│   ├── config.py          # Configuration mgmt
│   ├── state.py           # State definitions
│   └── orchestrator.py    # Workflow orchestrator
├── document_processing/
│   ├── __init__.py
│   ├── loader.py          # PDF loading
│   └── chunking.py        # Text chunking
├── vectorstore/
│   ├── __init__.py
│   └── chroma.py          # ChromaDB wrapper
├── data/
│   ├── chroma/            # Vector DB (persistent)
│   ├── documents/         # Uploaded PDFs (temp)
│   └── logs/              # Audit logs (JSONL)
├── utils/
│   └── __init__.py
├── app.py                 # Streamlit UI
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── .gitignore
└── README.md
```

---

## API Reference

### Orchestrator API

```python
from core.orchestrator import Orchestrator

orchestrator = Orchestrator()
result = orchestrator.run(
    query="What is the revenue?",
    query_id="uuid-string",
    audit_logger=audit_logger_instance
)

# Returns AgentState dict:
{
    "query": str,
    "route": str,
    "retrieved_docs": List[Dict],
    "extracted_data": Dict,
    "final_response": str,
    "audit_log": List[Dict]
}
```

### VectorStore API

```python
from vectorstore.chroma import VectorStoreManager

vs = VectorStoreManager()

# Add chunks
vs.add_chunks(chunks: List[Dict])

# Search
results = vs.similarity_search(query: str, k: int)
# Returns: List[Document]
```

### Audit Logger API

```python
from audit.logger import AuditLogger

logger = AuditLogger(log_dir="data/logs")

# Log individual step
logger.log_step(
    query_id="uuid",
    step_name="RetrievalAgent",
    status="Success",
    **kwargs
)

# Log full query
logger.log_query(query_id="uuid", state=agent_state)

# Read logs
logs = logger.get_logs()  # Returns List[Dict]
```

---

## Monitoring & Logging

### Log Files

#### `audit_trail.jsonl`
Full query execution logs:
```json
{
  "query_id": "uuid",
  "timestamp": "2025-12-22T20:00:00",
  "query": "What is revenue?",
  "final_response": "The revenue is...",
  "audit_trail": [...]
}
```

#### `audit_trail_steps.jsonl`
Progressive step logs:
```json
{
  "query_id": "uuid",
  "timestamp": "2025-12-22T20:00:01",
  "step": "RouterAgent",
  "status": "Success",
  "route": "document"
}
```

### Metrics Tracked
- Query classification route
- Retrieved document count
- Extraction skips/successes
- Response generation time (implicit via timestamps)
- Error rates by agent

---

## Performance & Optimization

### Resource Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB |
| **GPU** | None | NVIDIA GPU with 4GB+ VRAM |
| **Storage** | 5GB | 20GB |
| **CPU** | 4 cores | 8 cores |

### Optimization Strategies

#### 1. Model Selection
- Use smaller models for CPU: `qwen2.5:3b`
- Use larger models for GPU: `llama3` or `mixtral`

#### 2. Chunking Strategy
```python
# Adjust in document_processing/chunking.py
RecursiveCharacterTextSplitter(
    chunk_size=500,    # Increase for more context
    chunk_overlap=50   # Increase to reduce boundary loss
)
```

#### 3. Retrieval Top-K
```python
# In agents/retrieval.py
results = self.vector_store.similarity_search(query, k=5)
# Reduce k for faster search, increase for better recall
```

#### 4. Timeout Configuration
```python
# In agents/analysis.py and agents/extraction.py
ChatOllama(..., timeout=30)  # Adjust per hardware
```

#### 5. GPU Utilization
Ensure Ollama uses GPU:
```bash
# Check GPU usage
nvidia-smi

# Verify Ollama settings
ollama show qwen2.5:3b
```

### Performance Benchmarks
(Example on consumer hardware)

| Model | Hardware | Query Latency | Throughput |
|-------|----------|---------------|------------|
| qwen2.5:3b | CPU (8-core) | 5-10s | ~6 queries/min |
| qwen2.5:3b | GPU (RTX 3060) | 1-3s | ~20 queries/min |
| llama3 | GPU (RTX 3060) | 3-7s | ~10 queries/min |

---

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Error
**Error:** `Failed to connect to Ollama`
**Solution:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Verify .env configuration
cat .env
```

#### 2. GPU Not Used
**Error:** Slow inference on GPU system
**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall Ollama with GPU support
# Follow official Ollama GPU setup guide
```

#### 3. Timeout Errors
**Error:** `LLM call timed out`
**Solution:**
- Increase timeout in agent files
- Use smaller model
- Reduce context size

#### 4. ChromaDB Lock Error
**Error:** `Database locked`
**Solution:**
```bash
# Remove lock file
rm data/chroma/chroma.sqlite3-wal

# Or reset database
rm -rf data/chroma/*
```

#### 5. Import Errors
**Error:** `ModuleNotFoundError`
**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version (3.8+)
python --version
```

---

## Advanced Topics

### Custom Agent Development
To add a new agent:
1. Create file in `agents/`
2. Implement `invoke(state: AgentState) -> Dict`
3. Add to orchestrator graph
4. Update conditional routing logic

### Prompt Engineering
Modify prompts in:
- `agents/router.py` - Classification prompt
- `agents/extraction.py` - Extraction prompt
- `agents/analysis.py` - Response generation prompt

### Multi-Language Support
To support non-English documents:
1. Change embedding model in `core/config.py`
2. Update LLM model to multilingual variant
3. Adjust prompts for target language

---

## Security Considerations

### Data Privacy
- All processing is local (no external API calls)
- Documents stored on-disk (not cloud)
- Vector embeddings persist locally

### Access Control
- No built-in authentication (add reverse proxy if needed)
- Streamlit runs on localhost by default
- Expose carefully in production

### Input Validation
- PDF file type validation
- Query length limits (implicit via LLM context)
- Error handling for malicious inputs

---

## Future Enhancements

### Planned Features
- [ ] Multi-format support (DOCX, TXT, HTML)
- [ ] Multi-language document support
- [ ] Advanced extraction (forms, handwriting)
- [ ] Query history persistence
- [ ] Export chat to PDF
- [ ] Fine-tuning on domain-specific data
- [ ] Distributed processing for large documents

### Scalability Improvements
- [ ] Redis for distributed caching
- [ ] PostgreSQL for audit logs
- [ ] Kubernetes deployment
- [ ] Load balancing for multi-user

---

## Contributing

### Development Setup
```bash
# Install dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests (if available)
pytest tests/

# Format code
black .

# Lint
flake8 .
```

### Code Style
- PEP 8 compliance
- Type hints where applicable
- Docstrings for public methods
- Progressive logging for debugging

---

## License

Copyright © 2025. All rights reserved.

---

## Support & Contact

For issues and questions:
- GitHub Issues: `<repository-url>/issues`
- Documentation: This file
- Ollama Support: https://ollama.ai

---

**Last Updated:** December 23, 2025  
**Version:** 2.0  
**Status:** Production Ready
