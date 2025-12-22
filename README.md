# DocuMind AI

Intelligent Document Processing Platform.

## Setup

### Option 1: Docker (Recommended)
1. **Prerequisites**:
   - Docker Desktop installed
   - [Ollama](https://ollama.com/) installed and running locally (`ollama serve`)
   - Pull the LLaMA 3 model: `ollama pull llama3`

2. **Run**:
   ```bash
   docker compose up --build
   ```
   The app will connect to your local Ollama instance.

### Option 2: Native Python
1. **Prerequisites**:
   - Python 3.10+
   - **Tesseract OCR**:
     - **Windows**: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki). Add installation path (e.g., `C:\Program Files\Tesseract-OCR`) to System PATH.
     - **Linux**: `sudo apt-get install tesseract-ocr`
     - **Mac**: `brew install tesseract`

2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run**:
   ```bash
   streamlit run app.py
   ```

### Access
- App: http://localhost:8501

## Architecture
- **UI**: Streamlit
- **LLM**: LLaMA 3 via Ollama
- **Vector DB**: ChromaDB
- **Orchestration**: LangGraph

## Features
- **PDF Ingestion**: Supports digital and scanned PDFs (OCR via Tesseract).
- **RAG**: Retrieval Augmented Generation using local LLaMA 3.
- **Audit Trail**: Full logging of queries and reasoning steps.
- **Table Extraction**: Detects and extracts tables.
