# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Two-step process (IMPORTANT):**

**Step 1: Build RAG index (required first time):**
```bash
python build_index.py [--force-rebuild]
# Downloads PDFs, processes documents, builds FAISS vector index
# Takes ~2 minutes, only needed once unless PDFs change
```

**Step 2: Start the application:**
```bash
python app.py
# Runs on http://localhost:8000 by default
# Fast startup since index is pre-built
```

**Testing:**
```bash
python test_app.py  # Comprehensive test suite
python -m pytest   # If pytest tests are added
```

**Environment setup:**
```bash
python setup.py    # Interactive setup script
```

**Manual dependency installation:**
```bash
pip install -r requirements.txt
```

## Architecture Overview

This is a Flask-based RAG (Retrieval-Augmented Generation) application that serves as an AI academic advisor for NITK. The system consists of:

**Core Components:**
- `build_index.py`: Standalone preprocessing script for RAG pipeline setup
- `app.py`: Flask web server with REST API endpoints  
- `rag_pipeline.py`: Simplified RAG implementation focused on loading pre-built index
- `NITKRAGPipeline` class: Loads vector store and handles LLM querying

**Key Endpoints:**
- `/chat` (POST): Main Q&A endpoint - expects `{"question": "..."}` JSON
- `/status`: Detailed system status including vector count and LLM configuration
- `/health`: Simple health check
- `/test`: Test endpoint with sample question

**Data Flow:**
1. **Preprocessing (build_index.py)**: PDFs downloaded, chunked, and indexed once
2. **Runtime (app.py)**: Loads pre-built FAISS index on startup
3. **Query Processing**: User queries processed through similarity search + LLM generation
4. **Fast Response**: No initialization delay, immediate query processing

**LLM Provider Hierarchy:**
1. **Primary**: Google Gemini 2.0 Flash (requires `GOOGLE_API_KEY`)
2. **Fallback**: Ollama local models 
3. **Final fallback**: OpenAI GPT models

## Configuration

**Required Environment Variables:**
- `GOOGLE_API_KEY`: Google AI API key for Gemini models
- Copy `.env.example` to `.env` and configure

**Optional Configuration:**
- `LLM_PROVIDER`: gemini (default), ollama, openai  
- `LLM_MODEL`: gemini-2.0-flash-exp (default)
- `EMBEDDING_MODEL`: sentence-transformers/all-mpnet-base-v2 (default)
- `CHUNK_SIZE`: 1200, `CHUNK_OVERLAP`: 200, `RETRIEVAL_K`: 5

## Important Implementation Details

**Two-Phase Architecture:**
- **Phase 1 (build_index.py)**: Preprocessing runs separately, builds FAISS index
- **Phase 2 (app.py)**: Fast startup, loads pre-built index immediately  
- Global `_pipeline_initialized` flag tracks runtime initialization state
- Pipeline singleton managed through `get_pipeline()` function

**Document Processing:**
- PDFs downloaded to `data/pdfs/` directory (4MB+ BTech + PG handbooks)
- FAISS index persisted in `data/faiss_index/` with 2800+ vectors
- Metadata includes handbook type (BTech/PG) and source file info
- Text chunks: 1200 chars with 200 char overlap for context

**Error Handling:**
- Graceful LLM provider fallbacks on initialization failure
- Clear error messages guide users to run preprocessing first
- API returns structured JSON with `success` boolean and `sources` array
- Pipeline failures don't crash the Flask app

**Deployment Ready:**
- Configured for Railway deployment with `Procfile` and `railway.json`
- Uses Gunicorn WSGI server in production
- **Important**: Run `python build_index.py` in deployment environment

## Testing and Debugging

The `test_app.py` script provides comprehensive testing:
- Import verification
- Environment configuration checks  
- Direct RAG pipeline testing
- Live Flask endpoint testing

Use `/status` endpoint for detailed system diagnostics including vector counts and LLM provider status.