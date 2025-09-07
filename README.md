# NITK Academic Advisor 🎓

A Flask-based RAG (Retrieval-Augmented Generation) application that serves as an AI academic advisor for NITK (National Institute of Technology Karnataka). The app ingests official NITK curriculum handbooks and provides intelligent answers about academic policies, procedures, and regulations.

![NITK Advisor](https://img.shields.io/badge/NITK-Academic%20Advisor-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![Flask](https://img.shields.io/badge/Flask-3.0+-red)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.0%20Flash-yellow)

## ✨ Features

- **🤖 AI-Powered Academic Advisor**: Uses Google Gemini 2.0 Flash for intelligent responses
- **📚 Document RAG**: Processes official NITK BTech and PG curriculum handbooks
- **🔍 Semantic Search**: FAISS-powered vector similarity search
- **💬 Interactive Chat UI**: Clean, responsive web interface
- **📱 Mobile Friendly**: Responsive design for all devices
- **🚀 Production Ready**: Deployed on Railway with automated CI/CD
- **🔄 Auto-Ingestion**: Automatically downloads and processes PDFs
- **📖 Source Citations**: Shows relevant handbook sections with each answer

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask App      │    │  RAG Pipeline   │
│   (Chat UI)     ├────┤  (REST API)      ├────┤  (LangChain)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        ▼
                        ┌──────────────────┐    ┌─────────────────┐
                        │   Railway        │    │  Vector Store   │
                        │   (Hosting)      │    │  (FAISS)        │
                        └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  PDF Documents  │
                                                │  (NITK Handbooks)│
                                                └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google AI API Key (for Gemini)
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd flask-prototype-iris
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Build the RAG index (REQUIRED)**
   ```bash
   python build_index.py
   ```
   This downloads PDFs and builds the FAISS vector index (takes ~2 minutes)

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Open in browser**
   ```
   http://localhost:8000
   ```

### Two-Step Setup Process

**Step 1: Preprocessing (One-time setup)**
```bash
python build_index.py [--force-rebuild]
```
- Downloads NITK curriculum PDFs
- Processes documents and creates text chunks  
- Builds FAISS vector index with 2800+ embeddings
- Only needs to be run once (unless PDFs are updated)

**Step 2: Start Chat Application**
```bash
python app.py
```
- Loads pre-built index (fast startup)
- Ready to answer questions immediately

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/health` | GET | Health check |
| `/status` | GET | Detailed system status |
| `/chat` | POST | Ask questions (JSON: `{"question": "..."}`) |
| `/test` | GET | Test with sample question |

**Note:** The `/ingest` endpoint has been removed. Use `python build_index.py` for preprocessing.

## 🔧 Configuration

### Environment Variables

```bash
# Primary LLM (Recommended)
GOOGLE_API_KEY=your_google_api_key_here
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash-exp

# Alternative LLMs
OPENAI_API_KEY=your_openai_key  # Fallback
OLLAMA_BASE_URL=http://localhost:11434  # Local LLM

# RAG Configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
RETRIEVAL_K=5
LLM_TEMPERATURE=0.2

# Flask Settings
FLASK_ENV=production
DEBUG=False
```

### Supported LLM Providers

1. **Google Gemini** (Primary - Cost-effective)
   - `gemini-2.0-flash-exp`
   - `gemini-1.5-pro`
   - `gemini-1.5-flash`

2. **Ollama** (Local fallback)
   - `llama3.1:8b`
   - Any Ollama model

3. **OpenAI** (Final fallback)
   - `gpt-3.5-turbo`
   - `gpt-4`

## 🚢 Deployment

### Railway Deployment

1. **Connect to Railway**
   ```bash
   npm install -g @railway/cli
   railway login
   railway init
   ```

2. **Set Environment Variables**
   ```bash
   railway variables set GOOGLE_API_KEY=your_key_here
   railway variables set LLM_PROVIDER=gemini
   railway variables set LLM_MODEL=gemini-2.0-flash-exp
   ```

3. **Deploy**
   ```bash
   railway up
   ```

### GitHub Actions CI/CD

The repository includes automated deployment via GitHub Actions:

1. **Set Repository Secrets**:
   - `RAILWAY_TOKEN`: Your Railway API token
   - `GOOGLE_API_KEY`: Your Google AI API key
   - `LLM_PROVIDER`: `gemini`
   - `LLM_MODEL`: `gemini-2.0-flash-exp`

2. **Push to main branch** - Deployment happens automatically!

## 📖 Usage Examples

### Sample Questions

- "What are the attendance requirements for BTech students?"
- "How is CGPA calculated in NITK?"
- "What is the minimum credit requirement for graduation?"
- "What are the eligibility criteria for semester promotion?"
- "How many backlogs are allowed per semester?"
- "What is the grading system for PG courses?"

### API Usage

```bash
# First-time setup: Build the index
python build_index.py

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the attendance requirements?"}'

# Check system status
curl http://localhost:8000/status

# Rebuild index if needed
python build_index.py --force-rebuild
```

## 🛠️ Tech Stack

- **Backend**: Flask 3.0, Python 3.11+
- **AI/ML**: LangChain, Google Gemini 2.0 Flash
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: HuggingFace sentence-transformers
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Deployment**: Railway, Docker
- **CI/CD**: GitHub Actions

## 📁 Project Structure

```
flask-prototype-iris/
├── app.py                 # Main Flask application
├── rag_pipeline.py        # RAG implementation
├── requirements.txt       # Python dependencies
├── runtime.txt           # Python version
├── Procfile              # Railway configuration
├── railway.json          # Railway settings
├── .env.example          # Environment template
├── .gitignore            # Git ignore rules
├── data/                 # Data directory
│   ├── pdfs/             # PDF storage
│   └── faiss_index/      # Vector index
├── static/               # Frontend assets
│   ├── style.css         # Styles
│   └── script.js         # JavaScript
├── templates/            # HTML templates
│   └── chat.html         # Chat interface
├── .github/workflows/    # CI/CD
│   └── deploy.yml        # GitHub Actions
└── README.md            # Documentation
```

## 🔍 How It Works

1. **Document Ingestion**: Downloads NITK PDF handbooks automatically
2. **Text Processing**: Extracts and chunks text with overlap for context
3. **Embedding**: Creates vector embeddings using HuggingFace models
4. **Vector Storage**: Stores embeddings in FAISS for fast similarity search
5. **Query Processing**: Embeds user questions and retrieves relevant chunks
6. **Response Generation**: Uses Gemini to generate academic advisor responses
7. **Source Citation**: Returns relevant handbook sections with answers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- Create an [Issue](https://github.com/your-repo/issues) for bug reports
- Check the [Wiki](https://github.com/your-repo/wiki) for detailed documentation
- Join our [Discussions](https://github.com/your-repo/discussions) for questions

## 🙏 Acknowledgments

- **NITK** for providing comprehensive curriculum handbooks
- **Google** for the excellent Gemini 2.0 Flash model
- **LangChain** for the RAG framework
- **Railway** for seamless deployment
- **HuggingFace** for embedding models

---

**Built with ❤️ for the NITK community**