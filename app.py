import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from rag_pipeline import initialize_pipeline, query_pipeline, get_pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")

# Global variables
_pipeline_initialized = False

@app.route("/")
def index():
    """Serve the chat interface"""
    return render_template("chat.html")

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "pipeline_initialized": _pipeline_initialized,
        "service": "NITK Academic Advisor"
    })

@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Force rebuild the vector index from PDFs
    This endpoint downloads PDFs and rebuilds the FAISS index
    """
    global _pipeline_initialized
    
    try:
        logger.info("Starting manual ingestion process...")
        
        # Force rebuild the pipeline
        success = initialize_pipeline(force_rebuild=True)
        
        if success:
            _pipeline_initialized = True
            pipeline = get_pipeline()
            vector_count = pipeline.vectorstore.index.ntotal if pipeline.vectorstore else 0
            
            logger.info(f"Ingestion completed successfully with {vector_count} vectors")
            return jsonify({
                "status": "success",
                "message": "Documents ingested and indexed successfully",
                "vector_count": vector_count
            })
        else:
            logger.error("Ingestion failed")
            return jsonify({
                "status": "error",
                "message": "Failed to ingest documents"
            }), 500
            
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Ingestion failed: {str(e)}"
        }), 500

@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint for question answering
    Expects JSON: {"question": "your question here"}
    """
    global _pipeline_initialized
    
    try:
        # Parse request
        data = request.get_json(force=True) or {}
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({
                "error": "Missing or empty 'question' field"
            }), 400
        
        logger.info(f"Received question: {question[:100]}...")
        
        # Initialize pipeline if needed
        if not _pipeline_initialized:
            logger.info("Pipeline not initialized, initializing now...")
            success = initialize_pipeline(force_rebuild=False)
            if success:
                _pipeline_initialized = True
            else:
                return jsonify({
                    "error": "Failed to initialize RAG pipeline. Please try the /ingest endpoint first."
                }), 500
        
        # Query the pipeline
        result = query_pipeline(question)
        
        if result["success"]:
            logger.info("Query completed successfully")
            return jsonify({
                "answer": result["answer"],
                "sources": result["sources"],
                "question": question
            })
        else:
            logger.error("Query failed")
            return jsonify({
                "error": result["answer"]
            }), 500
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

@app.route("/status")
def status():
    """
    Detailed status endpoint for debugging
    """
    global _pipeline_initialized
    
    pipeline = get_pipeline()
    
    status_info = {
        "pipeline_initialized": _pipeline_initialized,
        "vector_store_loaded": pipeline.vectorstore is not None,
        "retriever_ready": pipeline.retriever is not None,
        "llm_ready": pipeline.llm is not None,
        "chain_ready": pipeline.chain is not None,
        "environment": {
            "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
            "llm_model": os.getenv("LLM_MODEL", "gemini-2.0-flash-exp"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY")),
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY"))
        }
    }
    
    if pipeline.vectorstore:
        status_info["vector_count"] = pipeline.vectorstore.index.ntotal
    
    return jsonify(status_info)

@app.route("/test", methods=["GET"])
def test_endpoint():
    """
    Test endpoint with a sample question
    """
    sample_question = "What are the attendance requirements for BTech students?"
    
    try:
        # Initialize if needed
        global _pipeline_initialized
        if not _pipeline_initialized:
            success = initialize_pipeline(force_rebuild=False)
            if success:
                _pipeline_initialized = True
            else:
                return jsonify({
                    "error": "Pipeline initialization failed"
                }), 500
        
        # Query with sample question
        result = query_pipeline(sample_question)
        
        return jsonify({
            "test_question": sample_question,
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Test failed: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Initialize pipeline on startup (optional, can be done on first request)
def startup_initialization():
    """Initialize pipeline on startup"""
    global _pipeline_initialized
    try:
        logger.info("Attempting to initialize pipeline on startup...")
        success = initialize_pipeline(force_rebuild=False)
        if success:
            _pipeline_initialized = True
            logger.info("Pipeline initialized successfully on startup")
        else:
            logger.warning("Pipeline initialization failed on startup - will retry on first request")
    except Exception as e:
        logger.warning(f"Startup initialization failed: {str(e)} - will retry on first request")

if __name__ == "__main__":
    # Optional: Initialize on startup
    # startup_initialization()
    
    # Get configuration
    host = "0.0.0.0"
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting NITK Academic Advisor on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'gemini')}")
    
    app.run(host=host, port=port, debug=debug)