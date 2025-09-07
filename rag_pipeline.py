import os
import logging
from typing import Optional
from dotenv import load_dotenv

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Try importing LLMs with fallbacks
try:
    from langchain.llms import Ollama
except ImportError:
    Ollama = None

try:
    from langchain_google_genai import GoogleGenerativeAI
except ImportError:
    GoogleGenerativeAI = None

try:
    from langchain.llms import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

# Configuration
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# RAG Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 5))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")

# System prompt for academic advisor behavior
SYSTEM_PROMPT = """You are an academic advisor for NITK (National Institute of Technology Karnataka). 
You help students and faculty understand academic policies, procedures, and regulations based on the official curriculum handbooks.

Instructions:
1. Answer questions precisely using ONLY the provided context from NITK handbooks
2. If the answer is not in the documents, clearly state "I don't find this information in the available NITK handbooks"
3. When possible, quote specific policy clauses (e.g., "According to clause G5.12..." or "As per section 4.2...")
4. Keep answers concise, accurate, and policy-focused
5. Maintain a helpful, professional academic advisor tone
6. If referring to specific procedures, mention the relevant handbook (BTech or PG)

Context from NITK Handbooks:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NITKRAGPipeline:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None
        
    def initialize_embeddings(self):
        """Initialize embedding model"""
        if self.embeddings is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    def load_vector_store(self) -> bool:
        """Load pre-built FAISS vector store"""
        if not os.path.exists(INDEX_DIR):
            logger.error(f"FAISS index not found at {INDEX_DIR}")
            logger.error("Please run 'python build_index.py' first to create the vector index")
            return False
        
        try:
            self.initialize_embeddings()
            logger.info("Loading pre-built FAISS index...")
            self.vectorstore = FAISS.load_local(
                INDEX_DIR, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            vector_count = self.vectorstore.index.ntotal
            logger.info(f"✅ Successfully loaded FAISS index with {vector_count} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            logger.error("The index may be corrupted. Try running 'python build_index.py --force-rebuild'")
            return False
    
    def initialize_llm(self):
        """Initialize LLM based on configuration"""
        if self.llm is not None:
            return
        
        # Try Gemini first (primary provider)
        if LLM_PROVIDER.lower() == "gemini" and GoogleGenerativeAI:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                try:
                    self.llm = GoogleGenerativeAI(
                        model=LLM_MODEL,
                        google_api_key=api_key,
                        temperature=LLM_TEMPERATURE
                    )
                    logger.info(f"Initialized Gemini LLM: {LLM_MODEL}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini: {str(e)}")
            else:
                logger.warning("GOOGLE_API_KEY not found for Gemini")
        
        # Fallback to Ollama
        if LLM_PROVIDER.lower() == "ollama" and Ollama:
            try:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                self.llm = Ollama(
                    model=LLM_MODEL if "gemini" not in LLM_MODEL else "llama3.1:8b",
                    base_url=base_url,
                    temperature=LLM_TEMPERATURE
                )
                logger.info(f"Initialized Ollama LLM")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {str(e)}")
        
        # Final fallback to OpenAI
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = OpenAI(
                    model="gpt-3.5-turbo",
                    temperature=LLM_TEMPERATURE
                )
                logger.info("Initialized OpenAI LLM")
                return
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {str(e)}")
        
        raise Exception("No LLM provider available. Please configure Gemini (GOOGLE_API_KEY), Ollama, or OpenAI.")
    
    def build_retrieval_chain(self):
        """Build the retrieval QA chain"""
        if not self.vectorstore:
            raise Exception("Vector store not initialized")
        
        self.initialize_llm()
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        )
        
        # Create retrieval chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )
        
        logger.info("Retrieval chain initialized successfully")
    
    def query(self, question: str) -> dict:
        """Query the RAG system"""
        if not self.chain:
            raise Exception("RAG chain not initialized")
        
        try:
            result = self.chain({"query": question})
            
            # Process sources
            sources = []
            for doc in result.get("source_documents", []):
                meta = doc.metadata or {}
                source_info = {
                    "handbook": meta.get("handbook", "Unknown"),
                    "source_file": meta.get("source_file", "Unknown"),
                    "page": meta.get("page", "n/a"),
                    "snippet": doc.page_content[:300].replace("\n", " ").strip()
                }
                sources.append(source_info)
            
            return {
                "answer": result.get("result", ""),
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def initialize(self) -> bool:
        """Initialize the RAG pipeline with pre-built index"""
        try:
            # Load pre-built vector store
            if not self.load_vector_store():
                return False
            
            # Build retrieval chain
            self.build_retrieval_chain()
            
            logger.info("✅ RAG pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            return False

# Global pipeline instance
_pipeline: Optional[NITKRAGPipeline] = None

def get_pipeline() -> NITKRAGPipeline:
    """Get or create global pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = NITKRAGPipeline()
    return _pipeline

def initialize_pipeline() -> bool:
    """Initialize the global pipeline"""
    pipeline = get_pipeline()
    return pipeline.initialize()

def query_pipeline(question: str) -> dict:
    """Query the pipeline with a question"""
    pipeline = get_pipeline()
    return pipeline.query(question)