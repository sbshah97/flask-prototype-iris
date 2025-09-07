import os
import logging
import requests
from typing import List, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

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
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

# PDF configurations
PDFS = {
    "Btech_Curriculum_2023.pdf": os.getenv("BTECH_PDF_URL", "https://www.nitk.ac.in/document/attachments/8305/Btech_Curriculum_2023.pdf"),
    "PG_Curriculum_2023.pdf": os.getenv("PG_PDF_URL", "https://www.nitk.ac.in/document/attachments/8306/PG_Curriculum_2023.pdf")
}

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
        
    def download_pdfs(self) -> bool:
        """Download PDFs if they don't exist locally"""
        os.makedirs(PDF_DIR, exist_ok=True)
        
        for filename, url in PDFS.items():
            filepath = os.path.join(PDF_DIR, filename)
            
            if os.path.exists(filepath):
                logger.info(f"PDF already exists: {filename}")
                continue
                
            try:
                logger.info(f"Downloading {filename}...")
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                logger.info(f"Successfully downloaded {filename}")
                
            except Exception as e:
                logger.error(f"Failed to download {filename}: {str(e)}")
                return False
                
        return True
    
    def load_documents(self) -> List[Document]:
        """Load PDF documents"""
        documents = []
        
        for filename in PDFS.keys():
            filepath = os.path.join(PDF_DIR, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"PDF not found: {filepath}")
                continue
                
            try:
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                
                # Add metadata about source handbook
                handbook_type = "BTech" if "Btech" in filename else "PG"
                for doc in docs:
                    doc.metadata["handbook"] = handbook_type
                    doc.metadata["source_file"] = filename
                    
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {filename}")
                
            except Exception as e:
                logger.error(f"Failed to load {filename}: {str(e)}")
                
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from documents")
        return chunks
    
    def initialize_embeddings(self):
        """Initialize embedding model"""
        if self.embeddings is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    def build_vector_store(self, force_rebuild: bool = False) -> bool:
        """Build or load vector store"""
        os.makedirs(DATA_DIR, exist_ok=True)
        
        self.initialize_embeddings()
        
        # Try to load existing index
        if not force_rebuild and os.path.exists(INDEX_DIR):
            try:
                logger.info("Loading existing FAISS index...")
                self.vectorstore = FAISS.load_local(
                    INDEX_DIR, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing index with {self.vectorstore.index.ntotal} vectors")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing index: {str(e)}, rebuilding...")
        
        # Build new index
        logger.info("Building new FAISS index...")
        
        # Download PDFs if needed
        if not self.download_pdfs():
            logger.error("Failed to download required PDFs")
            return False
        
        # Load and process documents
        documents = self.load_documents()
        if not documents:
            logger.error("No documents loaded")
            return False
            
        chunks = self.chunk_documents(documents)
        if not chunks:
            logger.error("No chunks created")
            return False
        
        # Create vector store
        try:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            self.vectorstore.save_local(INDEX_DIR)
            logger.info(f"Created and saved FAISS index with {self.vectorstore.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
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
    
    def initialize(self, force_rebuild: bool = False) -> bool:
        """Initialize the complete RAG pipeline"""
        try:
            # Build vector store
            if not self.build_vector_store(force_rebuild):
                return False
            
            # Build retrieval chain
            self.build_retrieval_chain()
            
            logger.info("RAG pipeline initialized successfully")
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

def initialize_pipeline(force_rebuild: bool = False) -> bool:
    """Initialize the global pipeline"""
    pipeline = get_pipeline()
    return pipeline.initialize(force_rebuild)

def query_pipeline(question: str) -> dict:
    """Query the pipeline with a question"""
    pipeline = get_pipeline()
    return pipeline.query(question)