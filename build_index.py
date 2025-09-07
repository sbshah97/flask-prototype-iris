#!/usr/bin/env python3
"""
Standalone preprocessing script for NITK Academic Advisor RAG Pipeline.

This script downloads PDFs, processes documents, and builds the FAISS vector index
independently of the Flask chat application.

Usage:
    python build_index.py [--force-rebuild]
"""

import os
import sys
import argparse
import logging
import requests
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NITKIndexBuilder:
    def __init__(self):
        self.embeddings = None
        
    def download_pdfs(self) -> bool:
        """Download PDFs if they don't exist locally"""
        logger.info("=" * 50)
        logger.info("STEP 1: Downloading PDFs")
        logger.info("=" * 50)
        
        os.makedirs(PDF_DIR, exist_ok=True)
        
        for filename, url in PDFS.items():
            filepath = os.path.join(PDF_DIR, filename)
            
            if os.path.exists(filepath):
                logger.info(f"✅ PDF already exists: {filename}")
                continue
                
            try:
                logger.info(f"📥 Downloading {filename}...")
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r  Progress: {progress:.1f}% ({downloaded:,} / {total_size:,} bytes)", end='')
                    
                    print()  # New line after progress
                        
                logger.info(f"✅ Successfully downloaded {filename}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {filename}: {str(e)}")
                return False
                
        logger.info("✅ All PDFs ready")
        return True
    
    def load_documents(self) -> List[Document]:
        """Load PDF documents"""
        logger.info("=" * 50)
        logger.info("STEP 2: Loading PDF Documents")
        logger.info("=" * 50)
        
        documents = []
        
        for filename in PDFS.keys():
            filepath = os.path.join(PDF_DIR, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"⚠️  PDF not found: {filepath}")
                continue
                
            try:
                logger.info(f"📖 Loading {filename}...")
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                
                # Add metadata about source handbook
                handbook_type = "BTech" if "Btech" in filename else "PG"
                for doc in docs:
                    doc.metadata["handbook"] = handbook_type
                    doc.metadata["source_file"] = filename
                    
                documents.extend(docs)
                logger.info(f"✅ Loaded {len(docs)} pages from {filename}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {filename}: {str(e)}")
                
        logger.info(f"✅ Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        logger.info("=" * 50)
        logger.info("STEP 3: Chunking Documents")
        logger.info("=" * 50)
        
        logger.info(f"📝 Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"✅ Created {len(chunks)} chunks from documents")
        
        # Show some statistics
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        logger.info(f"📊 Average chunk size: {avg_size:.0f} characters")
        
        return chunks
    
    def initialize_embeddings(self):
        """Initialize embedding model"""
        logger.info("=" * 50)
        logger.info("STEP 4: Initialize Embeddings")
        logger.info("=" * 50)
        
        if self.embeddings is None:
            logger.info(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded")
    
    def build_vector_store(self, chunks: List[Document], force_rebuild: bool = False) -> bool:
        """Build FAISS vector store from chunks"""
        logger.info("=" * 50)
        logger.info("STEP 5: Building Vector Store")
        logger.info("=" * 50)
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Check if index already exists
        if not force_rebuild and os.path.exists(INDEX_DIR):
            try:
                # Try to load existing index to validate it
                logger.info("🔍 Checking existing FAISS index...")
                existing_index = FAISS.load_local(
                    INDEX_DIR, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                vector_count = existing_index.index.ntotal
                logger.info(f"✅ Valid existing index found with {vector_count} vectors")
                logger.info("Use --force-rebuild to rebuild anyway")
                return True
            except Exception as e:
                logger.warning(f"⚠️  Existing index invalid: {str(e)}, rebuilding...")
        
        if not chunks:
            logger.error("❌ No chunks to process")
            return False
        
        # Create vector store
        try:
            logger.info(f"🔄 Creating FAISS index from {len(chunks)} chunks...")
            logger.info("This may take several minutes...")
            
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            logger.info("💾 Saving FAISS index to disk...")
            vectorstore.save_local(INDEX_DIR)
            
            vector_count = vectorstore.index.ntotal
            logger.info(f"✅ Created and saved FAISS index with {vector_count} vectors")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {str(e)}")
            return False
    
    def validate_index(self) -> bool:
        """Validate the built index"""
        logger.info("=" * 50)
        logger.info("STEP 6: Validating Index")
        logger.info("=" * 50)
        
        try:
            logger.info("🔍 Loading and testing the built index...")
            vectorstore = FAISS.load_local(
                INDEX_DIR, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            vector_count = vectorstore.index.ntotal
            logger.info(f"✅ Index loaded successfully with {vector_count} vectors")
            
            # Test a simple similarity search
            logger.info("🔍 Testing similarity search...")
            test_query = "attendance requirements"
            results = vectorstore.similarity_search(test_query, k=3)
            
            logger.info(f"✅ Similarity search returned {len(results)} results")
            logger.info(f"📄 Sample result snippet: {results[0].page_content[:100]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Index validation failed: {str(e)}")
            return False
    
    def build_complete_index(self, force_rebuild: bool = False) -> bool:
        """Complete index building process"""
        logger.info("🚀 Starting NITK Academic Advisor Index Building")
        logger.info(f"📍 Working directory: {os.getcwd()}")
        logger.info(f"📁 Data directory: {os.path.abspath(DATA_DIR)}")
        
        try:
            # Step 1: Download PDFs
            if not self.download_pdfs():
                return False
            
            # Step 2: Load documents
            documents = self.load_documents()
            if not documents:
                logger.error("❌ No documents loaded")
                return False
            
            # Step 3: Chunk documents
            chunks = self.chunk_documents(documents)
            if not chunks:
                logger.error("❌ No chunks created")
                return False
            
            # Step 4: Initialize embeddings
            self.initialize_embeddings()
            
            # Step 5: Build vector store
            if not self.build_vector_store(chunks, force_rebuild):
                return False
            
            # Step 6: Validate
            if not self.validate_index():
                return False
            
            logger.info("=" * 50)
            logger.info("🎉 INDEX BUILDING COMPLETE!")
            logger.info("=" * 50)
            logger.info("✅ FAISS vector index is ready")
            logger.info("✅ You can now start the chat application with: python app.py")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Index building failed: {str(e)}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Build FAISS index for NITK Academic Advisor")
    parser.add_argument("--force-rebuild", action="store_true", 
                       help="Force rebuild even if index exists")
    
    args = parser.parse_args()
    
    builder = NITKIndexBuilder()
    success = builder.build_complete_index(args.force_rebuild)
    
    if success:
        sys.exit(0)
    else:
        logger.error("❌ Index building failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()