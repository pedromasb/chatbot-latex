"""
Thesis Chat: A Python module for LaTeX document processing and conversational AI.

This module provides functionality to:
1. Process LaTeX thesis documents into structured chunks
2. Create semantic embeddings and store them in Pinecone vector database
3. Perform intelligent queries using retrieval-augmented generation (RAG)

Main Components:
- LaTeXProcessor: Handles LaTeX document parsing and chunking
- VectorStore: Manages Pinecone vector database operations
- QueryEngine: Implements semantic search and LLM-based responses
- ThesisChat: Main interface class combining all functionality

Example Usage:
    from thesis_chat import ThesisChat
    
    chat = ThesisChat(
        pinecone_api_key="your-key",
        openai_api_key="your-key",
        index_name="thesis-chat"
    )
    
    # Process and index a LaTeX document
    chat.process_latex_file("path/to/thesis.tex")
    
    # Query the document
    response = chat.query("What are the main conclusions?")
    print(response)
"""

from .core.latex_processor import LaTeXProcessor
from .core.vector_store import VectorStore
from .core.query_engine import QueryEngine
from .core.thesis_chat import ThesisChat
from .utils.config import Config
from .utils.exceptions import ThesisChatError, LaTeXProcessingError, VectorStoreError, QueryError

__version__ = "1.0.0"
__author__ = "Thesis Chat Team"
__email__ = "support@thesischat.com"

__all__ = [
    "ThesisChat",
    "LaTeXProcessor", 
    "VectorStore",
    "QueryEngine",
    "Config",
    "ThesisChatError",
    "LaTeXProcessingError", 
    "VectorStoreError",
    "QueryError"
]