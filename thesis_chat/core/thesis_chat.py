"""Main ThesisChat class that combines all functionality."""

import os
from typing import List, Dict, Any, Optional
from .latex_processor import LaTeXProcessor, Chunk
from .vector_store import VectorStore
from .query_engine import QueryEngine
from ..utils.exceptions import ThesisChatError
from ..utils.config import Config


class ThesisChat:
    """
    Main interface for the thesis chat system.
    
    Combines LaTeX processing, vector storage, and query capabilities
    into a single easy-to-use interface.
    """

    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        index_name: str = "thesis-chat",
        namespace: str = "default",
        config: Optional[Config] = None
    ):
        """
        Initialize ThesisChat with API keys and configuration.
        
        Args:
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key
            index_name: Pinecone index name
            namespace: Pinecone namespace
            config: Optional configuration object
        """
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.index_name = index_name
        self.namespace = namespace
        self.config = config or Config()
        
        # Initialize components
        self._init_components()
        
        # State tracking
        self.is_indexed = False
        self.chunks = []

    def _init_components(self) -> None:
        """Initialize all components."""
        try:
            # LaTeX processor
            self.latex_processor = LaTeXProcessor(
                chunk_size=self.config.chunk_size,
                overlap=self.config.overlap,
                keep_captions=self.config.keep_captions
            )
            
            # Vector store
            self.vector_store = VectorStore(
                api_key=self.pinecone_api_key,
                index_name=self.index_name,
                dimension=self.config.embedding_dimension,
                metric=self.config.metric,
                cloud=self.config.cloud,
                region=self.config.region,
                namespace=self.namespace,
                model_name=self.config.embedding_model
            )
            
            # Query engine will be initialized after vector store is ready
            self.query_engine = None
            
        except Exception as e:
            raise ThesisChatError(f"Failed to initialize components: {str(e)}") from e

    def setup(self, force_recreate_index: bool = False) -> None:
        """
        Set up the thesis chat system.
        
        Args:
            force_recreate_index: Whether to recreate the Pinecone index
        """
        try:
            # Create/connect to Pinecone index
            self.vector_store.create_index(force_recreate=force_recreate_index)
            self.vector_store.connect_to_index()
            
            # Initialize query engine
            self.query_engine = QueryEngine(
                vector_store=self.vector_store,
                openai_api_key=self.openai_api_key,
                reranker_model=self.config.reranker_model,
                llm_model=self.config.llm_model,
                max_context_chunks=self.config.max_context_chunks
            )
            
            print("ThesisChat setup complete!")
            
        except Exception as e:
            raise ThesisChatError(f"Setup failed: {str(e)}") from e

    def process_latex_file(self, latex_file_path: str) -> List[Chunk]:
        """
        Process a LaTeX file into chunks.
        
        Args:
            latex_file_path: Path to the LaTeX file
            
        Returns:
            List of processed chunks
        """
        if not os.path.exists(latex_file_path):
            raise ThesisChatError(f"LaTeX file not found: {latex_file_path}")
        
        try:
            print(f"Processing LaTeX file: {latex_file_path}")
            self.chunks = self.latex_processor.process_latex_file(latex_file_path)
            print(f"Generated {len(self.chunks)} chunks")
            return self.chunks
            
        except Exception as e:
            raise ThesisChatError(f"Failed to process LaTeX file: {str(e)}") from e

    def create_embeddings_and_index(self, chunks: Optional[List[Chunk]] = None) -> None:
        """
        Create embeddings for chunks and index them in Pinecone.
        
        Args:
            chunks: Optional list of chunks (uses instance chunks if None)
        """
        if not self.query_engine:
            raise ThesisChatError("System not set up. Call setup() first.")
        
        target_chunks = chunks or self.chunks
        if not target_chunks:
            raise ThesisChatError("No chunks to process. Call process_latex_file() first.")
        
        try:
            print("Creating embeddings...")
            target_chunks = self.vector_store.create_embeddings(target_chunks)
            
            print("Indexing chunks in Pinecone...")
            self.vector_store.upsert_chunks(target_chunks)
            
            self.is_indexed = True
            self.chunks = target_chunks
            
            print("Indexing complete!")
            
        except Exception as e:
            raise ThesisChatError(f"Failed to create embeddings and index: {str(e)}") from e

    def process_and_index_latex(self, latex_file_path: str) -> None:
        """
        End-to-end processing: LaTeX file to indexed chunks.
        
        Args:
            latex_file_path: Path to the LaTeX file
        """
        try:
            # Process LaTeX
            self.process_latex_file(latex_file_path)
            
            # Create embeddings and index
            self.create_embeddings_and_index()
            
            print(f"Successfully processed and indexed {len(self.chunks)} chunks")
            
        except Exception as e:
            raise ThesisChatError(f"End-to-end processing failed: {str(e)}") from e

    def query(
        self,
        question: str,
        language: str = "auto",
        include_sources: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Query the indexed document.
        
        Args:
            question: User's question
            language: Response language preference
            include_sources: Whether to include source citations
            temperature: LLM temperature
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.query_engine:
            raise ThesisChatError("System not set up. Call setup() first.")
        
        if not self.is_indexed:
            raise ThesisChatError("No content indexed. Call process_and_index_latex() first.")
        
        try:
            return self.query_engine.query(
                question=question,
                top_k_retrieval=self.config.top_k_retrieval,
                language=language,
                include_sources=include_sources,
                temperature=temperature
            )
            
        except Exception as e:
            raise ThesisChatError(f"Query failed: {str(e)}") from e

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search without LLM response generation.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.query_engine:
            raise ThesisChatError("System not set up. Call setup() first.")
        
        try:
            return self.query_engine.search_only(
                query=query,
                top_k_retrieval=self.config.top_k_retrieval,
                top_k_rerank=top_k
            )
            
        except Exception as e:
            raise ThesisChatError(f"Search failed: {str(e)}") from e

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        if not self.vector_store:
            raise ThesisChatError("Vector store not initialized.")
        
        try:
            return self.vector_store.get_index_stats()
        except Exception as e:
            raise ThesisChatError(f"Failed to get index stats: {str(e)}") from e

    def save_chunks(self, output_path: str, include_embeddings: bool = True) -> None:
        """
        Save processed chunks to file.
        
        Args:
            output_path: Output file path
            include_embeddings: Whether to include embeddings in output
        """
        if not self.chunks:
            raise ThesisChatError("No chunks to save. Call process_latex_file() first.")
        
        try:
            if include_embeddings:
                self.vector_store.save_chunks_with_embeddings(self.chunks, output_path)
            else:
                self.latex_processor.save_chunks_to_jsonl(self.chunks, output_path)
            
            print(f"Chunks saved to: {output_path}")
            
        except Exception as e:
            raise ThesisChatError(f"Failed to save chunks: {str(e)}") from e

    def load_chunks(self, file_path: str) -> List[Chunk]:
        """
        Load chunks from file.
        
        Args:
            file_path: Path to chunks file
            
        Returns:
            List of loaded chunks
        """
        try:
            self.chunks = self.vector_store.load_chunks_from_jsonl(file_path)
            
            # Check if chunks have embeddings
            has_embeddings = any(hasattr(chunk, 'embedding') and chunk.embedding for chunk in self.chunks)
            
            if has_embeddings:
                print("Loaded chunks with embeddings")
                # You may want to index these if not already indexed
            else:
                print("Loaded chunks without embeddings")
            
            return self.chunks
            
        except Exception as e:
            raise ThesisChatError(f"Failed to load chunks: {str(e)}") from e

    def clear_index(self) -> None:
        """Clear all vectors from the current namespace."""
        if not self.vector_store:
            raise ThesisChatError("Vector store not initialized.")
        
        try:
            self.vector_store.delete_namespace()
            self.is_indexed = False
            print(f"Cleared namespace: {self.namespace}")
            
        except Exception as e:
            raise ThesisChatError(f"Failed to clear index: {str(e)}") from e

    def get_chunk_summary(self) -> Dict[str, Any]:
        """Get summary information about processed chunks."""
        if not self.chunks:
            return {"total_chunks": 0, "message": "No chunks processed"}
        
        # Analyze chunks
        chunk_types = {}
        thesis_parts = {}
        chapters = set()
        
        for chunk in self.chunks:
            # Count types
            chunk_types[chunk.type] = chunk_types.get(chunk.type, 0) + 1
            
            # Count thesis parts
            if chunk.thesis_part:
                thesis_parts[chunk.thesis_part] = thesis_parts.get(chunk.thesis_part, 0) + 1
            
            # Collect chapters
            if chunk.chapter:
                chapters.add(chunk.chapter)
        
        return {
            "total_chunks": len(self.chunks),
            "chunk_types": chunk_types,
            "thesis_parts": thesis_parts,
            "unique_chapters": len(chapters),
            "chapters": sorted(list(chapters)),
            "has_embeddings": any(hasattr(chunk, 'embedding') and chunk.embedding for chunk in self.chunks),
            "is_indexed": self.is_indexed
        }

    def __repr__(self) -> str:
        """String representation of ThesisChat instance."""
        return f"ThesisChat(index='{self.index_name}', namespace='{self.namespace}', chunks={len(self.chunks)}, indexed={self.is_indexed})"