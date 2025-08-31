"""Vector store implementation using Pinecone for embedding storage and retrieval."""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from .latex_processor import Chunk
from ..utils.exceptions import VectorStoreError
from ..utils.text_utils import TextUtils


class VectorStore:
    """Manages Pinecone vector database operations for thesis chunks."""

    def __init__(
        self,
        api_key: str,
        index_name: str = "thesis-chat",
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        namespace: str = "default",
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
        """
        Initialize vector store with Pinecone configuration.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Vector dimension (must match model)
            metric: Distance metric ("cosine", "dotproduct", "euclidean")
            cloud: Cloud provider ("aws" or "gcp")
            region: Cloud region
            namespace: Pinecone namespace for versioning
            model_name: SentenceTransformer model name
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.namespace = namespace
        self.model_name = model_name
        
        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=api_key)
            self.index = None
            self.model = None
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone client: {str(e)}") from e

    def create_index(self, force_recreate: bool = False) -> None:
        """
        Create Pinecone index if it doesn't exist.
        
        Args:
            force_recreate: Whether to delete and recreate existing index
            
        Raises:
            VectorStoreError: If index creation fails
        """
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                if force_recreate:
                    self.pc.delete_index(self.index_name)
                    print(f"Deleted existing index: {self.index_name}")
                else:
                    print(f"Index {self.index_name} already exists")
                    self.index = self.pc.Index(self.index_name)
                    return
            
            # Create new index
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )
            
            self.index = self.pc.Index(self.index_name)
            print(f"Created new index: {self.index_name}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create index: {str(e)}") from e

    def load_embedding_model(self) -> None:
        """Load the sentence transformer model for embeddings."""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            raise VectorStoreError(f"Failed to load model: {str(e)}") from e

    def connect_to_index(self) -> None:
        """Connect to existing Pinecone index."""
        try:
            if not self.index:
                self.index = self.pc.Index(self.index_name)
            print(f"Connected to index: {self.index_name}")
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to index: {str(e)}") from e

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        if not self.index:
            raise VectorStoreError("Index not connected. Call connect_to_index() first.")
        
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            raise VectorStoreError(f"Failed to get index stats: {str(e)}") from e

    def create_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Create embeddings for chunks and attach them.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of chunks with embeddings attached
            
        Raises:
            VectorStoreError: If embedding creation fails
        """
        if not self.model:
            self.load_embedding_model()
        
        try:
            # Create texts for embedding (include context)
            texts = []
            for chunk in chunks:
                context_path = self._build_context_path(chunk)
                embed_text = f"{context_path}\n\n{chunk.text}".strip() if context_path else chunk.text
                texts.append(embed_text)
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Attach embeddings to chunks
            for i, embedding in enumerate(embeddings):
                # Store embedding as list for JSON serialization
                chunks[i].embedding = embedding.tolist()
            
            print(f"Created embeddings for {len(chunks)} chunks")
            print(f"Embedding shape: {embeddings.shape}")
            
            return chunks
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create embeddings: {str(e)}") from e

    def upsert_chunks(self, chunks: List[Chunk], batch_size: int = 200) -> None:
        """
        Upsert chunks into Pinecone index.
        
        Args:
            chunks: List of chunks with embeddings
            batch_size: Batch size for upsert operations
            
        Raises:
            VectorStoreError: If upsert fails
        """
        if not self.index:
            raise VectorStoreError("Index not connected. Call connect_to_index() first.")
        
        try:
            buffer = []
            total_upserted = 0
            
            for chunk in chunks:
                if not hasattr(chunk, 'embedding') or not chunk.embedding:
                    raise VectorStoreError(f"Chunk {chunk.id} has no embedding")
                
                if len(chunk.embedding) != self.dimension:
                    raise VectorStoreError(
                        f"Embedding dimension mismatch: got {len(chunk.embedding)}, expected {self.dimension}"
                    )
                
                # Prepare metadata (sanitize for Pinecone)
                metadata = self._sanitize_metadata({
                    "text": self._clip_text(chunk.text),
                    "type": chunk.type,
                    "chapter_key": chunk.chapter_key,
                    "chapter": chunk.chapter,
                    "section_key": chunk.section_key,
                    "section": chunk.section,
                    "subsection_key": chunk.subsection_key,
                    "subsection": chunk.subsection,
                    "thesis_part": chunk.thesis_part,
                    "chunk_idx": chunk.chunk_idx,
                    "chunk_total": chunk.chunk_total
                })
                
                buffer.append({
                    "id": chunk.id,
                    "values": chunk.embedding,
                    "metadata": metadata
                })
                
                if len(buffer) >= batch_size:
                    self._flush_batch(buffer)
                    total_upserted += len(buffer)
                    buffer = []
                    print(f"Upserted {total_upserted} vectors...")
            
            # Flush remaining batch
            if buffer:
                self._flush_batch(buffer)
                total_upserted += len(buffer)
            
            print(f"Successfully upserted {total_upserted} vectors to namespace '{self.namespace}'")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert chunks: {str(e)}") from e

    def query(
        self,
        query_text: str,
        top_k: int = 50,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Query the vector database for similar chunks.
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            Query results from Pinecone
            
        Raises:
            VectorStoreError: If query fails
        """
        if not self.index:
            raise VectorStoreError("Index not connected. Call connect_to_index() first.")
        
        if not self.model:
            self.load_embedding_model()
        
        try:
            # Create query embedding
            query_embedding = self.model.encode([query_text], convert_to_numpy=True)[0].tolist()
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=include_metadata,
                namespace=self.namespace
            )
            
            return results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to query index: {str(e)}") from e

    def delete_namespace(self, namespace: Optional[str] = None) -> None:
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: Namespace to delete (uses instance namespace if None)
        """
        if not self.index:
            raise VectorStoreError("Index not connected. Call connect_to_index() first.")
        
        target_namespace = namespace or self.namespace
        
        try:
            self.index.delete(delete_all=True, namespace=target_namespace)
            print(f"Deleted all vectors in namespace: {target_namespace}")
        except Exception as e:
            raise VectorStoreError(f"Failed to delete namespace: {str(e)}") from e

    def save_chunks_with_embeddings(self, chunks: List[Chunk], output_path: str) -> None:
        """
        Save chunks with embeddings to JSONL file.
        
        Args:
            chunks: List of chunks with embeddings
            output_path: Path for output file
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    chunk_dict = {
                        "id": chunk.id,
                        "text": chunk.text,
                        "type": chunk.type,
                        "page": chunk.page,
                        "chapter_key": chunk.chapter_key,
                        "chapter": chunk.chapter,
                        "section_key": chunk.section_key,
                        "section": chunk.section,
                        "subsection_key": chunk.subsection_key,
                        "subsection": chunk.subsection,
                        "thesis_part": chunk.thesis_part,
                        "chunk_idx": chunk.chunk_idx,
                        "chunk_total": chunk.chunk_total,
                        "embedding": getattr(chunk, 'embedding', None)
                    }
                    f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
            
            print(f"Saved chunks with embeddings to: {output_path}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to save chunks: {str(e)}") from e

    def load_chunks_from_jsonl(self, file_path: str) -> List[Chunk]:
        """
        Load chunks with embeddings from JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of Chunk objects with embeddings
        """
        try:
            chunks = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    chunk = Chunk(
                        id=data["id"],
                        text=data["text"],
                        type=data["type"],
                        page=data["page"],
                        chapter_key=data["chapter_key"],
                        chapter=data["chapter"],
                        section_key=data["section_key"],
                        section=data["section"],
                        subsection_key=data["subsection_key"],
                        subsection=data["subsection"],
                        thesis_part=data["thesis_part"],
                        chunk_idx=data["chunk_idx"],
                        chunk_total=data["chunk_total"]
                    )
                    
                    # Add embedding if present
                    if "embedding" in data and data["embedding"]:
                        chunk.embedding = data["embedding"]
                    
                    chunks.append(chunk)
            
            print(f"Loaded {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            raise VectorStoreError(f"Failed to load chunks: {str(e)}") from e

    def _build_context_path(self, chunk: Chunk) -> str:
        """Build context path for embedding."""
        parts = []
        if chunk.chapter:
            parts.append(f"Chapter {chunk.chapter}")
        return " / ".join(parts)

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for Pinecone compatibility."""
        cleaned = {}
        for key, value in metadata.items():
            clean_value = self._clean_value(value)
            if clean_value is not None:
                cleaned[key] = clean_value
        return cleaned

    def _clean_value(self, value: Any) -> Any:
        """Clean individual metadata values."""
        if value is None:
            return None
        
        if isinstance(value, (str, int, float, bool)):
            return value
        
        if isinstance(value, list):
            return [str(x) for x in value if x is not None]
        
        # Convert other types to string
        return str(value)

    def _clip_text(self, text: str, max_chars: int = 4000) -> str:
        """Clip text to maximum length for metadata."""
        if text is None:
            return ""
        text = str(text)
        return text if len(text) <= max_chars else text[:max_chars]

    def _flush_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Flush a batch to Pinecone."""
        try:
            self.index.upsert(vectors=batch, namespace=self.namespace)
        except Exception as e:
            # Enhanced error reporting
            print("Upsert failed; inspecting batch...")
            for record in batch:
                bad_fields = {}
                for key, value in record.get("metadata", {}).items():
                    if value is None:
                        bad_fields[key] = value
                if bad_fields:
                    print(f"Found None metadata fields in record {record.get('id')}: {bad_fields}")
                    break
            raise VectorStoreError(f"Batch upsert failed: {str(e)}") from e