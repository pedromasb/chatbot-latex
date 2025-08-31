"""Query engine for semantic search and LLM-based responses with reranking."""

from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder
from openai import OpenAI
from .vector_store import VectorStore
from ..utils.exceptions import QueryError


class QueryEngine:
    """Handles semantic search, reranking, and LLM response generation."""

    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: str,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "gpt-4o-mini",
        max_context_chunks: int = 6
    ):
        """
        Initialize query engine.
        
        Args:
            vector_store: VectorStore instance for retrieval
            openai_api_key: OpenAI API key for LLM responses
            reranker_model: Cross-encoder model for reranking
            llm_model: OpenAI model name for response generation
            max_context_chunks: Maximum chunks to include in LLM context
        """
        self.vector_store = vector_store
        self.openai_api_key = openai_api_key
        self.reranker_model = reranker_model
        self.llm_model = llm_model
        self.max_context_chunks = max_context_chunks
        
        # Initialize components
        try:
            self.reranker = CrossEncoder(reranker_model)
            self.openai_client = OpenAI(api_key=openai_api_key)
            print(f"Initialized query engine with reranker: {reranker_model}")
        except Exception as e:
            raise QueryError(f"Failed to initialize query engine: {str(e)}") from e

    def query(
        self,
        question: str,
        top_k_retrieval: int = 50,
        language: str = "auto",
        include_sources: bool = True,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Perform end-to-end query: retrieval, reranking, and LLM response.
        
        Args:
            question: User's question
            top_k_retrieval: Number of chunks to retrieve initially
            language: Response language preference ("auto", "en", "es", etc.)
            include_sources: Whether to include source citations
            temperature: LLM temperature for response generation
            
        Returns:
            Dictionary with response, sources, and metadata
            
        Raises:
            QueryError: If query processing fails
        """
        try:
            # Step 1: Vector retrieval
            retrieval_results = self._retrieve_chunks(question, top_k_retrieval)
            
            if not retrieval_results["matches"]:
                return {
                    "answer": "I don't have information to answer this question based on the available documents.",
                    "sources": [],
                    "query": question,
                    "chunks_retrieved": 0,
                    "chunks_used": 0
                }
            
            # Step 2: Rerank results
            reranked_chunks = self._rerank_results(question, retrieval_results["matches"])
            
            # Step 3: Select top chunks for context
            context_chunks = reranked_chunks[:self.max_context_chunks]
            
            # Step 4: Generate LLM response
            response = self._generate_response(question, context_chunks, language, temperature)
            
            # Step 5: Prepare sources
            sources = self._prepare_sources(context_chunks) if include_sources else []
            
            return {
                "answer": response,
                "sources": sources,
                "query": question,
                "chunks_retrieved": len(retrieval_results["matches"]),
                "chunks_used": len(context_chunks)
            }
            
        except Exception as e:
            raise QueryError(f"Query processing failed: {str(e)}") from e

    def _retrieve_chunks(self, query: str, top_k: int) -> Dict[str, Any]:
        """Retrieve chunks using vector similarity search."""
        try:
            return self.vector_store.query(
                query_text=query,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            raise QueryError(f"Vector retrieval failed: {str(e)}") from e

    def _rerank_results(self, query: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank retrieval results using cross-encoder."""
        try:
            if not matches:
                return []
            
            # Prepare query-document pairs for reranking
            pairs = []
            for match in matches:
                text = match["metadata"].get("text", "")
                pairs.append((query, text))
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Sort matches by reranking scores
            scored_matches = list(zip(matches, scores))
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Return reranked matches with scores
            reranked = []
            for match, score in scored_matches:
                match["rerank_score"] = float(score)
                reranked.append(match)
            
            return reranked
            
        except Exception as e:
            raise QueryError(f"Reranking failed: {str(e)}") from e

    def _generate_response(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        language: str,
        temperature: float
    ) -> str:
        """Generate LLM response using context chunks."""
        try:
            # Build context from chunks
            context_blocks = []
            for i, chunk in enumerate(context_chunks, 1):
                metadata = chunk.get("metadata", {})
                
                # Build path information
                path_parts = []
                if metadata.get("chapter_key") or metadata.get("chapter"):
                    chapter_info = f"Ch.{metadata.get('chapter_key', '')}: {metadata.get('chapter', '')}".strip(": ")
                    path_parts.append(chapter_info)
                
                if metadata.get("section_key") or metadata.get("section"):
                    section_info = f"S.{metadata.get('section_key', '')}: {metadata.get('section', '')}".strip(": ")
                    path_parts.append(section_info)
                
                if metadata.get("subsection_key") or metadata.get("subsection"):
                    subsection_info = f"SS.{metadata.get('subsection_key', '')}: {metadata.get('subsection', '')}".strip(": ")
                    path_parts.append(subsection_info)
                
                if metadata.get("type"):
                    path_parts.append(f"Text type: {metadata.get('type')}")
                
                path = " | ".join([p for p in path_parts if p and not p.endswith(': ')]).strip()
                
                # Build context block
                header = f"[[{i}]] {path}" if path else f"[[{i}]]"
                text = self._trim_text(metadata.get("text", ""), 1200)
                context_blocks.append(f"{header}\n{text}")
            
            context_blob = "\n\n---\n\n".join(context_blocks)
            
            # Build system message
            system_msg = self._build_system_message(language)
            
            # Build user message
            user_msg = f"Question: {question}\n\nContext:\n{context_blob}\n\n"
            
            # Generate response
            completion = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=temperature
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise QueryError(f"LLM response generation failed: {str(e)}") from e

    def _build_system_message(self, language: str) -> str:
        """Build system message based on language preference."""
        base_msg = (
            "You answer questions using ONLY the provided context blocks. "
            "You answer questions in an extended way, don't be concise. "
            "Cite the blocks you use by their bracket number like [1], [2]. "
            "If the answer is not contained in the context, say you don't know."
        )
        
        if language.lower() in ["es", "spanish", "español"]:
            return (
                "Respondes preguntas usando ÚNICAMENTE los bloques de contexto proporcionados. "
                "Responde las preguntas de manera extendida, no seas conciso. "
                "Cita los bloques que uses por su número entre corchetes como [1], [2]. "
                "Si la respuesta no está contenida en el contexto, di que no lo sabes."
            )
        else:
            return base_msg

    def _prepare_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information for response."""
        sources = []
        
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            
            # Build readable source reference
            source_parts = []
            if metadata.get("chapter_key") or metadata.get("chapter"):
                source_parts.append(f"Ch.{metadata.get('chapter_key', '')}: {metadata.get('chapter', '')}".strip(": "))
            
            if metadata.get("section_key") or metadata.get("section"):
                source_parts.append(f"S.{metadata.get('section_key', '')}: {metadata.get('section', '')}".strip(": "))
            
            if metadata.get("subsection_key") or metadata.get("subsection"):
                source_parts.append(f"SS.{metadata.get('subsection_key', '')}: {metadata.get('subsection', '')}".strip(": "))
            
            if metadata.get("type"):
                source_parts.append(f"Text type: {metadata.get('type')}")
            
            source_ref = " | ".join([p for p in source_parts if p and not p.endswith(': ')]).strip()
            
            sources.append({
                "index": i,
                "reference": source_ref or "(no path)",
                "similarity_score": chunk.get("score", 0.0),
                "rerank_score": chunk.get("rerank_score", 0.0),
                "chapter": metadata.get("chapter"),
                "section": metadata.get("section"),
                "subsection": metadata.get("subsection"),
                "thesis_part": metadata.get("thesis_part"),
                "text_preview": self._trim_text(metadata.get("text", ""), 200)
            })
        
        return sources

    def _trim_text(self, text: str, max_chars: int) -> str:
        """Trim text to maximum length."""
        if not text:
            return ""
        text = str(text).strip()
        return text if len(text) <= max_chars else text[:max_chars] + " …"

    def search_only(
        self,
        query: str,
        top_k_retrieval: int = 20,
        top_k_rerank: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform search without LLM response generation.
        
        Args:
            query: Search query
            top_k_retrieval: Number of chunks to retrieve initially
            top_k_rerank: Number of chunks to return after reranking
            
        Returns:
            List of reranked search results
        """
        try:
            # Retrieve chunks
            retrieval_results = self._retrieve_chunks(query, top_k_retrieval)
            
            if not retrieval_results["matches"]:
                return []
            
            # Rerank and return top results
            reranked_chunks = self._rerank_results(query, retrieval_results["matches"])
            
            return reranked_chunks[:top_k_rerank]
            
        except Exception as e:
            raise QueryError(f"Search failed: {str(e)}") from e

    def get_chunk_context(self, chunk_metadata: Dict[str, Any]) -> str:
        """
        Build readable context string for a chunk.
        
        Args:
            chunk_metadata: Chunk metadata dictionary
            
        Returns:
            Formatted context string
        """
        parts = []
        
        if chunk_metadata.get("chapter"):
            parts.append(f"Chapter: {chunk_metadata['chapter']}")
        
        if chunk_metadata.get("section"):
            parts.append(f"Section: {chunk_metadata['section']}")
        
        if chunk_metadata.get("subsection"):
            parts.append(f"Subsection: {chunk_metadata['subsection']}")
        
        if chunk_metadata.get("thesis_part"):
            parts.append(f"Part: {chunk_metadata['thesis_part']}")
        
        return " | ".join(parts) if parts else "No context available"