# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a chatbot-LaTeX project that implements a thesis document processing and question-answering system using vector embeddings and retrieval-augmented generation (RAG). The system processes LaTeX thesis documents, chunks them into semantic segments, creates embeddings, and enables conversational queries about the thesis content.

## Repository Structure

```
.
├── README.md           # Basic project description
└── nbk/               # Jupyter notebooks directory
    ├── pinecone.ipynb            # Pinecone vector database integration
    └── thesis_from_latex.ipynb  # LaTeX thesis processing pipeline
```

## Core Components

### 1. LaTeX Document Processing (`thesis_from_latex.ipynb`)
- **Purpose**: Converts LaTeX thesis documents into structured, searchable chunks
- **Key Features**:
  - Resolves `\input` and `\include` commands to build complete document
  - Extracts hierarchical structure (chapters, sections, subsections)
  - Handles special cases like starred chapters (e.g., "Data and Software Availability")
  - Converts LaTeX to plain text while preserving semantic structure
  - Creates sliding window chunks with configurable overlap
- **Output**: `chunks.jsonl` with structured metadata

### 2. Vector Database Integration (`pinecone.ipynb`)
- **Purpose**: Implements semantic search using Pinecone vector database
- **Key Features**:
  - Creates embeddings using multilingual sentence transformers
  - Manages Pinecone index creation and data upsert
  - Implements retrieval with cross-encoder reranking
  - Provides LLM-based question answering with source citations
- **Models Used**:
  - Embeddings: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
  - Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Key Configuration

### LaTeX Processing Settings
- **Chunk Size**: 300 words (configurable via `CHUNK_SIZE`)
- **Overlap**: 60 words (configurable via `OVERLAP`)
- **Citation Handling**: Citations replaced with `(CITATION)` placeholders
- **Math Handling**: Math expressions replaced with `[MATH]` placeholders

### Vector Database Settings
- **Index Name**: "thesis-chat"
- **Dimension**: 768 (matches multilingual mpnet model)
- **Metric**: Cosine similarity
- **Namespace**: "multiling" for versioning
- **Batch Size**: 200 for upsert operations

## Development Workflow

### Processing a New Thesis
1. Place LaTeX files in accessible directory
2. Update `ROOT_TEX` path in `thesis_from_latex.ipynb`
3. Run notebook to generate `chunks.jsonl`
4. Update embedding configuration if needed
5. Run `pinecone.ipynb` to create embeddings and populate vector database

### Querying the System
1. Ensure Pinecone index is populated
2. Configure API keys (Pinecone, OpenAI)
3. Use query interface in notebook for semantic search
4. System returns ranked results with source citations

## Data Pipeline

```
LaTeX Source → Document Parsing → Structural Analysis → Text Extraction → 
Chunking → Embedding Generation → Vector Database → Semantic Search → 
Reranking → LLM Response Generation
```

## Dependencies

The notebooks rely on several key libraries:
- `sentence-transformers` - For multilingual embeddings
- `pinecone` - Vector database integration
- `openai` - LLM API for response generation
- Standard data processing: `pandas`, `numpy`, `json`

## Thesis Structure Handling

The system recognizes canonical thesis structure:
- Front matter (Abstract/Resumen)
- Numbered chapters (1-6)
- Hierarchical sections (X.Y) and subsections (X.Y.Z)
- Special chapters like "Data and Software Availability"
- Content classification: Introduction, Methods/Results, Conclusions, etc.

## Search and Retrieval Features

- **Multilingual Support**: Handles queries in multiple languages
- **Semantic Search**: Uses dense vector representations for content matching
- **Reranking**: Cross-encoder improves initial retrieval results
- **Source Attribution**: Results include chapter/section references
- **Context-Aware Responses**: LLM responses cite specific document sections