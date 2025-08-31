# ThesisChat: LaTeX Document Processing and Conversational AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](#testing)

**ThesisChat** is a Python module that converts LaTeX thesis documents into searchable, conversational interfaces using retrieval-augmented generation (RAG). It processes LaTeX files, creates semantic embeddings, stores them in Pinecone vector database, and enables intelligent querying using state-of-the-art language models.

## 🚀 Key Features

- **LaTeX Processing**: Automatically parses LaTeX documents, resolving includes and extracting hierarchical structure
- **Semantic Search**: Creates multilingual embeddings using sentence transformers for accurate content retrieval
- **Vector Storage**: Integrates with Pinecone for scalable, high-performance vector search
- **Smart Reranking**: Uses cross-encoder models to improve search result relevance
- **LLM Integration**: Generates contextual responses using OpenAI's GPT models
- **Multilingual Support**: Handles queries and responses in multiple languages
- **Configurable**: Extensive configuration options for different use cases
- **Extensible**: Modular design allows easy customization and integration

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - [Pinecone](https://www.pinecone.io/) (vector database)
  - [OpenAI](https://openai.com/) (language model)

### Install from Source

```bash
git clone https://github.com/your-repo/thesis-chat.git
cd thesis-chat
pip install -e .
```

### Dependencies

The module automatically installs all required dependencies:

- `sentence-transformers` - For creating embeddings
- `pinecone-client` - For vector database operations
- `openai` - For LLM responses
- `numpy` - For numerical operations
- `pathlib` - For file path handling

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from thesis_chat import ThesisChat

# Initialize with your API keys
chat = ThesisChat(
    pinecone_api_key="your-pinecone-key",
    openai_api_key="your-openai-key",
    index_name="my-thesis"
)

# Setup the system
chat.setup()

# Process your LaTeX thesis
chat.process_and_index_latex("path/to/your/thesis.tex")

# Ask questions!
response = chat.query("What are the main contributions of this thesis?")
print(response["answer"])
```

### Advanced Configuration

```python
from thesis_chat import ThesisChat, Config

# Custom configuration
config = Config(
    chunk_size=400,           # Larger chunks for more context
    overlap=80,               # More overlap between chunks
    keep_captions=True,       # Include figure/table captions
    max_context_chunks=8,     # Use more context for responses
    llm_model="gpt-4"        # Use GPT-4 for better responses
)

chat = ThesisChat(
    pinecone_api_key="your-key",
    openai_api_key="your-key",
    config=config
)
```

## 📖 Documentation

### Core Components

#### LaTeXProcessor
Handles LaTeX document parsing and chunking:

```python
from thesis_chat import LaTeXProcessor

processor = LaTeXProcessor(chunk_size=300, overlap=60)
chunks = processor.process_latex_file("thesis.tex")
```

#### VectorStore
Manages Pinecone vector database operations:

```python
from thesis_chat import VectorStore

store = VectorStore(api_key="your-key", index_name="thesis")
store.create_index()
store.upsert_chunks(chunks)
```

#### QueryEngine
Handles search and response generation:

```python
from thesis_chat import QueryEngine

engine = QueryEngine(vector_store, openai_api_key="your-key")
results = engine.query("What methodology was used?")
```

### Configuration Options

The `Config` class provides extensive customization:

```python
config = Config(
    # LaTeX Processing
    chunk_size=300,           # Words per chunk
    overlap=60,               # Word overlap between chunks
    keep_captions=True,       # Include figure/table captions
    
    # Embedding Settings  
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    embedding_dimension=768,
    
    # Pinecone Settings
    metric="cosine",          # Distance metric
    cloud="aws",              # Cloud provider
    region="us-east-1",       # Region
    
    # Query Settings
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    llm_model="gpt-4o-mini", # Language model
    top_k_retrieval=50,       # Initial retrieval count
    max_context_chunks=6,     # Context chunks for LLM
    
    # Performance
    batch_size=200,           # Batch size for operations
    max_text_chars=4000       # Max text length for metadata
)
```

### API Reference

#### ThesisChat Class

**Main Methods:**

- `setup(force_recreate_index=False)`: Initialize system components
- `process_and_index_latex(latex_file_path)`: End-to-end document processing
- `query(question, language="auto", temperature=0.7)`: Query the document
- `search(query, top_k=10)`: Search without LLM response
- `get_chunk_summary()`: Get document statistics
- `save_chunks(output_path, include_embeddings=True)`: Save processed data
- `load_chunks(file_path)`: Load processed data

**Query Response Format:**

```python
{
    "answer": "Generated response text",
    "sources": [
        {
            "index": 1,
            "reference": "Ch.1: Introduction | S.1.1: Background",
            "similarity_score": 0.85,
            "rerank_score": 0.92,
            "chapter": "Introduction",
            "text_preview": "Preview of source text..."
        }
    ],
    "query": "Original question",
    "chunks_retrieved": 50,
    "chunks_used": 6
}
```

## 🔧 Advanced Usage

### Custom Processing Pipeline

```python
from thesis_chat.core import LaTeXProcessor, VectorStore, QueryEngine

# Manual pipeline for fine-grained control
processor = LaTeXProcessor(chunk_size=250, overlap=50)
chunks = processor.process_latex_file("thesis.tex")

vector_store = VectorStore(api_key="key", index_name="custom")
vector_store.create_index()
vector_store.load_embedding_model()

chunks_with_embeddings = vector_store.create_embeddings(chunks)
vector_store.upsert_chunks(chunks_with_embeddings)

query_engine = QueryEngine(vector_store, openai_api_key="key")
response = query_engine.query("Custom query")
```

### Batch Processing Multiple Documents

```python
documents = ["thesis1.tex", "thesis2.tex", "paper1.tex"]

for i, doc_path in enumerate(documents):
    chat = ThesisChat(
        pinecone_api_key="key",
        openai_api_key="key",
        index_name="multi-doc",
        namespace=f"doc_{i}"
    )
    chat.setup()
    chat.process_and_index_latex(doc_path)
```

### Multilingual Queries

```python
# English query
response_en = chat.query("What are the main findings?", language="en")

# Spanish query
response_es = chat.query("¿Cuáles son los principales hallazgos?", language="es")

# Auto-detect language
response_auto = chat.query("Quelles sont les conclusions?", language="auto")
```

### Performance Optimization

```python
# Use caching for repeated queries
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(question):
    return chat.query(question)

# Parallel processing for large documents
config = Config(
    batch_size=500,           # Larger batches
    embedding_model="all-mpnet-base-v2"  # Faster model
)
```

## 📊 Examples and Notebooks

The `examples/` directory contains comprehensive Jupyter notebooks:

- **`01_basic_usage.ipynb`**: Complete beginner tutorial
- **`02_advanced_usage.ipynb`**: Advanced features and customization

Run the notebooks to see the module in action:

```bash
jupyter notebook examples/01_basic_usage.ipynb
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_latex_processor.py -v
python -m pytest tests/test_vector_store.py -v
python -m pytest tests/test_query_engine.py -v

# Generate coverage report
python -m pytest tests/ --cov=thesis_chat --cov-report=html
```

### Test Coverage

The test suite covers:
- LaTeX processing and parsing
- Vector store operations
- Query engine functionality
- Configuration validation
- Text utility functions
- Error handling and edge cases

## 🗂️ Project Structure

```
thesis_chat/
├── __init__.py              # Main module interface
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── latex_processor.py   # LaTeX document processing
│   ├── vector_store.py      # Pinecone integration
│   ├── query_engine.py      # Search and LLM integration
│   └── thesis_chat.py       # Main interface class
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── exceptions.py       # Custom exceptions
│   └── text_utils.py       # Text processing utilities
examples/                    # Example notebooks
├── 01_basic_usage.ipynb    # Basic tutorial
└── 02_advanced_usage.ipynb # Advanced features
tests/                       # Test suite
├── __init__.py
├── test_latex_processor.py
├── test_vector_store.py
├── test_query_engine.py
├── test_config.py
└── test_text_utils.py
docs/                        # Documentation
setup.py                     # Package setup
requirements.txt             # Dependencies
README.md                    # This file
CLAUDE.md                    # AI assistant guidance
```

## 🔒 Environment Variables

For security, set your API keys as environment variables:

```bash
export PINECONE_API_KEY="your-pinecone-key"
export OPENAI_API_KEY="your-openai-key"
```

Then use them in Python:

```python
import os
from thesis_chat import ThesisChat

chat = ThesisChat(
    pinecone_api_key=os.getenv('PINECONE_API_KEY'),
    openai_api_key=os.getenv('OPENAI_API_KEY')
)
```

## 🐛 Troubleshooting

### Common Issues

**1. LaTeX Processing Errors**
```python
# Check file exists and is readable
import os
if not os.path.exists("thesis.tex"):
    print("LaTeX file not found")

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

**2. Pinecone Connection Issues**
```python
# Verify API key and region
try:
    chat.setup()
except Exception as e:
    print(f"Setup error: {e}")
    # Check API key, region, and network connection
```

**3. Memory Issues with Large Documents**
```python
# Use smaller chunks and batches
config = Config(
    chunk_size=200,     # Smaller chunks
    batch_size=50,      # Smaller batches
)
```

**4. Slow Performance**
```python
# Optimize for speed
config = Config(
    embedding_model="all-MiniLM-L6-v2",  # Faster model
    llm_model="gpt-3.5-turbo",           # Faster LLM
    top_k_retrieval=20                    # Fewer candidates
)
```

### Error Messages

- `LaTeXProcessingError`: Issues with LaTeX file parsing
- `VectorStoreError`: Pinecone connection or operation errors
- `QueryError`: Problems with search or LLM responses
- `ConfigurationError`: Invalid configuration parameters

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-repo/thesis-chat.git
cd thesis-chat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install in development mode
pip install -e .[dev]

# Run tests
python -m pytest tests/ -v
```

### Code Style

We use `black` for code formatting and `flake8` for linting:

```bash
black thesis_chat/ tests/
flake8 thesis_chat/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of excellent open-source libraries:
  - [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for embeddings
  - [Pinecone](https://www.pinecone.io/) for vector database
  - [OpenAI](https://openai.com/) for language models
- Inspired by the need for better academic document interaction tools
- Thanks to the research community for LaTeX standardization

## 📬 Support

- **Documentation**: Check the [examples](examples/) and docstrings
- **Issues**: Report bugs on [GitHub Issues](https://github.com/your-repo/thesis-chat/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/your-repo/thesis-chat/discussions)
- **Email**: contact@thesischat.com

---

**Made with ❤️ for researchers, students, and academics who want to make their documents more accessible and interactive.**