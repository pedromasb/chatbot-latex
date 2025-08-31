"""Configuration management for thesis chat module."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration settings for ThesisChat."""
    
    # LaTeX processing settings
    chunk_size: int = 300
    overlap: int = 60
    keep_captions: bool = True
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dimension: int = 768
    
    # Pinecone settings
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    
    # Query settings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gpt-5-mini"
    top_k_retrieval: int = 50
    max_context_chunks: int = 6
    
    # Processing settings
    batch_size: int = 200
    max_text_chars: int = 4000

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create Config from dictionary."""
        # Filter only known fields
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def update(self, **kwargs) -> 'Config':
        """Create new Config with updated values."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return Config.from_dict(config_dict)

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        
        if self.top_k_retrieval <= 0:
            raise ValueError("top_k_retrieval must be positive")
        
        if self.max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.metric not in ["cosine", "dotproduct", "euclidean"]:
            raise ValueError("metric must be one of: cosine, dotproduct, euclidean")
        
        if self.cloud not in ["aws", "gcp"]:
            raise ValueError("cloud must be 'aws' or 'gcp'")

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()