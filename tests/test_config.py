"""Tests for configuration management."""

import pytest
from thesis_chat.utils.config import Config
from thesis_chat.utils.exceptions import ConfigurationError


class TestConfig:
    """Test cases for Config class."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = Config()
        
        assert config.chunk_size == 300
        assert config.overlap == 60
        assert config.keep_captions is True
        assert config.embedding_model == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        assert config.embedding_dimension == 768
        assert config.metric == "cosine"
        assert config.cloud == "aws"
        assert config.region == "us-east-1"

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = Config(
            chunk_size=500,
            overlap=100,
            keep_captions=False,
            embedding_dimension=384
        )
        
        assert config.chunk_size == 500
        assert config.overlap == 100
        assert config.keep_captions is False
        assert config.embedding_dimension == 384

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "chunk_size": 400,
            "overlap": 80,
            "metric": "dotproduct",
            "unknown_field": "should_be_ignored"
        }
        
        config = Config.from_dict(config_dict)
        
        assert config.chunk_size == 400
        assert config.overlap == 80
        assert config.metric == "dotproduct"
        # Unknown fields should be ignored, defaults should be preserved
        assert config.keep_captions is True  # Default value

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config(chunk_size=400, overlap=80)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["chunk_size"] == 400
        assert config_dict["overlap"] == 80
        assert "keep_captions" in config_dict

    def test_update(self):
        """Test updating config with new values."""
        config = Config()
        updated_config = config.update(chunk_size=400, overlap=80)
        
        # Original config should be unchanged
        assert config.chunk_size == 300
        assert config.overlap == 60
        
        # Updated config should have new values
        assert updated_config.chunk_size == 400
        assert updated_config.overlap == 80

    def test_validation_chunk_size_positive(self):
        """Test validation of chunk_size must be positive."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            Config(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            Config(chunk_size=-10)

    def test_validation_overlap_non_negative(self):
        """Test validation of overlap must be non-negative."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            Config(overlap=-1)

    def test_validation_overlap_less_than_chunk_size(self):
        """Test validation that overlap must be less than chunk_size."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            Config(chunk_size=100, overlap=100)
        
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            Config(chunk_size=100, overlap=150)

    def test_validation_embedding_dimension_positive(self):
        """Test validation of embedding_dimension must be positive."""
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            Config(embedding_dimension=0)
        
        with pytest.raises(ValueError, match="embedding_dimension must be positive"):
            Config(embedding_dimension=-768)

    def test_validation_top_k_retrieval_positive(self):
        """Test validation of top_k_retrieval must be positive."""
        with pytest.raises(ValueError, match="top_k_retrieval must be positive"):
            Config(top_k_retrieval=0)

    def test_validation_max_context_chunks_positive(self):
        """Test validation of max_context_chunks must be positive."""
        with pytest.raises(ValueError, match="max_context_chunks must be positive"):
            Config(max_context_chunks=0)

    def test_validation_batch_size_positive(self):
        """Test validation of batch_size must be positive."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            Config(batch_size=0)

    def test_validation_metric_valid_values(self):
        """Test validation of metric must be valid."""
        # Valid metrics should work
        Config(metric="cosine")
        Config(metric="dotproduct")
        Config(metric="euclidean")
        
        # Invalid metric should fail
        with pytest.raises(ValueError, match="metric must be one of"):
            Config(metric="invalid")

    def test_validation_cloud_valid_values(self):
        """Test validation of cloud must be valid."""
        # Valid clouds should work
        Config(cloud="aws")
        Config(cloud="gcp")
        
        # Invalid cloud should fail
        with pytest.raises(ValueError, match="cloud must be 'aws' or 'gcp'"):
            Config(cloud="azure")

    def test_validation_edge_cases(self):
        """Test validation edge cases."""
        # Valid edge case: overlap = chunk_size - 1
        Config(chunk_size=100, overlap=99)  # Should work
        
        # Minimum valid values
        Config(
            chunk_size=1,
            overlap=0,
            embedding_dimension=1,
            top_k_retrieval=1,
            max_context_chunks=1,
            batch_size=1
        )

    def test_manual_validation_call(self):
        """Test calling validate() manually."""
        config = Config()
        config.validate()  # Should not raise
        
        # Manually set invalid value and test validation
        config.chunk_size = -1
        with pytest.raises(ValueError):
            config.validate()

    def test_config_immutability_through_update(self):
        """Test that original config is not modified by update."""
        original = Config(chunk_size=300)
        updated = original.update(chunk_size=500)
        
        # Original should be unchanged
        assert original.chunk_size == 300
        assert updated.chunk_size == 500
        
        # Should be different objects
        assert original is not updated

    def test_all_fields_in_to_dict(self):
        """Test that to_dict includes all configuration fields."""
        config = Config()
        config_dict = config.to_dict()
        
        # Check that all expected fields are present
        expected_fields = {
            'chunk_size', 'overlap', 'keep_captions',
            'embedding_model', 'embedding_dimension',
            'metric', 'cloud', 'region',
            'reranker_model', 'llm_model', 'top_k_retrieval', 'max_context_chunks',
            'batch_size', 'max_text_chars'
        }
        
        assert set(config_dict.keys()) == expected_fields