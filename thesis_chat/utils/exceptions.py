"""Custom exceptions for the thesis chat module."""


class ThesisChatError(Exception):
    """Base exception for thesis chat module."""
    pass


class LaTeXProcessingError(ThesisChatError):
    """Raised when LaTeX processing fails."""
    pass


class VectorStoreError(ThesisChatError):
    """Raised when vector store operations fail."""
    pass


class QueryError(ThesisChatError):
    """Raised when query processing fails."""
    pass


class ConfigurationError(ThesisChatError):
    """Raised when configuration is invalid."""
    pass


class APIError(ThesisChatError):
    """Raised when external API calls fail."""
    pass


class FileNotFoundError(ThesisChatError):
    """Raised when required files are not found."""
    pass


class ValidationError(ThesisChatError):
    """Raised when data validation fails."""
    pass