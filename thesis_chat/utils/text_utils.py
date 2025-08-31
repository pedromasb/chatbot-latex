"""Text processing utilities for LaTeX and general text handling."""

import re
from typing import List, Dict, Any, Optional


class TextUtils:
    """Utility class for text processing operations."""

    @staticmethod
    def strip_math(text: str) -> str:
        """Remove math expressions from LaTeX text."""
        # Display math
        text = re.sub(r'\$\$.*?\$\$', ' [MATH] ', text, flags=re.DOTALL)
        # Inline math
        text = re.sub(r'\$[^$]*\$', ' [MATH] ', text)
        # LaTeX math environments
        text = re.sub(r'\\\(.*?\\\)', ' [MATH] ', text, flags=re.DOTALL)
        text = re.sub(r'\\\[.*?\\\]', ' [MATH] ', text, flags=re.DOTALL)
        return text

    @staticmethod
    def replace_citations(text: str) -> str:
        """Replace LaTeX citations with placeholder."""
        return re.sub(r'\\cite[a-zA-Z]*\*?\{[^}]*\}', ' (CITATION) ', text)

    @staticmethod
    def strip_commands_preserve_text(text: str) -> str:
        """Remove LaTeX commands while preserving text content."""
        # Text formatting commands - preserve content
        text = re.sub(r'\\(textbf|textit|emph|texttt|textsc)\*?\{([^}]*)\}', r'\2', text)
        
        # Links - preserve link text
        text = re.sub(r'\\href\{[^}]*\}\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\url\{([^}]*)\}', r'\1', text)
        
        # References and labels - remove
        text = re.sub(r'\\(label|ref|eqref|autoref|nameref)\{[^}]*\}', ' ', text)
        
        # Footnotes - remove
        text = re.sub(r'\\footnote\{[^}]*\}', ' ', text)
        
        # Generic commands - remove
        text = re.sub(r'\\[a-zA-Z@]+\*?(\[[^\]]*\])?(\{[^}]*\})?', ' ', text)
        
        return text

    @staticmethod
    def latex_to_text(text: str) -> str:
        """Convert LaTeX text to plain text."""
        if not text:
            return ""
        
        # Remove math
        text = TextUtils.strip_math(text)
        
        # Replace citations
        text = TextUtils.replace_citations(text)
        
        # Remove commands while preserving text
        text = TextUtils.strip_commands_preserve_text(text)
        
        # Replace special characters
        text = re.sub(r'~', ' ', text)  # Non-breaking space
        text = re.sub(r'---', '—', text)  # Em dash
        text = re.sub(r'--', '–', text)  # En dash
        text = re.sub(r'``', '"', text)  # Opening quotes
        text = re.sub(r"''", '"', text)  # Closing quotes
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def clean_text(text: str) -> str:
        """General text cleaning operations."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Clean up brackets and parentheses
        text = re.sub(r'\[\s*\]', '', text)
        text = re.sub(r'\(\s*\)', '', text)
        
        return text.strip()

    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text."""
        if not text:
            return []
        
        # Simple sentence splitting (can be improved with more sophisticated methods)
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length with suffix."""
        if not text or len(text) <= max_length:
            return text
        
        # Try to break at word boundary
        truncated = text[:max_length - len(suffix)]
        
        # Find last space to avoid cutting words
        last_space = truncated.rfind(' ')
        if last_space > 0 and last_space > max_length * 0.8:
            truncated = truncated[:last_space]
        
        return truncated + suffix

    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """Extract potential keywords from text."""
        if not text:
            return []
        
        # Convert to lowercase and split
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter by length
        keywords = [word for word in words if len(word) >= min_length]
        
        # Remove common stop words (basic set)
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 
            'may', 'say', 'she', 'use', 'way', 'will', 'with', 'this', 'that', 
            'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 
            'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 
            'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'
        }
        
        keywords = [word for word in keywords if word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        if not text:
            return ""
        
        # Replace tabs and newlines with spaces
        text = re.sub(r'[\t\n\r]+', ' ', text)
        
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()

    @staticmethod
    def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters from text."""
        if not text:
            return ""
        
        if keep_punctuation:
            # Keep letters, numbers, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,;:!?()-]', '', text)
        else:
            # Keep only letters, numbers, and spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        return TextUtils.normalize_whitespace(text)

    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text."""
        if not text:
            return 0
        
        words = re.findall(r'\b\w+\b', text)
        return len(words)

    @staticmethod
    def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
        """Estimate reading time in minutes."""
        word_count = TextUtils.count_words(text)
        return word_count / words_per_minute

    @staticmethod
    def create_slug(text: str, max_length: int = 50) -> str:
        """Create a URL-friendly slug from text."""
        if not text:
            return ""
        
        # Convert to lowercase
        slug = text.lower()
        
        # Replace spaces and special chars with hyphens
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        
        # Truncate if necessary
        if len(slug) > max_length:
            slug = slug[:max_length].rstrip('-')
        
        return slug