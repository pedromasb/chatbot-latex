"""Tests for text utility functions."""

import pytest
from thesis_chat.utils.text_utils import TextUtils


class TestTextUtils:
    """Test cases for TextUtils class."""

    def test_strip_math(self):
        """Test math expression removal."""
        text = "This is $inline math$ and $$display math$$ text."
        result = TextUtils.strip_math(text)
        assert "$inline math$" not in result
        assert "$$display math$$" not in result
        assert "[MATH]" in result

    def test_strip_math_latex_environments(self):
        """Test LaTeX math environment removal."""
        text = "Text with \\(inline\\) and \\[display\\] math."
        result = TextUtils.strip_math(text)
        assert "\\(inline\\)" not in result
        assert "\\[display\\]" not in result
        assert "[MATH]" in result

    def test_replace_citations(self):
        """Test citation replacement."""
        text = "This work \\cite{author2023} shows that \\citep{other2022} stuff."
        result = TextUtils.replace_citations(text)
        assert "\\cite{author2023}" not in result
        assert "\\citep{other2022}" not in result
        assert "(CITATION)" in result

    def test_strip_commands_preserve_text(self):
        """Test LaTeX command removal while preserving text."""
        text = "This is \\textbf{bold} and \\textit{italic} text with \\href{url}{link}."
        result = TextUtils.strip_commands_preserve_text(text)
        assert "bold" in result
        assert "italic" in result
        assert "link" in result
        assert "\\textbf" not in result
        assert "\\href" not in result

    def test_latex_to_text_comprehensive(self):
        """Test comprehensive LaTeX to text conversion."""
        latex_text = """
        This is a test with $math$ and \\cite{ref} and \\textbf{bold text}.
        We also have~non-breaking spaces and ``quotes''.
        """
        
        result = TextUtils.latex_to_text(latex_text)
        
        # Math should be replaced
        assert "[MATH]" in result
        assert "$math$" not in result
        
        # Citations should be replaced
        assert "(CITATION)" in result
        assert "\\cite" not in result
        
        # Bold text should be preserved without command
        assert "bold text" in result
        assert "\\textbf" not in result
        
        # Special characters should be normalized
        assert "~" not in result
        assert "``" not in result

    def test_clean_text(self):
        """Test general text cleaning."""
        text = "This   has   multiple    spaces and... many dots!!!"
        result = TextUtils.clean_text(text)
        
        assert "   " not in result  # Multiple spaces removed
        assert result.count(".") <= 3  # Multiple dots normalized

    def test_extract_sentences(self):
        """Test sentence extraction."""
        text = "First sentence. Second sentence! Third sentence? Short."
        sentences = TextUtils.extract_sentences(text)
        
        # Should extract sentences (filtering very short ones)
        assert len(sentences) >= 3
        assert "First sentence" in sentences[0]

    def test_truncate_text(self):
        """Test text truncation."""
        text = "This is a long text that should be truncated at some point."
        result = TextUtils.truncate_text(text, 30)
        
        assert len(result) <= 30
        assert result.endswith("...")

    def test_truncate_text_word_boundary(self):
        """Test text truncation at word boundaries."""
        text = "This is a test sentence"
        result = TextUtils.truncate_text(text, 10)
        
        # Should break at word boundary, not in middle of word
        assert not result.endswith("te...")  # Shouldn't cut "test"

    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "This is a test document with important keywords and concepts."
        keywords = TextUtils.extract_keywords(text)
        
        assert "test" in keywords
        assert "important" in keywords
        assert "keywords" in keywords
        # Stop words should be removed
        assert "this" not in keywords
        assert "is" not in keywords

    def test_extract_keywords_min_length(self):
        """Test keyword extraction with minimum length."""
        text = "A big dog ran to the park."
        keywords = TextUtils.extract_keywords(text, min_length=4)
        
        # Only words >= 4 characters
        assert "park" in keywords
        assert "big" not in keywords  # Too short
        assert "dog" not in keywords  # Too short

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Text\twith\ttabs\nand\nnewlines\r\nand   spaces."
        result = TextUtils.normalize_whitespace(text)
        
        assert "\t" not in result
        assert "\n" not in result
        assert "\r" not in result
        assert "   " not in result  # Multiple spaces

    def test_remove_special_chars_keep_punctuation(self):
        """Test special character removal keeping punctuation."""
        text = "Text with @#$% special chars, but keep punctuation!"
        result = TextUtils.remove_special_chars(text, keep_punctuation=True)
        
        assert "@#$%" not in result
        assert "," in result  # Punctuation kept
        assert "!" in result  # Punctuation kept

    def test_remove_special_chars_no_punctuation(self):
        """Test special character removal without punctuation."""
        text = "Text with @#$% special chars, and punctuation!"
        result = TextUtils.remove_special_chars(text, keep_punctuation=False)
        
        assert "@#$%" not in result
        assert "," not in result  # Punctuation removed
        assert "!" not in result  # Punctuation removed

    def test_count_words(self):
        """Test word counting."""
        text = "This is a test sentence with exactly seven words."
        count = TextUtils.count_words(text)
        assert count == 9  # Actual count

    def test_count_words_empty(self):
        """Test word counting with empty text."""
        assert TextUtils.count_words("") == 0
        assert TextUtils.count_words(None) == 0

    def test_estimate_reading_time(self):
        """Test reading time estimation."""
        # 200 words at 200 wpm should take 1 minute
        text = " ".join(["word"] * 200)
        time_mins = TextUtils.estimate_reading_time(text, words_per_minute=200)
        assert time_mins == 1.0

    def test_create_slug(self):
        """Test slug creation."""
        text = "This is a Test Title with Special Characters!"
        slug = TextUtils.create_slug(text)
        
        assert slug == "this-is-a-test-title-with-special-characters"
        assert " " not in slug
        assert "!" not in slug

    def test_create_slug_max_length(self):
        """Test slug creation with max length."""
        text = "This is a very long title that should be truncated"
        slug = TextUtils.create_slug(text, max_length=20)
        
        assert len(slug) <= 20
        assert not slug.endswith("-")  # Should not end with hyphen

    def test_empty_text_handling(self):
        """Test handling of empty/None text inputs."""
        assert TextUtils.latex_to_text("") == ""
        assert TextUtils.latex_to_text(None) == ""
        assert TextUtils.clean_text("") == ""
        assert TextUtils.clean_text(None) == ""
        assert TextUtils.normalize_whitespace("") == ""
        assert TextUtils.create_slug("") == ""