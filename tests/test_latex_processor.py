"""Tests for LaTeX processor functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from thesis_chat.core.latex_processor import LaTeXProcessor, Chunk, Event, Segment
from thesis_chat.utils.exceptions import LaTeXProcessingError


class TestLaTeXProcessor:
    """Test cases for LaTeXProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = LaTeXProcessor(chunk_size=50, overlap=10, keep_captions=True)

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.chunk_size == 50
        assert self.processor.overlap == 10
        assert self.processor.keep_captions is True

    def test_custom_initialization(self):
        """Test processor with custom parameters."""
        processor = LaTeXProcessor(chunk_size=100, overlap=20, keep_captions=False)
        assert processor.chunk_size == 100
        assert processor.overlap == 20
        assert processor.keep_captions is False

    def test_sliding_windows_basic(self):
        """Test sliding window creation."""
        text = "This is a test sentence with exactly ten words here."
        windows = self.processor._sliding_windows(text)
        
        # Should create one window since text is short
        assert len(windows) == 1
        assert windows[0] == text

    def test_sliding_windows_long_text(self):
        """Test sliding window with longer text."""
        words = ["word"] * 100
        text = " ".join(words)
        windows = self.processor._sliding_windows(text)
        
        # Should create multiple windows
        assert len(windows) > 1
        
        # Check overlap
        if len(windows) > 1:
            window1_words = windows[0].split()
            window2_words = windows[1].split()
            assert len(window1_words) == 50  # chunk_size
            # Second window should start with some overlap
            overlap_start = 50 - 10  # chunk_size - overlap
            assert window2_words[0] == window1_words[overlap_start]

    def test_canonical_title(self):
        """Test canonical title mapping."""
        assert self.processor._canonical_title("1", "Introduction") == "General Introduction"
        assert self.processor._canonical_title("999", "Unknown") == "Unknown"

    def test_is_dsa_title(self):
        """Test DSA title detection."""
        assert self.processor._is_dsa_title("Data and Software Availability")
        assert self.processor._is_dsa_title("Data & Software Availability")
        assert self.processor._is_dsa_title("data and software availability")
        assert not self.processor._is_dsa_title("Regular Chapter Title")

    def test_thesis_part_from_ctx(self):
        """Test thesis part classification."""
        assert self.processor._thesis_part_from_ctx("1", None) == "Introduction"
        assert self.processor._thesis_part_from_ctx("2", None) == "Methods/Results"
        assert self.processor._thesis_part_from_ctx("6", None) == "Conclusions"
        assert self.processor._thesis_part_from_ctx("DSA", None) == "Data & Software"
        assert self.processor._thesis_part_from_ctx("FRONT", None) == "Front Matter"
        assert self.processor._thesis_part_from_ctx("2", "Conclusions") == "Conclusions"

    def test_extract_document_body(self):
        """Test document body extraction."""
        latex_text = """
        \\documentclass{article}
        \\begin{document}
        This is the main content.
        \\end{document}
        """
        body = self.processor._extract_document_body(latex_text)
        assert "This is the main content." in body
        assert "\\documentclass" not in body

    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = """
        % This is a comment
        Regular text here.
        \\bibliography{refs}
        \\begin{figure}
        \\caption{Test caption}
        \\end{figure}
        """
        
        processed = self.processor._preprocess_text(text)
        
        # Comments should be removed
        assert "% This is a comment" not in processed
        
        # Regular text should remain
        assert "Regular text here." in processed
        
        # Bibliography should be removed
        assert "\\bibliography{refs}" not in processed
        
        # Captions should be preserved (if keep_captions=True)
        if self.processor.keep_captions:
            assert "Test caption" in processed

    def test_keep_captions_only(self):
        """Test caption extraction."""
        text = """
        \\begin{figure}
        \\includegraphics{image.png}
        \\caption{This is a figure caption.}
        \\label{fig:test}
        \\end{figure}
        """
        
        result = self.processor._keep_captions_only(text)
        assert "[[CAPTION]] This is a figure caption." in result
        assert "includegraphics" not in result

    def test_find_headings(self):
        """Test heading detection."""
        text = """
        \\chapter{Introduction}
        Some text here.
        \\chapter*{Data and Software Availability}
        More text.
        """
        
        events = self.processor._find_headings(text, self.processor.ch_re, 'chapter')
        
        assert len(events) == 2
        assert events[0].content == "Introduction"
        assert events[0].starred is False
        assert events[1].content == "Data and Software Availability"
        assert events[1].starred is True

    def test_save_and_load_chunks(self):
        """Test saving chunks to JSONL."""
        chunks = [
            Chunk(
                id="test-1",
                text="Test chunk 1",
                type="body",
                page=1,
                chapter_key="1",
                chapter="Test Chapter",
                section_key="1.1",
                section="Test Section",
                subsection_key=None,
                subsection=None,
                thesis_part="Introduction",
                chunk_idx=0,
                chunk_total=1
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            self.processor.save_chunks_to_jsonl(chunks, temp_path)
            
            # Verify file exists and has content
            assert os.path.exists(temp_path)
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "test-1" in content
                assert "Test chunk 1" in content
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_process_simple_latex_file(self):
        """Test processing a simple LaTeX file."""
        latex_content = """
        \\documentclass{article}
        \\begin{document}
        \\chapter{Introduction}
        This is the introduction with some content that should be processed into chunks.
        \\section{Background}
        This is the background section with more content to process.
        \\end{document}
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
            f.write(latex_content)
            temp_path = f.name
        
        try:
            chunks = self.processor.process_latex_file(temp_path)
            
            # Should produce some chunks
            assert len(chunks) > 0
            
            # Check chunk structure
            chunk = chunks[0]
            assert hasattr(chunk, 'id')
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'type')
            assert hasattr(chunk, 'chapter_key')
            
            # Check content processing
            assert any("introduction" in chunk.text.lower() for chunk in chunks)
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_process_nonexistent_file(self):
        """Test processing a non-existent file raises error."""
        with pytest.raises(LaTeXProcessingError):
            self.processor.process_latex_file("/nonexistent/path/file.tex")

    def test_chunk_dataclass(self):
        """Test Chunk dataclass functionality."""
        chunk = Chunk(
            id="test-id",
            text="Test text",
            type="body",
            page=1,
            chapter_key="1",
            chapter="Test",
            section_key="1.1",
            section="Test Section",
            subsection_key=None,
            subsection=None,
            thesis_part="Introduction",
            chunk_idx=0,
            chunk_total=1
        )
        
        assert chunk.id == "test-id"
        assert chunk.text == "Test text"
        assert chunk.chapter == "Test"

    def test_event_dataclass(self):
        """Test Event dataclass functionality."""
        event = Event("chapter", 0, 10, "Introduction", starred=True)
        
        assert event.kind == "chapter"
        assert event.start == 0
        assert event.end == 10
        assert event.content == "Introduction"
        assert event.starred is True

    def test_segment_dataclass(self):
        """Test Segment dataclass functionality."""
        segment = Segment(
            "body", "Test text", "1", "Introduction",
            "1.1", "Background", None, None
        )
        
        assert segment.type == "body"
        assert segment.text == "Test text"
        assert segment.chapter == "Introduction"