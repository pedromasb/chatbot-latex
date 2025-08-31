"""LaTeX document processor for extracting and chunking thesis content."""

import os
import re
import pathlib
import uuid
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from ..utils.exceptions import LaTeXProcessingError
from ..utils.text_utils import TextUtils


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    type: str
    page: int
    chapter_key: Optional[str]
    chapter: Optional[str]
    section_key: Optional[str]
    section: Optional[str]
    subsection_key: Optional[str]
    subsection: Optional[str]
    thesis_part: Optional[str]
    chunk_idx: int
    chunk_total: int


@dataclass
class Event:
    """Represents a document event (heading, text, etc.)."""
    kind: str
    start: int
    end: int
    content: str
    starred: bool = False


@dataclass
class Segment:
    """Represents a document segment with hierarchical context."""
    type: str
    text: str
    chapter_key: Optional[str]
    chapter: Optional[str]
    section_key: Optional[str]
    section: Optional[str]
    subsection_key: Optional[str]
    subsection: Optional[str]


class LaTeXProcessor:
    """Processes LaTeX documents into structured chunks for embedding."""

    # Canonical titles mapping for standardization
    CANONICAL_TITLES = {
        "1": "General Introduction",
        "1.1": "M dwarfs and the substellar realm",
        "1.1.1": "M dwarfs",
        "1.1.2": "... and beyond",
        "1.2": "From stars to data: the Virtual Observatory",
        "1.3": "The age of artificial intelligence",
        "1.4": "Aims and objectives of the thesis",
        "2": "Ultracool Dwarfs in J-PLUS",
        "2.1": "J-PLUS",
        "2.2": "Methodology",
        "2.2.1": "Parallax-based selection",
        "2.2.2": "Proper motion-based selection",
        "2.2.3": "Photometry-based selection",
        "2.2.4": "VOSA filtering",
        "2.3": "Analysis",
        "2.3.1": "Temperatures and distances",
        "2.3.2": "Kinematics",
        "2.3.3": "Binarity",
        "2.4": "Known ultracool dwarfs",
        "2.4.1": "Recovered known UCDs",
        "2.4.2": "New candidate UCDs vs. previously known",
        "2.5": "Machine learning analysis",
        "2.5.1": "PCA cut",
        "2.5.2": "SVM model",
        "2.5.3": "Blind test",
        "2.6": "Detection of strong emission line emitters",
        "2.7": "Conclusions",
        "3": "Detection of Flaring M dwarfs with multi-filter Photometry",
        "3.1": "Observations",
        "3.1.1": "Sample selection",
        "3.1.2": "Observational details",
        "3.1.3": "Data reduction",
        "3.2": "Results and discussion",
        "3.2.1": "Reduced spectra",
        "3.2.2": "Light curve analysis",
        "3.3": "Planetary habitability",
        "3.4": "Conclusions",
        "4": "Autoencoders and Deep Transfer Learning in CARMENES",
        "4.1": "Context",
        "4.2": "Data",
        "4.3": "Methodology",
        "4.3.1": "Feature extraction using an autoencoder",
        "4.3.2": "Deep transfer learning",
        "4.3.3": "Stellar parameter estimation",
        "4.4": "Results and discussion",
        "4.4.1": "Stellar parameters analysis",
        "4.4.2": "Comparison with the literature",
        "4.5": "Conclusions",
        "5": "Characterisation of Ultracool Dwarfs with Deep Transfer Learning",
        "5.1": "Testbed environment with SpeX",
        "5.2": "Ultracool dwarf characterisation",
        "6": "General conclusions and future work",
        "6.1": "Summary of the Thesis",
        "6.2": "Future Directions",
        "DSA": "Data and Software Availability",
    }

    def __init__(self, chunk_size: int = 300, overlap: int = 60, keep_captions: bool = True):
        """
        Initialize the LaTeX processor.
        
        Args:
            chunk_size: Maximum words per chunk
            overlap: Word overlap between chunks
            keep_captions: Whether to preserve figure/table captions
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.keep_captions = keep_captions
        
        # Compile regex patterns
        self.include_cmd = re.compile(r'\\(?:input|include)\{([^}]+)\}')
        self.begin_doc = re.compile(r'\\begin\{document\}', re.IGNORECASE)
        self.end_doc = re.compile(r'\\end\{document\}', re.IGNORECASE)
        
        # Structure patterns
        self.abs_re = re.compile(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', re.DOTALL | re.IGNORECASE)
        self.res_re = re.compile(r'\\begin\{resumen\}(.*?)\\end\{resumen\}', re.DOTALL | re.IGNORECASE)
        self.ch_re = re.compile(r'\\chapter\*?\{(.*?)\}', re.DOTALL)
        self.sec_re = re.compile(r'\\section\*?\{(.*?)\}', re.DOTALL)
        self.sub_re = re.compile(r'\\subsection\*?\{(.*?)\}', re.DOTALL)
        
        # Caption patterns
        self.env_re = re.compile(r'\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}', re.DOTALL | re.IGNORECASE)
        self.cap_re = re.compile(r'\\caption\{(.*?)\}', re.DOTALL)
        self.cap_line = re.compile(r'\[\[CAPTION\]\]\s+(.*)$', re.MULTILINE)

    def process_latex_file(self, latex_file_path: str) -> List[Chunk]:
        """
        Process a LaTeX file and return structured chunks.
        
        Args:
            latex_file_path: Path to the main LaTeX file
            
        Returns:
            List of Chunk objects with embedded metadata
            
        Raises:
            LaTeXProcessingError: If processing fails
        """
        try:
            # Load and resolve includes
            full_text = self._read_tex_with_includes(latex_file_path)
            
            # Extract document body
            body = self._extract_document_body(full_text)
            
            # Preprocess
            clean_text = self._preprocess_text(body)
            
            # Parse structure
            events = self._parse_structure(clean_text)
            
            # Create segments
            segments = self._create_segments(events, clean_text)
            
            # Create chunks
            chunks = self._create_chunks(segments)
            
            return chunks
            
        except Exception as e:
            raise LaTeXProcessingError(f"Failed to process LaTeX file: {str(e)}") from e

    def save_chunks_to_jsonl(self, chunks: List[Chunk], output_path: str) -> None:
        """
        Save chunks to JSONL format.
        
        Args:
            chunks: List of Chunk objects
            output_path: Path for output JSONL file
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        except Exception as e:
            raise LaTeXProcessingError(f"Failed to save chunks: {str(e)}") from e

    def _read_file_text(self, path: str) -> str:
        """Read text file with error handling."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise LaTeXProcessingError(f"Cannot read file {path}: {str(e)}") from e

    def _resolve_path(self, base_dir: str, rel_path: str) -> Optional[str]:
        """Resolve relative path for includes."""
        p = pathlib.Path(base_dir) / rel_path
        if p.suffix != ".tex":
            if p.exists():
                return str(p)
            p2 = p.with_suffix(".tex")
            if p2.exists():
                return str(p2)
        else:
            if p.exists():
                return str(p)
        return None

    def _read_tex_with_includes(self, path: str, visited: Optional[set] = None) -> str:
        """Recursively read LaTeX file and resolve includes."""
        if visited is None:
            visited = set()
            
        path = str(pathlib.Path(path).resolve())
        if path in visited:
            return ""
            
        visited.add(path)
        base_dir = str(pathlib.Path(path).parent)
        text = self._read_file_text(path)

        def repl(match):
            rel_path = match.group(1)
            resolved_path = self._resolve_path(base_dir, rel_path)
            if resolved_path:
                return self._read_tex_with_includes(resolved_path, visited)
            return ""  # Skip missing includes

        return self.include_cmd.sub(repl, text)

    def _extract_document_body(self, full_text: str) -> str:
        """Extract content between \\begin{document} and \\end{document}."""
        begin_match = self.begin_doc.search(full_text)
        end_match = self.end_doc.search(full_text)
        
        if begin_match and end_match and end_match.start() > begin_match.end():
            return full_text[begin_match.end():end_match.start()]
        
        return full_text

    def _preprocess_text(self, text: str) -> str:
        """Preprocess LaTeX text: remove comments, bibliography, keep captions."""
        # Strip comments
        text = re.sub(r'(?m)^[ \t]*%.*$', '', text)
        
        # Remove bibliography
        text = re.sub(r'\\bibliography\{[^}]*\}', '', text)
        text = re.sub(r'\\bibliographystyle\{[^}]*\}', '', text)
        text = re.sub(r'\\printbibliography\b.*', '', text)
        text = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Handle captions
        if self.keep_captions:
            text = self._keep_captions_only(text)
        else:
            text = self.env_re.sub('', text)
            
        return text

    def _keep_captions_only(self, text: str) -> str:
        """Extract only captions from figure/table environments."""
        def repl(match):
            inner = match.group(2)
            caps = self.cap_re.findall(inner)
            if not caps:
                return ""
            
            texts = []
            for cap in caps:
                clean_cap = re.sub(r'\s+', ' ', cap).strip()
                if clean_cap:
                    texts.append(f"[[CAPTION]] {clean_cap}")
            
            return "\n".join(texts) + "\n"
        
        return self.env_re.sub(repl, text)

    def _parse_structure(self, text: str) -> List[Event]:
        """Parse document structure into events."""
        events = []
        
        # Abstract/Resumen
        for match in self.abs_re.finditer(text):
            content = re.sub(r'\s+', ' ', match.group(1)).strip()
            events.append(Event('abs', match.start(), match.end(), content))
            
        for match in self.res_re.finditer(text):
            content = re.sub(r'\s+', ' ', match.group(1)).strip()
            events.append(Event('res', match.start(), match.end(), content))
        
        # Headings
        events.extend(self._find_headings(text, self.ch_re, 'chapter'))
        events.extend(self._find_headings(text, self.sec_re, 'section'))
        events.extend(self._find_headings(text, self.sub_re, 'subsection'))
        
        # Captions
        for match in self.cap_line.finditer(text):
            content = match.group(1).strip()
            events.append(Event('caption', match.start(), match.end(), content))
        
        # Sort events by position
        events.sort(key=lambda e: e.start)
        
        # Merge with text content
        merged = []
        last_end = 0
        
        for event in events:
            if event.start > last_end:
                txt = text[last_end:event.start]
                if txt.strip():
                    merged.append(Event('text', last_end, event.start, txt))
            
            merged.append(event)
            last_end = event.end
        
        if last_end < len(text):
            tail = text[last_end:]
            if tail.strip():
                merged.append(Event('text', last_end, len(text), tail))
        
        return merged

    def _find_headings(self, text: str, regex: re.Pattern, kind: str) -> List[Event]:
        """Find headings of a specific type."""
        events = []
        for match in regex.finditer(text):
            content = re.sub(r'\s+', ' ', match.group(1)).strip()
            starred = text[match.start():match.end()].startswith(f"\\{kind}*")
            events.append(Event(kind, match.start(), match.end(), content, starred))
        return events

    def _create_segments(self, events: List[Event], text: str) -> List[Segment]:
        """Create segments with hierarchical context."""
        segments = []
        
        # Context tracking
        ch = sec = sub = 0
        ctx = {
            "chapter_key": None, "chapter": None,
            "section_key": None, "section": None,
            "subsection_key": None, "subsection": None
        }
        
        for event in events:
            if event.kind in ("abs", "res"):
                seg_text = TextUtils.latex_to_text(event.content)
                if seg_text:
                    seg_type = "abstract" if event.kind == "abs" else "resumen"
                    segments.append(Segment(
                        seg_type, seg_text, "FRONT", "Front Matter",
                        event.kind.upper(), seg_type.capitalize(), None, None
                    ))
                    
            elif event.kind == "chapter":
                title = TextUtils.latex_to_text(event.content)
                if event.starred and self._is_dsa_title(title):
                    # Special starred DSA chapter
                    ctx.update({
                        "chapter_key": "DSA", "chapter": self.CANONICAL_TITLES["DSA"],
                        "section_key": None, "section": None,
                        "subsection_key": None, "subsection": None
                    })
                    continue
                    
                ch += 1
                sec = sub = 0
                ck = str(ch)
                ctx.update({
                    "chapter_key": ck, "chapter": self._canonical_title(ck, title),
                    "section_key": None, "section": None,
                    "subsection_key": None, "subsection": None
                })
                
            elif event.kind == "section":
                if ctx["chapter_key"] is None:
                    ch += 1
                    ctx["chapter_key"] = str(ch)
                    ctx["chapter"] = self._canonical_title(ctx["chapter_key"], f"Chapter {ch}")
                    
                sec += 1
                sub = 0
                sk = f"{ctx['chapter_key']}.{sec}"
                title = TextUtils.latex_to_text(event.content)
                ctx.update({
                    "section_key": sk, "section": self._canonical_title(sk, title),
                    "subsection_key": None, "subsection": None
                })
                
            elif event.kind == "subsection":
                if ctx["section_key"] is None:
                    sec += 1
                    ctx["section_key"] = f"{ctx['chapter_key']}.{sec}"
                    ctx["section"] = self._canonical_title(ctx["section_key"], f"Section {ctx['section_key']}")
                    
                sub += 1
                sbk = f"{ctx['chapter_key']}.{sec}.{sub}"
                title = TextUtils.latex_to_text(event.content)
                ctx.update({
                    "subsection_key": sbk, "subsection": self._canonical_title(sbk, title)
                })
                
            elif event.kind == "caption" and self.keep_captions:
                cap_text = TextUtils.latex_to_text(event.content)
                if cap_text:
                    segments.append(Segment(
                        "caption", cap_text,
                        ctx["chapter_key"], ctx["chapter"],
                        ctx["section_key"], ctx["section"],
                        ctx["subsection_key"], ctx["subsection"]
                    ))
                    
            elif event.kind == "text":
                body_text = TextUtils.latex_to_text(event.content)
                if body_text:
                    segments.append(Segment(
                        "body", body_text,
                        ctx["chapter_key"], ctx["chapter"],
                        ctx["section_key"], ctx["section"],
                        ctx["subsection_key"], ctx["subsection"]
                    ))
        
        return segments

    def _create_chunks(self, segments: List[Segment]) -> List[Chunk]:
        """Create chunks from segments using sliding window."""
        chunks = []
        
        for segment in segments:
            if not segment.text.strip():
                continue
                
            windows = self._sliding_windows(segment.text.strip())
            total = len(windows)
            
            for i, window in enumerate(windows):
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    text=window,
                    type=segment.type,
                    page=-1,
                    chapter_key=segment.chapter_key,
                    chapter=segment.chapter,
                    section_key=segment.section_key,
                    section=segment.section,
                    subsection_key=segment.subsection_key,
                    subsection=segment.subsection,
                    thesis_part=self._thesis_part_from_ctx(segment.chapter_key, segment.section),
                    chunk_idx=i,
                    chunk_total=total
                )
                chunks.append(chunk)
        
        return chunks

    def _sliding_windows(self, paragraph: str) -> List[str]:
        """Create sliding windows from text."""
        words = paragraph.split()
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [paragraph]
            
        windows = []
        start = 0
        
        while start < len(words):
            end = min(len(words), start + self.chunk_size)
            windows.append(" ".join(words[start:end]))
            
            if end == len(words):
                break
                
            start = max(0, end - self.overlap)
        
        return windows

    def _canonical_title(self, key: str, fallback: str) -> str:
        """Get canonical title or fallback."""
        return self.CANONICAL_TITLES.get(key, fallback)

    def _is_dsa_title(self, title: str) -> bool:
        """Check if title is Data and Software Availability."""
        title_lower = title.lower().strip()
        return "data and software availability" in title_lower or "data & software availability" in title_lower

    def _thesis_part_from_ctx(self, chapter_key: Optional[str], section_title: Optional[str]) -> Optional[str]:
        """Determine thesis part from context."""
        if chapter_key == "1":
            return "Introduction"
        if chapter_key in {"2", "3", "4", "5"}:
            return "Conclusions" if (section_title and "conclusion" in section_title.lower()) else "Methods/Results"
        if chapter_key == "6":
            return "Conclusions"
        if chapter_key == "DSA":
            return "Data & Software"
        if chapter_key == "FRONT":
            return "Front Matter"
        return None