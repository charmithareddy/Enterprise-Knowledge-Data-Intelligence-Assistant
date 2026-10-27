"""
Document Ingestion & Multi-Format Processing Engine.
Extracts text from TXT, PDF, DOCX, and CSV files with policy-aware semantic chunking.
"""

from pathlib import Path
import re
from typing import List, Dict, Any, Union
import io
import pandas as pd
from pypdf import PdfReader
from docx import Document

from src.config.settings import DOCS_DIR, MAX_CHUNK_SIZE, CHUNK_OVERLAP

class DocumentManager:
    """Manages document loading, format conversion, and semantic policy chunking."""

    @staticmethod
    def read_file_content(file_source: Union[Path, Any], filename: str) -> str:
        """
        Extract text from file path or file-like uploaded buffer (Streamlit UploadedFile).
        Supports .txt, .pdf, .docx, .csv, and .md formats.
        """
        suffix = Path(filename).suffix.lower()

        # If it's a file path
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if suffix in [".txt", ".md"]:
                return path.read_text(encoding="utf-8", errors="replace").strip()
            elif suffix == ".pdf":
                reader = PdfReader(str(path))
                return "\n\n".join([page.extract_text() or "" for page in reader.pages]).strip()
            elif suffix == ".docx":
                doc = Document(str(path))
                text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
                return "\n\n".join(text_parts).strip()
            elif suffix == ".csv":
                df = pd.read_csv(str(path))
                return df.to_string(index=False)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        # If it's an uploaded file buffer (e.g. Streamlit UploadedFile)
        else:
            if suffix in [".txt", ".md"]:
                return file_source.getvalue().decode("utf-8", errors="replace").strip()
            elif suffix == ".pdf":
                reader = PdfReader(file_source)
                return "\n\n".join([page.extract_text() or "" for page in reader.pages]).strip()
            elif suffix == ".docx":
                doc = Document(file_source)
                text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
                for table in doc.tables:
                    for row in table.rows:
                        text_parts.append(" | ".join([cell.text.strip() for cell in row.cells]))
                return "\n\n".join(text_parts).strip()
            elif suffix == ".csv":
                df = pd.read_csv(file_source)
                return df.to_string(index=False)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def chunk_document(
        text: str,
        filename: str,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
    ) -> List[Dict[str, Any]]:
        """
        Chunk document while preserving numbered section headings where available.
        Splits lengthy sections with sliding window chunking.
        """
        if not text or not text.strip():
            return []

        # Matches headings like "1. REFUND POLICY", "2. ACCESS CONTROL", "1. EXECUTIVE SUMMARY"
        heading_pattern = re.compile(r"(?m)^\s*(\d+\.\s+[A-Z0-9 &()/-]+)\s*$")
        matches = list(heading_pattern.finditer(text))

        sections = []
        if matches:
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                section_heading = match.group(1).strip()
                sec_text = text[start:end].strip()
                if sec_text:
                    sections.append((section_heading, sec_text))
        else:
            sections.append((Path(filename).stem, text.strip()))

        chunks = []
        for section_heading, content in sections:
            if len(content) <= max_chunk_size:
                chunks.append({
                    "section": section_heading,
                    "content": content,
                    "source": filename,
                    "length": len(content)
                })
                continue

            # Paragraph or sliding window splitting
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            current = ""
            for p in paragraphs:
                candidate = (current + "\n\n" + p).strip() if current else p
                if len(candidate) > max_chunk_size and current:
                    chunks.append({
                        "section": section_heading,
                        "content": current,
                        "source": filename,
                        "length": len(current)
                    })
                    current = p
                else:
                    current = candidate

            if current:
                chunks.append({
                    "section": section_heading,
                    "content": current,
                    "source": filename,
                    "length": len(current)
                })

        return chunks

    @classmethod
    def load_preloaded_documents(cls, docs_dir: Path = DOCS_DIR) -> List[Dict[str, str]]:
        """Load all preloaded enterprise documents from data/documents."""
        docs = []
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            return docs

        for file_path in sorted(docs_path.glob("*.*")):
            if file_path.suffix.lower() in [".txt", ".pdf", ".docx", ".csv", ".md"]:
                try:
                    content = cls.read_file_content(file_path, file_path.name)
                    if content:
                        docs.append({
                            "name": file_path.name,
                            "text": content,
                            "path": str(file_path)
                        })
                except Exception as err:
                    print(f"Warning: Failed to load {file_path.name}: {err}")

        return docs
